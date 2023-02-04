//===- Builder.cpp - Testing bytecode reader ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>
#include <stack>
#include <string>

// Parsing inc configuration.
typedef mlir::OperationState MlirBytecodeOperationState;
typedef mlir::Operation MlirBytecodeOperation;
// Include bytecode parsing implementation.
#include "mlir/Bytecode/BytecodeTypes.h"
#include "mlir/Bytecode/Parse.c.inc"
// Dialect and attribute parsing helpers.
#include "mlir/Bytecode/DialectBytecodeReader.c.inc"

#define DEBUG_TYPE "mlir-bytecode-reader"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BytecodeDialect
//===----------------------------------------------------------------------===//

namespace {

struct ParsingState;

/// This struct represents a dialect entry within the bytecode.
struct BytecodeDialect {
  /// Load the dialect into the provided context if it hasn't been loaded yet.
  /// Returns failure if the dialect couldn't be loaded *and* the provided
  /// context does not allow unregistered dialects. The provided reader is used
  /// for error emission if necessary.
  LogicalResult load(ParsingState &state, MLIRContext *ctx);

  /// Return the loaded dialect, or nullptr if the dialect is unknown. This can
  /// only be called after `load`.
  Dialect *getLoadedDialect() const {
    assert(dialect &&
           "expected `load` to be invoked before `getLoadedDialect`");
    return *dialect;
  }

  /// The loaded dialect entry. This field is std::nullopt if we haven't
  /// attempted to load, nullptr if we failed to load, otherwise the loaded
  /// dialect.
  std::optional<Dialect *> dialect;

  /// The bytecode interface of the dialect, or nullptr if the dialect does not
  /// implement the bytecode interface. This field should only be checked if the
  /// `dialect` field is not std::nullopt.
  const BytecodeDialectInterface *interface = nullptr;

  /// The name of the dialect.
  StringRef name;
};

/// Range corresponding to Attribute or Type.
struct MlirBytecodeAttributeOrTypeRange {
  MlirBytecodeBytesRef bytes;
  MlirBytecodeDialectHandle dialectHandle;
  bool hasCustom;
};

/// Represent either range in file or materialized Attribute.
struct BytecodeAttribute {
  BytecodeAttribute() : range(), value(nullptr) {}
  BytecodeAttribute(MlirBytecodeAttributeOrTypeRange range)
      : range(std::move(range)), value(nullptr) {}

  MlirBytecodeAttributeOrTypeRange range;
  Attribute value;
};

/// Represent either range in file or materialized Type.
struct BytecodeType {
  BytecodeType() : range(), value(nullptr) {}
  BytecodeType(MlirBytecodeAttributeOrTypeRange range)
      : range(std::move(range)), value(nullptr) {}

  MlirBytecodeAttributeOrTypeRange range;
  Type value;
};

/// This struct represents an operation name entry within the bytecode.
struct BytecodeOperationName {
  BytecodeOperationName(BytecodeDialect *dialect, StringRef name)
      : dialect(dialect), name(name) {}

  /// The loaded operation name, or std::nullopt if it hasn't been processed
  /// yet.
  std::optional<OperationName> opName;

  /// The dialect that owns this operation name.
  BytecodeDialect *dialect;

  /// The name of the operation, without the dialect prefix.
  StringRef name;
};

/// This struct represents the current read state of a range of regions. This
/// struct is used to enable iterative parsing of regions.
struct RegionReadState {
  RegionReadState(Operation *op, bool isIsolatedFromAbove)
      : RegionReadState(op->getRegions(), isIsolatedFromAbove) {}
  RegionReadState(MutableArrayRef<Region> regions, bool isIsolatedFromAbove)
      : curRegion(regions.begin()), endRegion(regions.end()),
        isIsolatedFromAbove(isIsolatedFromAbove) {}

  /// The current regions being read.
  MutableArrayRef<Region>::iterator curRegion, endRegion;

  /// The number of values defined immediately within this region.
  unsigned numValues = 0;

  /// The current blocks of the region being read.
  SmallVector<Block *> curBlocks;
  Region::iterator curBlock = {};

  /// A flag indicating if the regions being read are isolated from above.
  bool isIsolatedFromAbove = false;
};

/// This class represents a single value scope, in which a value scope is
/// delimited by isolated from above regions.
struct ValueScope {
  /// Push a new region state onto this scope, reserving enough values for
  /// those defined within the current region of the provided state.
  void push(RegionReadState &readState) {
    nextValueIDs.push_back(values.size());
    values.resize(values.size() + readState.numValues);
  }

  /// Pop the values defined for the current region within the provided region
  /// state.
  void pop(RegionReadState &readState) {
    values.resize(values.size() - readState.numValues);
    nextValueIDs.pop_back();
  }

  /// The set of values defined in this scope.
  std::vector<Value> values;

  /// The ID for the next defined value for each region current being
  /// processed in this scope.
  SmallVector<unsigned, 4> nextValueIDs;
};

struct ParsingState {
  ParsingState(Location fileLoc, const ParserConfig &config,
               const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
      : config(config), fileLoc(fileLoc),
        pendingOperationState(fileLoc, "builtin.unrealized_conversion_cast"),
        // Use the builtin unrealized conversion cast operation to represent
        // forward references to values that aren't yet defined.
        forwardRefOpState(UnknownLoc::get(fileLoc.getContext()),
                          "builtin.unrealized_conversion_cast", ValueRange(),
                          NoneType::get(fileLoc.getContext())),
        bufferOwnerRef(bufferOwnerRef) {}

  InFlightDiagnostic emitError(const Twine &msg = {}) {
    return ::emitError(fileLoc, msg);
  }

  Attribute attribute(MlirBytecodeAttrHandle handle) {
    uint64_t i = handle.id;
    if (i >= attributes.size())
      return nullptr;
    if (attributes[i].value)
      return attributes[i].value;
    if (!mlirBytecodeSucceeded(mlirBytecodeParseAttribute(
            this, (MlirBytecodeAttrHandle){.id = i})))
      return nullptr;
    return attributes[i].value;
  }

  FailureOr<Dialect *> dialect(MlirBytecodeDialectHandle handle) {
    if (handle.id >= dialects.size())
      return failure();
    BytecodeDialect &entry = dialects[handle.id];
    if (entry.dialect)
      return *entry.dialect;
    if (failed(entry.load(*this, getContext())))
      return failure();
    return *entry.dialect;
  }

  FailureOr<OperationName> opName(MlirBytecodeOpHandle handle) {
    if (handle.id >= opNames.size())
      return failure();
    BytecodeOperationName &entry = opNames[handle.id];
    if (entry.opName)
      return *entry.opName;
    if (failed(entry.dialect->load(*this, getContext())))
      return failure();
    entry.opName = {(entry.dialect->name + "." + entry.name).str(),
                    fileLoc.getContext()};
    return *entry.opName;
  }

  FailureOr<StringRef> string(MlirBytecodeStringHandle handle) {
    if (handle.id >= strings.size())
      return failure();
    return strings[handle.id];
  }

  Type type(MlirBytecodeTypeHandle handle) {
    uint64_t i = handle.id;
    if (i >= types.size())
      return nullptr;
    if (types[i].value)
      return types[i].value;
    if (!mlirBytecodeSucceeded(
            mlirBytecodeParseType(this, (MlirBytecodeTypeHandle){.id = i})))
      return nullptr;
    return types[i].value;
  }

  MLIRContext *getContext() const { return fileLoc->getContext(); }

  // Cached parsed entries.
  std::vector<AsmDialectResourceHandle> dialectResources;
  std::vector<BytecodeAttribute> attributes;
  std::vector<BytecodeDialect> dialects;
  std::vector<BytecodeOperationName> opNames;
  std::vector<BytecodeType> types;
  std::vector<StringRef> strings;

  /// The configuration of the parser.
  const ParserConfig &config;

  /// The resource parser to use for the current resource group.
  std::function<LogicalResult(AsmParsedResourceEntry &)> resourceHandler;

  /// Location to use for reporting errors.
  Location fileLoc;

  /// Final destination Block
  Block *dest;
  // Temporary top-level operations to parse into.
  OwningOpRef<ModuleOp> moduleOp;

  /// Nested regions of operations being parsed.
  std::vector<RegionReadState> regionStack;

  /// OperationState used to construct the current operation.
  OperationState pendingOperationState;
  /// A flag indicating if the pending operation is isolated from above.
  bool isIsolatedFromAbove = false;

  /// The current set of available IR value scopes.
  std::vector<ValueScope> valueScopes;
  /// A block containing the set of operations defined to create forward
  /// references.
  Block forwardRefOps;
  /// A block containing previously created, and no longer used, forward
  /// reference operations.
  Block openForwardRefOps;
  /// An operation state used when instantiating forward references.
  OperationState forwardRefOpState;

  /// The optional owning source manager, which when present may be used to
  /// extend the lifetime of the input buffer.
  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
};

LogicalResult BytecodeDialect::load(ParsingState &state, MLIRContext *ctx) {
  if (dialect)
    return success();
  Dialect *loadedDialect = ctx->getOrLoadDialect(name);
  if (!loadedDialect && false) { // !ctx->allowsUnregisteredDialects()) {
    return state.emitError("dialect '")
           << name
           << "' is unknown; if this is intended, please call "
              "allowUnregisteredDialects() on the MLIRContext, or use "
              "-allow-unregistered-dialect with the MLIR tool used";
  }
  dialect = loadedDialect;

  // If the dialect was actually loaded, check to see if it has a bytecode
  // interface.
  if (loadedDialect)
    interface = dyn_cast<BytecodeDialectInterface>(loadedDialect);
  return success();
}

//===----------------------------------------------------------------------===//
// Value Processing

Value createForwardRef(ParsingState &state) {
  // Check for an avaliable existing operation to use. Otherwise, create a new
  // fake operation to use for the reference.
  if (!state.openForwardRefOps.empty()) {
    Operation *op = &state.openForwardRefOps.back();
    op->moveBefore(&state.forwardRefOps, state.forwardRefOps.end());
  } else {
    state.forwardRefOps.push_back(Operation::create(state.forwardRefOpState));
  }
  return state.forwardRefOps.back().getResult(0);
}

LogicalResult defineValues(ParsingState &state, ValueRange newValues) {
  ValueScope &valueScope = state.valueScopes.back();
  std::vector<Value> &values = valueScope.values;

  unsigned &valueID = valueScope.nextValueIDs.back();
  unsigned valueIDEnd = valueID + newValues.size();
  if (valueIDEnd > values.size()) {
    return state.emitError(
               "value index range was outside of the expected range for "
               "the parent region, got [")
           << valueID << ", " << valueIDEnd << "), but the maximum index was "
           << (values.size() - 1);
  }

  // Assign the values and update any forward references.
  for (unsigned i = 0, e = newValues.size(); i != e; ++i, ++valueID) {
    Value newValue = newValues[i];

    // Check to see if a definition for this value already exists.
    if (Value oldValue = std::exchange(values[valueID], newValue)) {
      Operation *forwardRefOp = oldValue.getDefiningOp();

      // Assert that this is a forward reference operation. Given how we compute
      // definition ids (incrementally as we parse), it shouldn't be possible
      // for the value to be defined any other way.
      assert(forwardRefOp && forwardRefOp->getBlock() == &state.forwardRefOps &&
             "value index was already defined?");

      oldValue.replaceAllUsesWith(newValue);
      forwardRefOp->moveBefore(&state.openForwardRefOps,
                               state.openForwardRefOps.end());
    }
  }
  return success();
}

Value parseOperand(ParsingState &state, uint64_t i) {
  std::vector<Value> &values = state.valueScopes.back().values;
  Value &value = values[i];
  // Create a new forward reference if necessary.
  if (!value)
    value = createForwardRef(state);
  return value;
}

} // namespace

#ifdef MLIRBC_VERBOSE_ERROR
static MlirBytecodeStatus mlirBytecodeEmitErrorImpl(void *context,
                                                    const char *fmt, ...) {
  ParsingState &state = *(ParsingState *)context;
  const int kLimit = 300;
  auto msg = std::make_unique<char[]>(kLimit);
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg.get(), kLimit, fmt, args);
  va_end(args);
  state.emitError(msg.get());
  return mlirBytecodeFailure();
}
#endif

/// Wrapper around DialectBytecodeReader invoking C MlirBytecode API.
struct MlirBytecodeDialectBytecodeReader : public mlir::DialectBytecodeReader {
  MlirBytecodeDialectBytecodeReader(ParsingState &state,
                                    MlirBytecodeStream &stream)
      : reader(
            (MlirBytecodeDialectReader){.context = &state, .stream = &stream}),
        state(state){};

  InFlightDiagnostic emitError(const Twine &msg = {}) final;
  LogicalResult readAttribute(Attribute &result) final;
  LogicalResult readType(Type &result) final;
  LogicalResult readVarInt(uint64_t &result) final;
  LogicalResult readSignedVarInt(int64_t &result) final;
  FailureOr<APInt> readAPIntWithKnownWidth(unsigned bitWidth) final;
  FailureOr<APFloat>
  readAPFloatWithKnownSemantics(const llvm::fltSemantics &semantics) final;
  LogicalResult readString(StringRef &result) final;
  LogicalResult readBlob(ArrayRef<char> &result) final;
  FailureOr<AsmDialectResourceHandle> readResourceHandle() final;

  MlirBytecodeDialectReader reader;
  ParsingState &state;
};

InFlightDiagnostic
MlirBytecodeDialectBytecodeReader::emitError(const Twine &msg) {
  return state.emitError(msg);
}

LogicalResult
MlirBytecodeDialectBytecodeReader::readAttribute(Attribute &result) {
  MlirBytecodeAttrHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadAttribute(&reader, &handle)))
    return failure();
  result = state.attribute(handle);
  return success(result);
}

LogicalResult MlirBytecodeDialectBytecodeReader::readType(Type &result) {
  MlirBytecodeTypeHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadType(&reader, &handle)))
    return failure();
  FailureOr<Type> type = state.type(handle);
  if (failed(type))
    return failure();
  result = *type;
  return success();
}

LogicalResult MlirBytecodeDialectBytecodeReader::readVarInt(uint64_t &result) {
  return failure(!mlirBytecodeSucceeded(
      mlirBytecodeDialectReaderReadVarInt(&reader, &result)));
}

LogicalResult
MlirBytecodeDialectBytecodeReader::readSignedVarInt(int64_t &result) {
  return failure(!mlirBytecodeSucceeded(
      mlirBytecodeDialectReaderReadSignedVarInt(&reader, &result)));
}

FailureOr<APInt>
MlirBytecodeDialectBytecodeReader::readAPIntWithKnownWidth(unsigned bitWidth) {
  MlirBytecodeAPInt result;
  // TODO: This could be improved to not malloc and free for each large APInt.
  MlirBytecodeStatus ret = mlirBytecodeDialectReaderReadAPIntWithKnownWidth(
      &reader, bitWidth, malloc, &result);
  if (!mlirBytecodeSucceeded(ret))
    return failure();
  if (result.bitWidth <= 64)
    return APInt(result.bitWidth, result.U.value);

  const uint64_t bitsPerWord = sizeof(uint64_t) * CHAR_BIT;
  uint64_t numWords = ((uint64_t)bitWidth + bitsPerWord - 1) / bitsPerWord;
  APInt retVal(bitWidth, ArrayRef(result.U.data, numWords));
  free(result.U.data);
  return retVal;
}

FailureOr<APFloat>
MlirBytecodeDialectBytecodeReader::readAPFloatWithKnownSemantics(
    const llvm::fltSemantics &semantics) {
  FailureOr<APInt> intVal =
      readAPIntWithKnownWidth(APFloat::getSizeInBits(semantics));
  if (failed(intVal))
    return failure();
  return APFloat(semantics, *intVal);
}

LogicalResult MlirBytecodeDialectBytecodeReader::readString(StringRef &result) {
  MlirBytecodeBytesRef ref;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadString(&reader, &ref)))
    return failure();
  result = StringRef((const char *)ref.data, ref.length);
  return success();
}

LogicalResult
MlirBytecodeDialectBytecodeReader::readBlob(ArrayRef<char> &result) {
  MlirBytecodeBytesRef ref;
  if (!mlirBytecodeSucceeded(mlirBytecodeDialectReaderReadBlob(&reader, &ref)))
    return failure();
  result = ArrayRef((const char *)ref.data, ref.length);
  return success();
}

FailureOr<AsmDialectResourceHandle>
MlirBytecodeDialectBytecodeReader::readResourceHandle() {
  MlirBytecodeResourceHandle handle;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeDialectReaderReadResourceHandle(&reader, &handle)))
    return failure();
  if (handle.id >= state.dialectResources.size())
    return failure();
  return state.dialectResources[handle.id];
}

void mlirBytecodeIRSectionEnter(void *context, void *retBlock) {
  ParsingState &state = *(ParsingState *)context;
  state.dest = (Block *)retBlock;
  state.moduleOp = ModuleOp::create(state.fileLoc);
  state.regionStack.emplace_back(*state.moduleOp, /*isIsolatedFromAbove=*/true);
  state.regionStack.back().curBlocks.push_back(state.moduleOp->getBody());
  state.regionStack.back().curBlock =
      state.regionStack.back().curRegion->begin();
  state.valueScopes.emplace_back();
  state.valueScopes.back().push(state.regionStack.back());
}

MlirBytecodeStatus
mlirBytecodeOperationStatePush(void *context, MlirBytecodeOpHandle opHandle,
                               MlirBytecodeLocHandle locHandle,
                               MlirBytecodeOperationStateHandle *opState) {
  ParsingState &state = *(ParsingState *)context;
  LocationAttr locAttr =
      dyn_cast_if_present<LocationAttr>(state.attribute(locHandle));
  if (!locAttr)
    return mlirBytecodeEmitError(context, "invalid operation location");
  FailureOr<OperationName> opName = state.opName(opHandle);
  if (failed(opName))
    return mlirBytecodeFailure();

  // Initialize pending operation's state.
  state.pendingOperationState.location = locAttr;
  state.pendingOperationState.name = *opName;
  // Clear currently unknown operation state.
  state.pendingOperationState.attributes = {};
  state.pendingOperationState.operands.clear();
  state.pendingOperationState.regions.clear();
  state.pendingOperationState.successors.clear();
  state.pendingOperationState.types.clear();

  *opState = &state.pendingOperationState;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeAttrHandle dictHandle) {
  ParsingState &state = *(ParsingState *)context;
  if (dictHandle.id >= state.attributes.size())
    return mlirBytecodeEmitError(context, "out of range attribute handle");

  OperationState &opState = *opStateHandle;
  DictionaryAttr attr =
      dyn_cast_if_present<DictionaryAttr>(state.attribute(dictHandle));
  if (!attr)
    return mlirBytecodeEmitError(context, "invalid dictionary attribute");
  opState.addAttributes(attr.getValue());
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSize numResults) {
  OperationState &opState = *opStateHandle;
  opState.types.clear();
  opState.types.reserve(numResults);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultType(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeTypeHandle type) {
  ParsingState &state = *(ParsingState *)context;
  OperationState &opState = *opStateHandle;

  FailureOr<Type> resultType = state.type(type);
  if (MLIRBC_UNLIKELY(failed(resultType)))
    return mlirBytecodeEmitError(context, "invalid result type");
  opState.types.push_back(*resultType);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSize numOperands) {
  OperationState &opState = *opStateHandle;
  opState.operands.clear();
  opState.operands.reserve(numOperands);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperand(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeValueHandle value) {
  ParsingState &state = *(ParsingState *)context;
  OperationState &opState = *opStateHandle;

  opState.operands.push_back(parseOperand(state, value.id));
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    uint64_t numRegions, bool isIsolatedFromAbove) {
  ParsingState &state = *(ParsingState *)context;
  OperationState &opState = *opStateHandle;
  opState.regions.reserve(numRegions);
  for (int i = 0, e = numRegions; i < e; ++i)
    opState.regions.push_back(std::make_unique<Region>());
  state.isIsolatedFromAbove = isIsolatedFromAbove;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSize numSuccessors) {
  OperationState &opState = *opStateHandle;
  opState.successors.clear();
  opState.successors.reserve(numSuccessors);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessor(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeHandle successor) {
  ParsingState &state = *(ParsingState *)context;
  OperationState &opState = *opStateHandle;

  auto &readState = state.regionStack.back();
  if (MLIRBC_UNLIKELY(successor.id >= readState.curBlocks.size()))
    return mlirBytecodeEmitError(context, "invalid successor index");
  opState.successors.push_back(readState.curBlocks[successor.id]);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *context,
                              MlirBytecodeOperationStateHandle opStateHandle,
                              MlirBytecodeOperationHandle *opHandle) {
  ParsingState &state = *(ParsingState *)context;
  OperationState &opState = *opStateHandle;

  // Create the operation at the back of the current block.
  Operation *op = Operation::create(opState);
  *opHandle = op;
  state.regionStack.back().curBlock->push_back(op);

  // If the operation had results, update the value references.
  if (op->getNumResults()) {
    auto ret = defineValues(state, op->getResults());
    if (MLIRBC_UNLIKELY(failed(ret)))
      return mlirBytecodeEmitError(context, "invalid operation results");
  }

  if (!opState.regions.empty()) {
    state.regionStack.emplace_back(op, state.isIsolatedFromAbove);

    // If the op is isolated from above, push a new value scope.
    if (state.isIsolatedFromAbove)
      state.valueScopes.emplace_back();
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPush(void *context,
                                MlirBytecodeOperationHandle opHandle,
                                size_t numBlocks, size_t numValues) {
  ParsingState &state = *(ParsingState *)context;

  // If the region is empty, there is nothing else to do.
  if (numBlocks == 0)
    return mlirBytecodeFailure();

  // Create the blocks within this region. We do this before processing so that
  // we can rely on the blocks existing when creating operations.
  auto &readState = state.regionStack.back();
  readState.curBlocks.clear();
  readState.curBlocks.reserve(numBlocks);
  for (uint64_t i = 0; i < numBlocks; ++i) {
    readState.curBlocks.push_back(new Block());
    readState.curRegion->push_back(readState.curBlocks.back());
  }
  readState.numValues = numValues;

  // Prepare the current value scope for this region.
  auto &valueScopes = state.valueScopes;
  valueScopes.back().push(readState);

  // Parse the entry block of the region.
  readState.curBlock = readState.curRegion->begin();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPush(void *context,
                               MlirBytecodeOperationHandle opHandle,
                               MlirBytecodeSize numArgs) {
  // TODO: Add method to pre-size numArgs.
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationBlockAddArgument(
    void *context, MlirBytecodeOperationHandle opHandle,
    MlirBytecodeTypeHandle type, MlirBytecodeLocHandle loc) {
  ParsingState &state = *(ParsingState *)context;
  Type t = state.type(type);
  Attribute locAttr = state.attribute(loc);
  if (MLIRBC_UNLIKELY(!t))
    return mlirBytecodeEmitError(context, "invalid type");
  if (MLIRBC_UNLIKELY(!locAttr))
    return mlirBytecodeEmitError(context, "invalid location");

  auto &readState = state.regionStack.back();
  if (failed(defineValues(state, readState.curBlock->addArgument(
                                     t, cast<LocationAttr>(locAttr)))))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationBlockPop(void *context,
                                                 MlirBytecodeOperationHandle) {
  ParsingState &state = *(ParsingState *)context;
  auto &readState = state.regionStack.back();
  ++readState.curBlock;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPop(void *context,
                               MlirBytecodeOperationHandle opHandle) {
  ParsingState &state = *(ParsingState *)context;
  auto &valueScopes = state.valueScopes;

  auto &readState = state.regionStack.back();
  valueScopes.back().pop(readState);
  readState.curBlock = {};
  ++readState.curRegion;

  // Pop barrier value scope for isolated from above op.
  if (readState.curRegion == readState.endRegion) {
    if (readState.isIsolatedFromAbove)
      valueScopes.pop_back();
    state.regionStack.pop_back();
  }

  if (state.regionStack.size() == 1) {
    // Verify that the parsed operations are valid.
    if (state.config.shouldVerifyAfterParse() &&
        failed(verify(*state.moduleOp)))
      return mlirBytecodeFailure();

    // Splice the parsed operations over to the provided top-level block.
    auto &parsedOps = state.moduleOp->getBody()->getOperations();
    auto &destOps = state.dest->getOperations();
    destOps.splice(destOps.end(), parsedOps, parsedOps.begin(),
                   parsedOps.end());
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseAttribute(void *context,
                                              MlirBytecodeAttrHandle handle) {
  ParsingState &state = *(ParsingState *)context;
  if (handle.id >= state.attributes.size())
    return mlirBytecodeEmitError(context,
                                 "invalid attribute id %" PRIu64 " / %" PRIu64,
                                 handle.id, state.attributes.size());
  BytecodeAttribute &attr = state.attributes[handle.id];
  if (attr.value)
    return mlirBytecodeSuccess();

  if (attr.range.dialectHandle.id >= state.dialects.size())
    return mlirBytecodeEmitError(
        context, "invalid dialect id %" PRIu64 " / %" PRIu64,
        attr.range.dialectHandle.id, state.dialects.size());

  auto &dialect = state.dialects[attr.range.dialectHandle.id];
  if (attr.range.hasCustom) {
    if (failed(dialect.load(state, state.fileLoc.getContext())))
      return mlirBytecodeFailure();

    // Ensure that the dialect implements the bytecode interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          context, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(attr.range.bytes);
    MlirBytecodeDialectBytecodeReader reader(state, stream);
    attr.value = dialect.interface->readAttribute(reader);
    if (!attr.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  auto asmStr = StringRef((const char *)attr.range.bytes.data,
                          attr.range.bytes.length - 1);
  // Invoke the MLIR assembly parser to parse the entry text.
  size_t numRead = 0;
  attr.value = ::parseAttribute(asmStr, state.getContext(), numRead);
  // Ensure there weren't dangling characters after the entry.
  if (numRead != asmStr.size()) {
    return mlirBytecodeEmitError(
        context,
        "trailing characters found after Attribute assembly format: %s",
        asmStr.drop_front(numRead).str().c_str());
  }

  return attr.value ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

MlirBytecodeStatus mlirBytecodeParseType(void *context,
                                         MlirBytecodeTypeHandle handle) {
  ParsingState &state = *(ParsingState *)context;
  if (handle.id >= state.types.size())
    return mlirBytecodeEmitError(context,
                                 "invalid type id %" PRIu64 " / %" PRIu64,
                                 handle.id, state.types.size());
  BytecodeType &type = state.types[handle.id];
  if (type.value)
    return mlirBytecodeSuccess();

  if (type.range.dialectHandle.id >= state.dialects.size())
    return mlirBytecodeEmitError(
        context, "invalid dialect id %" PRIu64 " / %" PRIu64,
        type.range.dialectHandle.id, state.dialects.size());

  auto &dialect = state.dialects[type.range.dialectHandle.id];
  if (type.range.hasCustom) {
    if (failed(dialect.load(state, state.fileLoc.getContext())))
      return mlirBytecodeFailure();

    // Ensure that the dialect implements the bytecode interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          context, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(type.range.bytes);
    MlirBytecodeDialectBytecodeReader reader(state, stream);
    type.value = dialect.interface->readType(reader);
    if (!type.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  auto asmStr = StringRef((const char *)type.range.bytes.data,
                          type.range.bytes.length - 1);
  // Invoke the MLIR assembly parser to parse the entry text.
  size_t numRead = 0;
  type.value = ::parseType(asmStr, state.getContext(), numRead);
  // Ensure there weren't dangling characters after the entry.
  if (numRead != asmStr.size()) {
    return mlirBytecodeEmitError(
        context, "trailing characters found after Type assembly format: %s",
        asmStr.drop_front(numRead).str().c_str());
  }

  return type.value ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

MlirBytecodeStatus mlirBytecodeAssociateAttributeRange(
    void *context, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeDialectHandle dialectHandle, MlirBytecodeBytesRef bytes,
    bool hasCustom) {
  ParsingState &state = *(ParsingState *)context;
  state.attributes[attrHandle.id].range = {
      .bytes = bytes, .dialectHandle = dialectHandle, .hasCustom = hasCustom};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAssociateTypeRange(void *context, MlirBytecodeTypeHandle typeHandle,
                               MlirBytecodeDialectHandle dialectHandle,
                               MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState &state = *(ParsingState *)context;
  state.types[typeHandle.id].range = {
      .bytes = bytes, .dialectHandle = dialectHandle, .hasCustom = hasCustom};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *context,
                            MlirBytecodeDialectHandle dialectHandle,
                            MlirBytecodeStringHandle stringHandle) {
  ParsingState &state = *(ParsingState *)context;
  if (dialectHandle.id >= state.dialects.size())
    return mlirBytecodeFailure();
  BytecodeDialect &dialect = state.dialects[dialectHandle.id];
  auto name = state.string(stringHandle);
  if (failed(name))
    return mlirBytecodeFailure();
  dialect.name = *name;
  mlirBytecodeEmitDebug("dialect[%d] = %s", (int)dialectHandle.id, *name);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectOpNames(void *context,
                                              MlirBytecodeSize numOps) {
  ParsingState &state = *(ParsingState *)context;
  state.opNames.reserve(numOps);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *context, MlirBytecodeOpHandle opHandle,
                              MlirBytecodeDialectHandle dialectHandle,
                              MlirBytecodeStringHandle strHandle) {
  ParsingState &state = *(ParsingState *)context;
  assert(state.opNames.size() == opHandle.id);

  if (dialectHandle.id >= state.dialects.size())
    return mlirBytecodeEmitError(context, "invalid dialect");
  BytecodeDialect *dialect = &state.dialects[dialectHandle.id];
  FailureOr<StringRef> name = state.string(strHandle);
  if (failed(name))
    return mlirBytecodeEmitError(context, "invalid op name");
  state.opNames.emplace_back(dialect, *name);
  return mlirBytecodeSuccess();
}

//===----------------------------------------------------------------------===//
// ResourceSectionReader
//===----------------------------------------------------------------------===//

namespace {
class ParsedResourceEntry : public AsmParsedResourceEntry {
public:
  ParsedResourceEntry(StringRef key, bool value, ParsingState &state)
      : key(key), kind(AsmResourceEntryKind::Bool), U(value), state(state) {}
  ParsedResourceEntry(StringRef key, MlirBytecodeStringHandle value,
                      ParsingState &state)
      : key(key), kind(AsmResourceEntryKind::String), U(value), state(state) {}
  ParsedResourceEntry(StringRef key, ArrayRef<uint8_t> data, int64_t alignment,
                      ParsingState &state)
      : key(key), kind(AsmResourceEntryKind::Blob),
        U(data, alignment, state.bufferOwnerRef), state(state) {}

  ~ParsedResourceEntry() override = default;

  StringRef getKey() const final { return key; }

  InFlightDiagnostic emitError() const final { return state.emitError(); }

  AsmResourceEntryKind getKind() const final { return kind; }

  FailureOr<bool> parseAsBool() const final {
    if (kind != AsmResourceEntryKind::Bool)
      return emitError() << "expected a bool resource entry, but found a "
                         << toString(kind) << " entry instead";
    return U.boolValue;
  }
  FailureOr<std::string> parseAsString() const final {
    if (kind != AsmResourceEntryKind::String)
      return emitError() << "expected a string resource entry, but found a "
                         << toString(kind) << " entry instead";
    return state.string(U.stringHandle);
  }

  FailureOr<AsmResourceBlob>
  parseAsBlob(BlobAllocatorFn allocator) const final {
    if (kind != AsmResourceEntryKind::Blob)
      return emitError() << "expected a blob resource entry, but found a "
                         << toString(kind) << " entry instead";

    // If we have an extendable reference to the buffer owner, we don't need to
    // allocate a new buffer for the data, and can use the data directly.
    if (U.blob.bufferOwnerRef) {
      ArrayRef<char> charData(
          reinterpret_cast<const char *>(U.blob.data.data()),
          U.blob.data.size());

      // Allocate an unmanaged buffer which captures a reference to the owner.
      // For now we just mark this as immutable, but in the future we should
      // explore marking this as mutable when desired.
      return UnmanagedAsmResourceBlob::allocateWithAlign(
          charData, U.blob.alignment,
          [bufferOwnerRef = U.blob.bufferOwnerRef](void *, size_t, size_t) {});
    }

    // Allocate memory for the blob using the provided allocator and copy the
    // data into it.
    AsmResourceBlob blob = allocator(U.blob.data.size(), U.blob.alignment);
    assert(llvm::isAddrAligned(llvm::Align(U.blob.alignment),
                               blob.getData().data()) &&
           blob.isMutable() &&
           "blob allocator did not return a properly aligned address");
    memcpy(blob.getMutableData().data(), U.blob.data.data(),
           U.blob.data.size());
    return blob;
  }

private:
  StringRef key;
  AsmResourceEntryKind kind;

  /// The union of possible resource values parsed.
  union ParsedResouce {
    ParsedResouce(bool value) : boolValue(value) {}
    ParsedResouce(ArrayRef<uint8_t> data, int64_t alignment,
                  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
        : blob(data, alignment, bufferOwnerRef) {}
    ParsedResouce(MlirBytecodeStringHandle value) : stringHandle(value) {}

    struct blob {
      blob(ArrayRef<uint8_t> data, int64_t alignment,
           const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
          : data(data), alignment(alignment), bufferOwnerRef(bufferOwnerRef) {}

      ArrayRef<uint8_t> data;
      int64_t alignment;
      const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
    } blob;
    MlirBytecodeStringHandle stringHandle;
    bool boolValue;
  } U;

  ParsingState &state;
};
} // namespace

MlirBytecodeStatus
mlirBytecodeResourceDialectGroupEnter(void *context,
                                      MlirBytecodeDialectHandle dialect,
                                      MlirBytecodeSize numResources) {
  mlirBytecodeEmitDebug("entering dialect resource group");
  ParsingState &state = *(ParsingState *)context;
  state.resourceHandler = nullptr;

  auto dialectOr = state.dialects[dialect.id];
  FailureOr<Dialect *> loadedDialect = state.dialect(dialect);
  if (MLIRBC_UNLIKELY(failed(loadedDialect))) {
    return mlirBytecodeEmitError(context, "dialect '%s' is unknown",
                                 dialectOr.name);
  }
  auto parser = dyn_cast<OpAsmDialectInterface>(*loadedDialect);
  if (!parser) {
    return mlirBytecodeEmitError(
        context, "unexpected resources for dialect '%s'", dialectOr.name);
  }

  state.resourceHandler =
      [&, parser](AsmParsedResourceEntry &entry) -> LogicalResult {
    StringRef key = entry.getKey();
    FailureOr<AsmDialectResourceHandle> handle = parser->declareResource(key);
    if (failed(handle)) {
      return state.emitError() << "unknown 'resource' key '" << key
                               << "' for dialect '" << dialectOr.name << "'";
    }
    state.dialectResources.push_back(*handle);
    return parser->parseResource(entry);
  };

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceExternalGroupEnter(void *context,
                                       MlirBytecodeStringHandle groupKey,
                                       MlirBytecodeSize numResources) {
  mlirBytecodeEmitDebug("entering external resource group");
  ParsingState &state = *(ParsingState *)context;
  state.resourceHandler = nullptr;
  FailureOr<StringRef> group = state.string(groupKey);
  if (failed(group))
    return mlirBytecodeEmitError(context, "invalid string index");

  AsmResourceParser *parser = state.config.getResourceParser(*group);
  if (parser) {
    state.resourceHandler = [parser](AsmParsedResourceEntry &entry) {
      return parser->parseResource(entry);
    };
  } else {
    emitWarning(state.fileLoc)
        << "ignoring unknown external resources for '" << *group << "'";
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *context, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeSize alignment, MlirBytecodeBytesRef blob) {
  ParsingState &state = *(ParsingState *)context;
  if (!state.resourceHandler)
    return mlirBytecodeUnhandled();
  auto keyOr = state.string(resourceKey);
  if (failed(keyOr))
    return mlirBytecodeFailure();
  ParsedResourceEntry entry(
      keyOr.value(),
      ArrayRef(static_cast<const uint8_t *>(blob.data), blob.length), alignment,
      state);
  auto ret = state.resourceHandler(entry);
  return succeeded(ret) ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *context, MlirBytecodeStringHandle resourceKey, const uint8_t value) {
  ParsingState &state = *(ParsingState *)context;
  if (!state.resourceHandler)
    return mlirBytecodeUnhandled();
  auto keyOr = state.string(resourceKey);
  if (failed(keyOr))
    return mlirBytecodeFailure();

  ParsedResourceEntry entry(keyOr.value(), value, state);
  auto ret = state.resourceHandler(entry);
  return succeeded(ret) ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

MlirBytecodeStatus
mlirBytecodeResourceStringCallBack(void *context,
                                   MlirBytecodeStringHandle resourceKey,
                                   MlirBytecodeStringHandle value) {
  ParsingState &state = *(ParsingState *)context;
  if (!state.resourceHandler)
    return mlirBytecodeUnhandled();
  auto keyOr = state.string(resourceKey);
  if (failed(keyOr))
    return mlirBytecodeFailure();
  ParsedResourceEntry entry(keyOr.value(), value, state);
  auto ret = state.resourceHandler(entry);
  return succeeded(ret) ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

MlirBytecodeStatus
mlirBytecodeGetStringSectionValue(void *context,
                                  MlirBytecodeStringHandle strHandle,
                                  MlirBytecodeBytesRef *result) {
  ParsingState &state = *(ParsingState *)context;
  auto str = state.string(strHandle);
  if (failed(str))
    return mlirBytecodeEmitError(context, "invalid string reference");
  result->data = (const uint8_t *)str->data();
  result->length = str->size();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeAttributesPush(void *context,
                                              MlirBytecodeSize numArgs) {
  ParsingState &state = *(ParsingState *)context;
  state.attributes.resize(numArgs);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectsPush(void *context,
                                            MlirBytecodeSize numDialects) {
  ParsingState &state = *(ParsingState *)context;
  state.dialects.resize(numDialects);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeStringsPush(void *context,
                                           MlirBytecodeSize numStrings) {
  ParsingState &state = *(ParsingState *)context;
  state.strings.resize(numStrings);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeTypesPush(void *context,
                                         MlirBytecodeSize numTypes) {
  ParsingState &state = *(ParsingState *)context;
  state.types.resize(numTypes);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAssociateStringRange(void *context, MlirBytecodeStringHandle handle,
                                 MlirBytecodeBytesRef bytes) {
  ParsingState &state = *(ParsingState *)context;
  if (handle.id >= state.strings.size())
    return mlirBytecodeFailure();

  state.strings[handle.id] = StringRef((const char *)bytes.data, bytes.length);
  return mlirBytecodeUnhandled();
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

bool mlir::isBytecode(llvm::MemoryBufferRef buffer) {
    return buffer.getBuffer().startswith("ML\xefR");
}

MlirBytecodeStatus
readBytecodeFileImpl(llvm::MemoryBufferRef buffer, Block *block,
                     const ParserConfig &config,
                     const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef) {
  Location sourceFileLoc =
      FileLineColLoc::get(config.getContext(), buffer.getBufferIdentifier(),
                          /*line=*/0, /*column=*/0);
  MlirBytecodeBytesRef ref = {.data =
                                  (const uint8_t *)buffer.getBufferStart(),
                              .length = buffer.getBufferSize()};
  MlirBytecodeStream stream = mlirBytecodeStreamCreate(ref);
  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&stream, ref);
  ParsingState state(sourceFileLoc, config, bufferOwnerRef);
  if (mlirBytecodeParserStateEmpty(&parserState))
  return mlirBytecodeSuccess();

  return mlirBytecodeParse(&state, &parserState, block);
}

LogicalResult mlir::readBytecodeFile(llvm::MemoryBufferRef buffer, Block *block,
                                     const ParserConfig &config) {
    return success(mlirBytecodeSucceeded(
        readBytecodeFileImpl(buffer, block, config, /*bufferOwnerRef=*/{})));
}
LogicalResult
mlir::readBytecodeFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                       Block *block, const ParserConfig &config) {
    return success(mlirBytecodeSucceeded(readBytecodeFileImpl(
        *sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID()), block, config,
        sourceMgr)));
}
