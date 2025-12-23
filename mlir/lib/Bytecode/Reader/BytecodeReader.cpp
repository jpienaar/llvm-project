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
#include "mlir/IR/DialectResourceBlobManager.h"
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
#include <list>
#include <optional>
#include <stack>
#include <string>

// Parsing inc configuration.
typedef mlir::OperationState MlirBytecodeOperationState;
typedef mlir::Operation MlirBytecodeOperation;
// Include bytecode parsing implementation.
#define MLIRBC_VERBOSE_ERROR  // Required for proper error diagnostics
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

  /// The parsed dialect version, if any. This is set by mlirBytecodeDialectVersionCallBack.
  std::unique_ptr<DialectVersion> version;
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

/// Storage for use-list order information parsed from bytecode.
struct UseListOrderStorage {
  UseListOrderStorage() = default;
  UseListOrderStorage(bool indexPairEncoding, SmallVector<unsigned> indices)
      : indexPairEncoding(indexPairEncoding), indices(std::move(indices)) {}

  /// Flag indicating if the indices are encoded as pairs (src, dst).
  bool indexPairEncoding = false;
  /// The use-list order indices.
  SmallVector<unsigned> indices;
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
  
  /// For lazy loading: byte range of the isolated region's IR section.
  ArrayRef<uint8_t> deferredIRData;
};

/// Storage for a lazily loadable operation and its deferred region data.
struct LazyLoadableOpInfo {
  Operation *op;
  RegionReadState regionState;
  ArrayRef<uint8_t> irData;
};

/// Type aliases for lazy loading tracking.
using LazyLoadableOpsInfo = std::list<LazyLoadableOpInfo>;
using LazyLoadableOpsMap = DenseMap<Operation *, LazyLoadableOpsInfo::iterator>;

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
    MlirBytecodeAttrHandle attrHandle;
    attrHandle.id = i;
    if (!mlirBytecodeSucceeded(mlirBytecodeParseAttribute(this, attrHandle)))
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
    // Handle dialect-less operations (unregistered ops without dialect prefix).
    // The bytecode format stores them with the full op name as the dialect name
    // and an empty op name (because stripDialect() returns empty when no dot).
    std::string fullName;
    if (entry.name.empty())
      fullName = entry.dialect->name.str();  // Dialect name IS the full op name
    else if (entry.dialect->name.empty())
      fullName = entry.name.str();  // Just op name (shouldn't normally happen)
    else
      fullName = (entry.dialect->name + "." + entry.name).str();
    entry.opName = {fullName, fileLoc.getContext()};
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
    MlirBytecodeTypeHandle typeHandle;
    typeHandle.id = i;
    if (!mlirBytecodeSucceeded(mlirBytecodeParseType(this, typeHandle)))
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

  /// Properties section data for v5+ bytecode.
  ArrayRef<uint8_t> propertiesSection;
  /// Offsets into the properties section.
  SmallVector<int64_t> propertiesOffsets;

  /// Use-list order storage: maps Value's opaque pointer to its use-list order.
  DenseMap<void *, UseListOrderStorage> valueToUseListMap;
  /// Operation IDs for use-list order computation, populated after parsing.
  DenseMap<Operation *, unsigned> operationIDs;

  /// The optional owning source manager, which when present may be used to
  /// extend the lifetime of the input buffer.
  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
  
  /// Lazy loading state.
  bool lazyLoading = false;
  llvm::function_ref<bool(Operation *)> lazyOpsCallback = nullptr;
  LazyLoadableOpsInfo lazyLoadableOps;
  LazyLoadableOpsMap lazyLoadableOpsMap;
  /// Stored parser state for lazy loading materialization.
  MlirBytecodeParserState storedParserState = {};
  /// Bytecode version for lazy loading check (>= 2 supports lazy loading).
  unsigned bytecodeVersion = 0;
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

// Temporary allocation for large use-lists (> 64 entries).
static void *mlirBytecodeAllocateTemp(void *context, size_t bytes) {
  return std::malloc(bytes);
}

static void mlirBytecodeFreeTemp(void *context, void *ptr) {
  std::free(ptr);
}

/// Wrapper around DialectBytecodeReader invoking C MlirBytecode API.
struct MlirBytecodeDialectBytecodeReader : public mlir::DialectBytecodeReader {
  MlirBytecodeDialectBytecodeReader(ParsingState &state,
                                    MlirBytecodeStream &stream)
      : reader({&state, &stream}),
        state(state){};

  InFlightDiagnostic emitError(const Twine &msg = {}) const final;
  FailureOr<const DialectVersion *>
  getDialectVersion(StringRef dialectName) const final;
  MLIRContext *getContext() const final;
  uint64_t getBytecodeVersion() const final;
  LogicalResult readAttribute(Attribute &result) final;
  LogicalResult readOptionalAttribute(Attribute &result) final;
  LogicalResult readType(Type &result) final;
  LogicalResult readVarInt(uint64_t &result) final;
  LogicalResult readSignedVarInt(int64_t &result) final;
  FailureOr<APInt> readAPIntWithKnownWidth(unsigned bitWidth) final;
  FailureOr<APFloat>
  readAPFloatWithKnownSemantics(const llvm::fltSemantics &semantics) final;
  LogicalResult readString(StringRef &result) final;
  LogicalResult readBlob(ArrayRef<char> &result) final;
  LogicalResult readBool(bool &result) final;
  FailureOr<AsmDialectResourceHandle> readResourceHandle() final;

  MlirBytecodeDialectReader reader;
  ParsingState &state;
};

InFlightDiagnostic
MlirBytecodeDialectBytecodeReader::emitError(const Twine &msg) const {
  return state.emitError(msg);
}

FailureOr<const DialectVersion *>
MlirBytecodeDialectBytecodeReader::getDialectVersion(StringRef dialectName) const {
  // Search for the dialect by name and return its version if available.
  for (const auto &dialect : state.dialects) {
    if (dialect.name == dialectName && dialect.version)
      return dialect.version.get();
  }
  return failure();
}

MLIRContext *MlirBytecodeDialectBytecodeReader::getContext() const {
  return state.getContext();
}

uint64_t MlirBytecodeDialectBytecodeReader::getBytecodeVersion() const {
  // Return current bytecode version
  return 6; // TODO: Get from parser state
}

LogicalResult
MlirBytecodeDialectBytecodeReader::readOptionalAttribute(Attribute &result) {
  // Optional attributes use VarIntWithFlag encoding:
  // - The value is the attribute index
  // - The flag indicates presence (true = present, false = absent)
  uint64_t attrIdx;
  bool flag;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeParseVarIntWithFlag(&state, reader.stream, &attrIdx, &flag)))
    return failure();
  
  if (!flag) {
    // Attribute is absent
    result = Attribute();
    return success();
  }
  
  // Look up the attribute by index
  MlirBytecodeAttrHandle handle = {attrIdx};
  result = state.attribute(handle);
  return success(result != nullptr);
}

LogicalResult
MlirBytecodeDialectBytecodeReader::readBool(bool &result) {
  // The writer uses emitByte() to write bool values, so we must read a single
  // byte here (NOT a varint). Using readVarInt would cause stream corruption
  // when subsequent bytes have the high bit set.
  uint8_t value;
  if (!mlirBytecodeSucceeded(
          mlirBytecodeParseByte(&state, reader.stream, &value)))
    return failure();
  result = value != 0;
  return success();
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

// Forward declaration for use-list processing.
static LogicalResult processUseLists(Operation *topLevelOp, ParsingState &state);

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
    return mlirBytecodeEmitError(context, "invalid operation name index: %" PRIu64,
                                  opHandle.id);

  // Recreate pending operation's state to properly reset properties.
  // Using destruction and placement-new to reset all fields including properties.
  state.pendingOperationState.~OperationState();
  new (&state.pendingOperationState) OperationState(locAttr, *opName);

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

  std::vector<Value> &values = state.valueScopes.back().values;
  if (MLIRBC_UNLIKELY(value.id >= values.size()))
    return mlirBytecodeEmitError(context, "invalid value index: %" PRIu64,
                                  value.id);
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
    return mlirBytecodeEmitError(context, "invalid successor index: %" PRIu64,
                                  successor.id);
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
    // Check if we should defer region parsing for lazy loading.
    // Conditions: lazy loading enabled, bytecode v2+, operation is isolated from above.
    // Callback returns false to indicate "don't materialize" (defer), 
    // or callback is null means defer everything.
    if (state.lazyLoading && state.bytecodeVersion >= 2 && state.isIsolatedFromAbove &&
        (!state.lazyOpsCallback || !state.lazyOpsCallback(op))) {
      // Track operation for lazy loading - regions will be parsed later.
      state.lazyLoadableOps.push_back({op, RegionReadState(op, true), {}});
      state.lazyLoadableOpsMap.try_emplace(
          op, std::prev(state.lazyLoadableOps.end()));
      // Don't push region state - regions stay empty until materialized.
      return mlirBytecodeSuccess();
    }
    
    state.regionStack.emplace_back(op, state.isIsolatedFromAbove);

    // If the op is isolated from above, push a new value scope.
    if (state.isIsolatedFromAbove)
      state.valueScopes.emplace_back();
  }

  return mlirBytecodeSuccess();
}

/// Check if the given operation was lazily loaded (regions deferred).
bool mlirBytecodeOperationWasLazyLoaded(void *context,
                                        MlirBytecodeOperationHandle opHandle) {
  ParsingState &state = *(ParsingState *)context;
  return state.lazyLoadableOpsMap.count(static_cast<Operation *>(opHandle));
}

/// Store deferred region IR data for later materialization.
void mlirBytecodeStoreDeferredRegionData(void *context,
                                          MlirBytecodeOperationHandle opHandle,
                                          const uint8_t *data, uint64_t length) {
  ParsingState &state = *(ParsingState *)context;
  auto it = state.lazyLoadableOpsMap.find(static_cast<Operation *>(opHandle));
  if (it != state.lazyLoadableOpsMap.end()) {
    it->second->irData = ArrayRef<uint8_t>(data, length);
  }
}

/// Get the number of regions in an operation.
uint64_t mlirBytecodeGetOperationNumRegions(MlirBytecodeOperationHandle opHandle) {
  return static_cast<Operation *>(opHandle)->getNumRegions();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPush(void *context,
                                MlirBytecodeOperationHandle opHandle,
                                size_t numBlocks, size_t numValues) {
  ParsingState &state = *(ParsingState *)context;

  // If the region is empty, there is nothing else to do.
  // Empty regions are valid (e.g., uninitialized llvm.mlir.global).
  if (numBlocks == 0)
    return mlirBytecodeSuccess();

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
  if (MLIRBC_UNLIKELY(!t))
    return mlirBytecodeEmitError(context, "invalid type");

  // Handle elided locations (v4+): UINT64_MAX indicates unknown location
  LocationAttr locAttr;
  if (loc.id == UINT64_MAX) {
    locAttr = UnknownLoc::get(state.getContext());
  } else {
    Attribute attr = state.attribute(loc);
    if (MLIRBC_UNLIKELY(!attr))
      return mlirBytecodeEmitError(context, "invalid location");
    locAttr = cast<LocationAttr>(attr);
  }

  auto &readState = state.regionStack.back();
  if (failed(defineValues(state, readState.curBlock->addArgument(t, locAttr))))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeBlockArgAddUseListOrder(
    void *context, uint64_t valueIndex, bool indexPairEncoding,
    const uint64_t *indices, uint64_t numIndices) {
  ParsingState &state = *(ParsingState *)context;
  
  // Get the current block's argument at the given index.
  auto &readState = state.regionStack.back();
  Block *curBlock = &*readState.curBlock;
  if (valueIndex >= curBlock->getNumArguments())
    return mlirBytecodeEmitError(context, "invalid block arg index for use-list");
  
  Value arg = curBlock->getArgument(valueIndex);
  
  // Store the use-list order.
  SmallVector<unsigned> indicesVec;
  indicesVec.reserve(numIndices);
  for (uint64_t i = 0; i < numIndices; ++i)
    indicesVec.push_back(static_cast<unsigned>(indices[i]));
  
  state.valueToUseListMap.try_emplace(arg.getAsOpaquePointer(),
                                       UseListOrderStorage(indexPairEncoding, std::move(indicesVec)));
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResultAddUseListOrder(
    void *context, MlirBytecodeOperationHandle opHandle, uint64_t resultIndex,
    bool indexPairEncoding, const uint64_t *indices, uint64_t numIndices) {
  ParsingState &state = *(ParsingState *)context;
  Operation *op = opHandle;
  if (!op || resultIndex >= op->getNumResults())
    return mlirBytecodeEmitError(context, "invalid result index for use-list");
  Value result = op->getResult(resultIndex);
  SmallVector<unsigned> indicesVec;
  indicesVec.reserve(numIndices);
  for (uint64_t i = 0; i < numIndices; ++i)
    indicesVec.push_back(static_cast<unsigned>(indices[i]));
  state.valueToUseListMap.try_emplace(result.getAsOpaquePointer(),
                                       UseListOrderStorage(indexPairEncoding, std::move(indicesVec)));
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
  
  // Only pop value scope if we actually pushed one.
  // Empty regions (0 blocks) don't call mlirBytecodeOperationRegionPush,
  // so no push happened and we must not pop.
  if (!readState.curBlocks.empty())
    valueScopes.back().pop(readState);
  readState.curBlock = {};
  ++readState.curRegion;

  // Pop barrier value scope for isolated from above op.
  if (readState.curRegion == readState.endRegion) {
    if (readState.isIsolatedFromAbove)
      valueScopes.pop_back();
    state.regionStack.pop_back();
  }

  // Finalization is now handled in readBytecodeFileImpl after mlirBytecodeParse returns.

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

    // Try BytecodeReaderConfig attribute callbacks first.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(attr.range.bytes);
    MlirBytecodeDialectBytecodeReader reader(state, stream);
    for (auto &callback : state.config.getBytecodeReaderConfig().getAttributeCallbacks()) {
      Attribute result;
      LogicalResult callbackResult = callback->read(reader, dialect.name, result);
      if (succeeded(callbackResult)) {
        if (result) {
          attr.value = result;
          return mlirBytecodeSuccess();
        }
      } else {
        // Callback explicitly returned failure - do not fall through to dialect
        return mlirBytecodeFailure();
      }
    }

    // Fall back to dialect interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          context, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    stream = mlirBytecodeStreamCreate(attr.range.bytes);
    MlirBytecodeDialectBytecodeReader dialectReader(state, stream);
    attr.value = dialect.interface->readAttribute(dialectReader);
    if (!attr.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  auto asmStr = StringRef((const char *)attr.range.bytes.data,
                          attr.range.bytes.length - 1);
  // Invoke the MLIR assembly parser to parse the entry text.
  // Pass numRead pointer to get actual bytes consumed.
  size_t numRead = 0;
  Type unusedType;
  attr.value = ::parseAttribute(asmStr, state.getContext(), unusedType, &numRead);
  
  // Ensure there weren't dangling characters after the entry.
  if (attr.value && numRead != asmStr.size()) {
    return mlirBytecodeEmitError(
        context,
        "trailing characters found after Attribute assembly format: '%s'",
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

    // Try BytecodeReaderConfig type callbacks first.
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(type.range.bytes);
    MlirBytecodeDialectBytecodeReader reader(state, stream);
    for (auto &callback : state.config.getBytecodeReaderConfig().getTypeCallbacks()) {
      Type result;
      LogicalResult callbackResult = callback->read(reader, dialect.name, result);
      if (succeeded(callbackResult)) {
        if (result) {
          type.value = result;
          return mlirBytecodeSuccess();
        }
      } else {
        // Callback explicitly returned failure - do not fall through to dialect
        return mlirBytecodeFailure();
      }
    }

    // Fall back to dialect interface.
    if (!dialect.interface) {
      return mlirBytecodeEmitError(
          context, "dialect '%s' does not implement the bytecode interface",
          dialect.name.str().c_str());
    }

    // Ask the dialect to parse the entry.
    stream = mlirBytecodeStreamCreate(type.range.bytes);
    MlirBytecodeDialectBytecodeReader dialectReader(state, stream);
    type.value = dialect.interface->readType(dialectReader);
    if (!type.value)
      return mlirBytecodeFailure();
    return mlirBytecodeSuccess();
  }

  auto asmStr = StringRef((const char *)type.range.bytes.data,
                          type.range.bytes.length - 1);
  // Invoke the MLIR assembly parser to parse the entry text.
  type.value = ::parseType(asmStr, state.getContext());
  // Type was parsed or null.
  size_t numRead = asmStr.size(); // Assume all consumed if parsed
  if (!type.value) {
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
  auto &range = state.attributes[attrHandle.id].range;
  range.bytes = bytes;
  range.dialectHandle = dialectHandle;
  range.hasCustom = hasCustom;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeAssociateTypeRange(void *context, MlirBytecodeTypeHandle typeHandle,
                               MlirBytecodeDialectHandle dialectHandle,
                               MlirBytecodeBytesRef bytes, bool hasCustom) {
  ParsingState &state = *(ParsingState *)context;
  auto &range = state.types[typeHandle.id].range;
  range.bytes = bytes;
  range.dialectHandle = dialectHandle;
  range.hasCustom = hasCustom;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *context,
                            MlirBytecodeDialectHandle dialectHandle,
                            MlirBytecodeStringHandle stringHandle) {
  ParsingState &state = *(ParsingState *)context;
  if (dialectHandle.id >= state.dialects.size())
    return mlirBytecodeEmitError(context, "invalid dialect index: %" PRIu64,
                                  dialectHandle.id);
  BytecodeDialect &dialect = state.dialects[dialectHandle.id];
  auto name = state.string(stringHandle);
  if (failed(name))
    return mlirBytecodeEmitError(context, "invalid string index: %" PRIu64,
                                  stringHandle.id);
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
    return mlirBytecodeEmitError(context, "invalid dialect index: %" PRIu64,
                                  dialectHandle.id);
  BytecodeDialect *dialect = &state.dialects[dialectHandle.id];
  FailureOr<StringRef> name = state.string(strHandle);
  if (failed(name))
    return mlirBytecodeEmitError(context, "invalid string index: %" PRIu64,
                                  strHandle.id);
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

  auto &dialectOr = state.dialects[dialect.id];
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
    
    // For empty/declaration resources (detected by Blob kind with 0 size data),
    // only declare but don't call parseResource (which would store empty data
    // that gets printed on output).
    if (entry.getKind() == AsmResourceEntryKind::Blob) {
      FailureOr<AsmResourceBlob> blob = entry.parseAsBlob();
      if (failed(blob))
        return failure();
      if (blob->getData().empty()) {
        // Empty blob - just a declaration, don't store
        return success();
      }
      // IMPORTANT: We cannot use parser->parseResource(entry) because 
      // declareResource may have renamed the key (e.g., "resource" -> "resource_1")
      // if an entry with that name already existed. parseResource would use 
      // entry.getKey() (the original name) for the update call, which would
      // update the wrong BlobEntry or fail.
      // Instead, directly set the blob on the handle's entry.
      auto *denseResHandle = dyn_cast<DenseResourceElementsHandle>(&*handle);
      if (denseResHandle) {
        auto *blobEntry = denseResHandle->getResource();
        if (blobEntry)
          blobEntry->setBlob(std::move(*blob));
      }
      return success();
    }
    
    // For non-blob resources (bool, string), use parseResource as before.
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
  mlirBytecodeEmitDebug("resource blob callback, handler=%p", (void*)&state.resourceHandler);
  if (!state.resourceHandler)
    return mlirBytecodeUnhandled();
  auto keyOr = state.string(resourceKey);
  if (failed(keyOr))
    return mlirBytecodeFailure();
  mlirBytecodeEmitDebug("resource blob key=%s size=%zu", keyOr->str().c_str(), blob.length);
  ParsedResourceEntry entry(
      keyOr.value(),
      ArrayRef(static_cast<const uint8_t *>(blob.data), blob.length), alignment,
      state);
  auto ret = state.resourceHandler(entry);
  mlirBytecodeEmitDebug("resource handler returned %d, dialectResources size=%zu", 
                        succeeded(ret), state.dialectResources.size());
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
// Callback Implementations for v1+ Bytecode Features
//===----------------------------------------------------------------------===//

MlirBytecodeStatus
mlirBytecodeDialectVersionCallBack(void *context, MlirBytecodeHandle dialectHandle,
                                   MlirBytecodeBytesRef version) {
  ParsingState &state = *(ParsingState *)context;
  
  if (dialectHandle.id >= state.dialects.size())
    return mlirBytecodeEmitError(context, "invalid dialect for version callback");
  
  BytecodeDialect &dialect = state.dialects[dialectHandle.id];
  
  // Load the dialect first to get access to the interface.
  if (failed(dialect.load(state, state.getContext())))
    return mlirBytecodeEmitError(context, "failed to load dialect for version");
  
  // If the dialect has a BytecodeDialectInterface, use it to parse the version.
  if (dialect.interface) {
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(version);
    MlirBytecodeDialectBytecodeReader reader(state, stream);
    dialect.version = dialect.interface->readVersion(reader);
    if (!dialect.version)
      return mlirBytecodeEmitError(context, "failed to read dialect version");
  }
  
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectOpWithRegisteredCallBack(void *context, MlirBytecodeHandle opHandle,
                                            MlirBytecodeHandle dialectHandle,
                                            MlirBytecodeHandle nameHandle, bool isRegistered) {
  // For v5+, register the operation name just like the regular callback.
  // The isRegistered flag indicates if the op was registered at serialization time.
  ParsingState &state = *(ParsingState *)context;
  assert(state.opNames.size() == opHandle.id);

  if (dialectHandle.id >= state.dialects.size())
    return mlirBytecodeEmitError(context, "invalid dialect");
  BytecodeDialect *dialect = &state.dialects[dialectHandle.id];
  MlirBytecodeStringHandle strHandle;
  strHandle.id = nameHandle.id;
  FailureOr<StringRef> name = state.string(strHandle);
  if (failed(name))
    return mlirBytecodeEmitError(context, "invalid op name");
  state.opNames.emplace_back(dialect, *name);
  // TODO: Store isRegistered flag if needed for verification
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceEmptyCallBack(void *context, MlirBytecodeHandle resource) {
  // Empty resource declaration - needs to declare the resource and add it to
  // dialectResources so later readResourceHandle calls work.
  // Unlike blob resources, we should NOT call parseResource since there's no data.
  ParsingState &state = *(ParsingState *)context;
  
  // Get the resource key by looking it up as a string handle
  MlirBytecodeStringHandle stringHandle = {resource.id};
  auto keyOr = state.string(stringHandle);
  if (failed(keyOr))
    return mlirBytecodeFailure();

  mlirBytecodeEmitDebug("empty resource declaration for key=%s", keyOr->str().c_str());
  
  // For empty resources, we need to get the current dialect parser and declare
  // the resource directly, without calling parseResource (which would store data).
  // The declaration is used to get the handle for later readResourceHandle calls.
  
  // We need to access the parser from the current dialect resource group.
  // The resourceHandler was set up by mlirBytecodeResourceDialectGroupEnter.
  // We need to duplicate the declare logic here.
  
  // Since we don't have direct access to the parser here, we create a minimal
  // ParsedResourceEntry and call the handler, but the handler will call
  // declareResource which registers the resource even with empty data.
  // The actual data won't be stored because parseAsBlob returns empty array.
  if (!state.resourceHandler)
    return mlirBytecodeSuccess();  // No handler, just skip

  // Call with empty blob - the handler's parseResource will register it
  ParsedResourceEntry entry(keyOr.value(), ArrayRef<uint8_t>(), /*alignment=*/1, state);
  auto ret = state.resourceHandler(entry);
  return succeeded(ret) ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

MlirBytecodeStatus
mlirBytecodeOperationStateAddProperties(void *context,
                                        MlirBytecodeOperationState *opState,
                                        MlirBytecodeHandle propsIndex) {
  ParsingState &state = *(ParsingState *)context;

  // Check if we have properties section data.
  if (state.propertiesSection.empty() || state.propertiesOffsets.empty()) {
    // No properties section parsed - skip.
    return mlirBytecodeSuccess();
  }

  // Verify the properties index is valid.
  if (propsIndex.id >= state.propertiesOffsets.size()) {
    return mlirBytecodeEmitError(context, "invalid properties index");
  }

  // Get the offset into the properties section.
  int64_t offset = state.propertiesOffsets[propsIndex.id];
  if (offset < 0 || static_cast<size_t>(offset) >= state.propertiesSection.size()) {
    return mlirBytecodeEmitError(context, "properties offset out of bounds");
  }

  // Create a stream from the properties data starting at the offset.
  MlirBytecodeBytesRef propsBytes = {
      state.propertiesSection.data() + offset,
      state.propertiesSection.size() - offset
  };
  MlirBytecodeStream propsStream = mlirBytecodeStreamCreate(propsBytes);
  MlirBytecodeDialectBytecodeReader blobReader(state, propsStream);
  
  // Read the blob (size + data) to get the raw property bytes.
  ArrayRef<char> rawProperties;
  if (failed(blobReader.readBlob(rawProperties)))
    return mlirBytecodeFailure();
  
  // Create a new reader from the raw properties for readProperties.
  MlirBytecodeBytesRef rawBytes = {
      reinterpret_cast<const uint8_t *>(rawProperties.data()),
      rawProperties.size()
  };
  MlirBytecodeStream rawStream = mlirBytecodeStreamCreate(rawBytes);
  MlirBytecodeDialectBytecodeReader propReader(state, rawStream);

  // Get the operation name to check for BytecodeOpInterface.
  OperationName opName = opState->name;
  auto *iface = opName.getInterface<BytecodeOpInterface>();
  
  if (iface) {
    // For registered ops with BytecodeOpInterface, use the interface to read properties.
    if (failed(iface->readProperties(propReader, *opState)))
      return mlirBytecodeFailure();
  } else if (opName.isRegistered()) {
    // Registered op without BytecodeOpInterface - emit error.
    return mlirBytecodeEmitError(context, 
        "op has properties but missing BytecodeOpInterface");
  } else {
    // Unregistered op - store properties as propertiesAttr (attribute form).
    if (failed(propReader.readAttribute(opState->propertiesAttr)))
      return mlirBytecodeFailure();
  }

  return mlirBytecodeSuccess();
}

//===----------------------------------------------------------------------===//
// Use-List Order Processing
//===----------------------------------------------------------------------===//

/// Compute a unique ID for a use, based on the operation order and operand number.
static uint64_t getUseID(OpOperand &use, unsigned operationID) {
  return (static_cast<uint64_t>(operationID) << 32) |
         static_cast<uint64_t>(use.getOperandNumber());
}

/// Sort the use-list of a value according to the order parsed from bytecode.
static LogicalResult sortUseListOrder(Value value, ParsingState &state) {
  // Early return for trivial use-lists.
  if (value.use_empty() || value.hasOneUse())
    return success();

  bool hasIncomingOrder =
      state.valueToUseListMap.contains(value.getAsOpaquePointer());

  // Compute the current order of the use-list with respect to the global ordering.
  bool alreadySorted = true;
  auto &firstUse = *value.use_begin();
  uint64_t prevID = getUseID(firstUse, state.operationIDs.at(firstUse.getOwner()));
  SmallVector<std::pair<unsigned, uint64_t>> currentOrder = {{0, prevID}};
  
  for (auto item : llvm::drop_begin(llvm::enumerate(value.getUses()))) {
    uint64_t currentID = getUseID(item.value(), 
                                   state.operationIDs.at(item.value().getOwner()));
    alreadySorted &= prevID > currentID;
    currentOrder.push_back({item.index(), currentID});
    prevID = currentID;
  }

  // If already sorted and no custom order, we're done.
  if (alreadySorted && !hasIncomingOrder)
    return success();

  // Sort by descending useIDs if not already sorted.
  if (!alreadySorted) {
    std::sort(currentOrder.begin(), currentOrder.end(),
              [](auto elem1, auto elem2) { return elem1.second > elem2.second; });
  }

  if (!hasIncomingOrder) {
    // No custom order - just use the sorted order.
    SmallVector<unsigned> shuffle;
    for (auto &p : currentOrder)
      shuffle.push_back(p.first);
    value.shuffleUseList(shuffle);
    return success();
  }

  // Apply custom order from bytecode.
  UseListOrderStorage &customOrder = state.valueToUseListMap.at(value.getAsOpaquePointer());
  SmallVector<unsigned> shuffle = std::move(customOrder.indices);
  uint64_t numUses = value.getNumUses();

  // Handle pair encoding: expand (src, dst) pairs to full permutation.
  if (customOrder.indexPairEncoding) {
    SmallVector<unsigned> fullShuffle(numUses);
    std::iota(fullShuffle.begin(), fullShuffle.end(), 0);
    for (size_t i = 0; i + 1 < shuffle.size(); i += 2) {
      unsigned src = shuffle[i];
      unsigned dst = shuffle[i + 1];
      if (src < numUses && dst < numUses)
        std::swap(fullShuffle[src], fullShuffle[dst]);
    }
    shuffle = std::move(fullShuffle);
  }

  // Compose with current order if needed.
  if (!alreadySorted) {
    SmallVector<unsigned> composed(numUses);
    for (size_t i = 0; i < numUses; ++i) {
      if (shuffle[i] < currentOrder.size())
        composed[i] = currentOrder[shuffle[i]].first;
      else
        composed[i] = i;
    }
    shuffle = std::move(composed);
  }

  value.shuffleUseList(shuffle);
  return success();
}

/// Process all use-lists in the parsed module.
static LogicalResult processUseLists(Operation *topLevelOp, ParsingState &state) {
  // If no use-list orders were parsed, skip processing.
  if (state.valueToUseListMap.empty())
    return success();

  // Compute operation IDs via pre-order walk.
  unsigned operationID = 0;
  topLevelOp->walk<WalkOrder::PreOrder>(
      [&](Operation *op) { state.operationIDs.try_emplace(op, operationID++); });

  // Sort block argument use-lists.
  auto blockWalk = topLevelOp->walk([&](Block *block) {
    for (auto arg : block->getArguments())
      if (failed(sortUseListOrder(arg, state)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // Sort operation result use-lists.
  auto resultWalk = topLevelOp->walk([&](Operation *op) {
    for (auto result : op->getResults())
      if (failed(sortUseListOrder(result, state)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return failure(blockWalk.wasInterrupted() || resultWalk.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

bool mlir::isBytecode(llvm::MemoryBufferRef buffer) {
    return buffer.getBuffer().starts_with("ML\xefR");
}

MlirBytecodeStatus
readBytecodeFileImpl(llvm::MemoryBufferRef buffer, Block *block,
                     const ParserConfig &config,
                     const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef,
                     ParsingState *externalState = nullptr) {
  Location sourceFileLoc =
      FileLineColLoc::get(config.getContext(), buffer.getBufferIdentifier(),
                          /*line=*/0, /*column=*/0);
  MlirBytecodeBytesRef ref;
  ref.data = (const uint8_t *)buffer.getBufferStart();
  ref.length = buffer.getBufferSize();
  
  // Create ParsingState first so mlirBytecodeEmitError has valid context.
  std::unique_ptr<ParsingState> localState;
  if (!externalState) {
    localState = std::make_unique<ParsingState>(sourceFileLoc, config, bufferOwnerRef);
    externalState = localState.get();
  }
  ParsingState &state = *externalState;
  
  // Now populate parser state with valid error context.
  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&state, ref);
  
  // Store bytecode version early for lazy loading version check
  state.bytecodeVersion = parserState.version;
  
  // If parser state is empty, parsing failed (version check, malformed file, etc.)
  if (mlirBytecodeParserStateEmpty(&parserState))
    return mlirBytecodeFailure();

  // Extract and parse properties section for v5+ bytecode.
  // The section data is stored in parserState.sectionData[8] (mbci_kProperties = 8).
  const MlirBytecodeBytesRef &propsSection = parserState.sectionData[8];
  if (propsSection.data != nullptr && propsSection.length > 0) {
    // Parse the offset table from the properties section.
    // Format: count (varint), then count entries of {size (varint) + raw_data}.
    MlirBytecodeStream propStream = mlirBytecodeStreamCreate(propsSection);
    uint64_t count;
    if (!mlirBytecodeSucceeded(mlirBytecodeParseVarInt(&state, &propStream, &count)))
      return mlirBytecodeFailure();
    
    // Store the remaining data as the properties buffer.
    size_t headerSize = propStream.pos - propStream.start;
    state.propertiesSection = ArrayRef<uint8_t>(propsSection.data + headerSize,
                                                 propsSection.length - headerSize);
    
    // Build the offset table by walking through entries.
    state.propertiesOffsets.reserve(count);
    MlirBytecodeStream offsetStream = mlirBytecodeStreamCreate(
        {state.propertiesSection.data(), state.propertiesSection.size()});
    
    for (uint64_t i = 0; i < count; ++i) {
      // Record the current offset.
      uint64_t currentOffset = offsetStream.pos - offsetStream.start;
      state.propertiesOffsets.push_back(currentOffset);
      
      // Parse the entry size and skip the data.
      uint64_t dataSize;
      if (!mlirBytecodeSucceeded(mlirBytecodeParseVarInt(&state, &offsetStream, &dataSize)))
        return mlirBytecodeFailure();
      
      // Skip the raw data.
      offsetStream.pos += dataSize;
      if (offsetStream.pos > offsetStream.end)
        return mlirBytecodeFailure();
    }
  }

  // Parse the bytecode.
  MlirBytecodeStatus parseResult = mlirBytecodeParse(&state, &parserState, block);
  if (!mlirBytecodeSucceeded(parseResult))
    return parseResult;

  // Finalization: Process use-list orders before moving content.
  // This must happen while operations are still in moduleOp.
  if (!state.valueToUseListMap.empty()) {
    if (failed(processUseLists(*state.moduleOp, state)))
      return mlirBytecodeFailure();
  }

  // Apply dialect upgrades for dialects that have version information.
  for (BytecodeDialect &dialect : state.dialects) {
    if (dialect.interface && dialect.version) {
      if (failed(dialect.interface->upgradeFromVersion(
              *state.moduleOp, *dialect.version)))
        return mlirBytecodeFailure();
    }
  }

  // Check for unresolved forward references.
  if (!state.forwardRefOps.empty()) {
    state.emitError() << "not all forward unresolved forward operand references";
    // Drop all uses before destroying to avoid LLVM fatal error.
    for (Operation &op : state.forwardRefOps)
      op.dropAllUses();
    return mlirBytecodeFailure();
  }

  // Verify that the parsed operations are valid.
  if (state.config.shouldVerifyAfterParse() &&
      failed(verify(*state.moduleOp)))
    return mlirBytecodeFailure();

  // Splice the parsed operations over to the provided top-level block.
  auto &parsedOps = state.moduleOp->getBody()->getOperations();
  auto &destOps = state.dest->getOperations();
  destOps.splice(destOps.end(), parsedOps, parsedOps.begin(),
                 parsedOps.end());
  
  // Store parserState and version for lazy loading materialization.
  state.storedParserState = parserState;
  state.bytecodeVersion = parserState.version;

  return mlirBytecodeSuccess();
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

//===----------------------------------------------------------------------===//
// BytecodeReader Class Implementation
//===----------------------------------------------------------------------===//

/// Implementation class for BytecodeReader.
class BytecodeReader::Impl {
public:
  Impl(llvm::MemoryBufferRef buffer, const ParserConfig &config, bool lazyLoad,
       const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
      : buffer(buffer), config(config),
        bufferOwnerRef(bufferOwnerRef),
        fileLoc(FileLineColLoc::get(config.getContext(),
                                    buffer.getBufferIdentifier(),
                                    /*line=*/0, /*column=*/0)),
        state(fileLoc, config, bufferOwnerRef) {
    state.lazyLoading = lazyLoad;
  }

  LogicalResult readTopLevel(Block *block,
                             llvm::function_ref<bool(Operation *)> lazyOps) {
    // Set the lazy loading callback on the member state.
    state.lazyOpsCallback = lazyOps;
    
    // Use the member state (with lazy loading configured) for parsing.
    return success(mlirBytecodeSucceeded(
        readBytecodeFileImpl(buffer, block, config, bufferOwnerRef, &state)));
  }

  int64_t getNumOpsToMaterialize() const {
    return state.lazyLoadableOpsMap.size();
  }

  bool isMaterializable(Operation *op) {
    return state.lazyLoadableOpsMap.count(op);
  }

  LogicalResult materialize(Operation *op,
                            llvm::function_ref<bool(Operation *)> lazyOpsCallback) {
    auto it = state.lazyLoadableOpsMap.find(op);
    if (it == state.lazyLoadableOpsMap.end())
      return failure();
    
    state.lazyOpsCallback = lazyOpsCallback;
    
    // Get the deferred region info - save the list iterator before parsing
    // because new operations added during parsing can invalidate the map iterator.
    auto listIt = it->second;
    LazyLoadableOpInfo &info = *listIt;
    
    // Parse the deferred regions using the stored byte range
    if (!info.irData.empty()) {
      // Set up initial state for parsing context
      state.regionStack.clear();
      state.regionStack.emplace_back(op, /*isIsolatedFromAbove=*/true);
      // Ensure we have a valueScope for the isolated region
      state.valueScopes.emplace_back();
      
      // Parse the deferred region using the stored parser state
      if (!mlirBytecodeSucceeded(
              mlirBytecodeParseDeferredRegion(&state, &state.storedParserState,
                  info.irData.data(), info.irData.size(), op))) {
        return failure();
      }
    }
    
    // Erase from tracking - list iterator is still valid
    state.lazyLoadableOps.erase(listIt);
    state.lazyLoadableOpsMap.erase(op);
    
    return success();
  }

  LogicalResult finalize(function_ref<bool(Operation *)> shouldMaterialize) {
    while (!state.lazyLoadableOps.empty()) {
      Operation *op = state.lazyLoadableOps.begin()->op;
      if (shouldMaterialize(op)) {
        if (failed(materialize(op, nullptr)))
          return failure();
        continue;
      }
      op->dropAllReferences();
      op->erase();
      state.lazyLoadableOps.pop_front();
      state.lazyLoadableOpsMap.erase(op);
    }
    return success();
  }

private:
  llvm::MemoryBufferRef buffer;
  const ParserConfig &config;
  const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef;
  Location fileLoc;
  ParsingState state;
};

BytecodeReader::BytecodeReader(llvm::MemoryBufferRef buffer,
                               const ParserConfig &config, bool lazyLoad,
                               const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
    : impl(std::make_unique<Impl>(buffer, config, lazyLoad, bufferOwnerRef)) {}

BytecodeReader::~BytecodeReader() = default;

LogicalResult
BytecodeReader::readTopLevel(Block *block,
                             llvm::function_ref<bool(Operation *)> lazyOps) {
  return impl->readTopLevel(block, lazyOps);
}

int64_t BytecodeReader::getNumOpsToMaterialize() const {
  return impl->getNumOpsToMaterialize();
}

bool BytecodeReader::isMaterializable(Operation *op) {
  return impl->isMaterializable(op);
}

LogicalResult
BytecodeReader::materialize(Operation *op,
                            llvm::function_ref<bool(Operation *)> lazyOpsCallback) {
  return impl->materialize(op, lazyOpsCallback);
}

LogicalResult
BytecodeReader::finalize(function_ref<bool(Operation *)> shouldMaterialize) {
  return impl->finalize(shouldMaterialize);
}
