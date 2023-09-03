//===- ti-opt.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm-c/DebugInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

// This would normally be in separate files but kept local here for smallish
// example.

#include "SimpleDepTypeDialect.h.inc"
// Avoid sorting
#include "SimpleDepTypeDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "SimpleDepTypeAttrDefs.h.inc"
#define GET_ATTRDEF_CLASSES
#include "SimpleDepTypeAttrDefs.cpp.inc"
#define GET_OP_CLASSES
#include "SimpleDepType.h.inc"
#define GET_OP_CLASSES
#include "SimpleDepType.cpp.inc"

using namespace example::simple_deptype;

struct SimpleDepTypeInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all operations to be inlined everywhere.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
};

void SimpleDepTypeDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "SimpleDepTypeAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "SimpleDepType.cpp.inc"
      >();
  addInterfaces<SimpleDepTypeInlinerInterface>();
}

Attribute SimpleDepTypeDialect::parseAttribute(DialectAsmParser &parser,
                                               Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag))) return Attribute();
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
  if (parseResult.hasValue()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown 'simple_deptype' attribute");
  return Attribute();
}

//===----------------------------------------------------------------------===//
// TensorWithSymbolicShapes
//===----------------------------------------------------------------------===//

LogicalResult TensorEncodingAttr::verify(
    llvm::function_ref<InFlightDiagnostic()>,
    llvm::ArrayRef<FlatSymbolRefAttr>) {
  return success();
}

LogicalResult TensorEncodingAttr::verifyEncoding(
    llvm::ArrayRef<long> shape, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  size_t numDynamic = llvm::count(shape, ShapedType::kDynamicSize);
  if (this->getSymbolic().size() == numDynamic) return success();
  emitError() << "symbolic values do not match number of dynamic dims";
  return failure();
}

struct SymbolOrConstant {
  SymbolOrConstant(int64_t c) {
    constant = c;
    kind = Kind::kConstant;
  }
  SymbolOrConstant(FlatSymbolRefAttr s) {
    symbol = s.getAttr();
    kind = Kind::kSymbol;
  }

  void print(raw_ostream &os) {
    if (kind == SymbolOrConstant::Kind::kConstant)
      os << constant;
    else
      os << symbol;
  }

  union {
    StringAttr symbol;
    int64_t constant;
  };
  enum class Kind { kSymbol, kConstant } kind;
};

inline raw_ostream &operator<<(raw_ostream &os, SymbolOrConstant val) {
  val.print(os);
  return os;
}

struct SymbolicOrConstantDims {
  SymbolicOrConstantDims(ArrayRef<int64_t> dims, TensorEncodingAttr attr)
      : dims(dims), attr(attr) {}

  SymbolOrConstant get(int index) const {
    auto dim = dims[index];
    if (!ShapedType::isDynamic(dim) || !attr) return {dim};
    int dynamicIndex = 0;
    for (int i = 0; i < index; ++i)
      dynamicIndex += ShapedType::isDynamic(dims[i]);
    return {attr.getSymbolic()[dynamicIndex]};
  }

  bool isStatic() const { return !attr || attr.getSymbolic().empty(); }

  int rank() { return dims.size(); }

 private:
  ArrayRef<int64_t> dims;
  TensorEncodingAttr attr;
};

/// Simple wrapper class to enable "isa querying" and simple accessing of
/// shapes.
class TensorWithSymbolicShapes : public RankedTensorType {
 public:
  using RankedTensorType::RankedTensorType;

  SymbolicOrConstantDims getDims() const {
    TensorEncodingAttr attr;
    if (Attribute enc = getEncoding()) attr = enc.cast<TensorEncodingAttr>();
    return {getShape(), attr};
  }
  ArrayRef<FlatSymbolRefAttr> getSymbolic() const {
    auto encoding = RankedTensorType::getEncoding();
    if (!encoding) return {};
    return encoding.cast<TensorEncodingAttr>().getSymbolic();
  }

  TensorWithSymbolicShapes clone(DenseMap<StringAttr, int> resultMapping);
  TensorWithSymbolicShapes clone(DenseMap<StringAttr, StringAttr> remap);

  static bool classof(Type type);
};

bool TensorWithSymbolicShapes::classof(Type type) {
  if (auto rt = type.dyn_cast_or_null<RankedTensorType>())
    return !rt.getEncoding() || rt.getEncoding().isa<TensorEncodingAttr>();
  return false;
}

TensorWithSymbolicShapes TensorWithSymbolicShapes::clone(
    DenseMap<StringAttr, int> resultMapping) {
  if (hasStaticShape()) return *this;
  SmallVector<int64_t> shape(getShape().begin(), getShape().end());
  // Propagate known dims.
  const auto *dynIt = getSymbolic().begin();
  for (int64_t &dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      auto it = resultMapping.find(dynIt->getAttr());
      ++dynIt;
      if (it == resultMapping.end()) continue;
      dim = it->second;
    }
  }
  // Retain only unknown symbols.
  TensorEncodingAttr attr = TensorEncodingAttr::get(
      getContext(), to_vector(llvm::make_filter_range(
                        getSymbolic(), [&](FlatSymbolRefAttr attr) {
                          return resultMapping.count(attr.getAttr()) == 0;
                        })));
  if (attr.getSymbolic().empty()) attr = nullptr;
  return RankedTensorType::get(shape, getElementType(), attr)
      .cast<TensorWithSymbolicShapes>();
}

TensorWithSymbolicShapes TensorWithSymbolicShapes::clone(
    DenseMap<StringAttr, StringAttr> remap) {
  if (hasStaticShape()) return *this;
  // Remap
  SmallVector<FlatSymbolRefAttr> attrs(getSymbolic().begin(),
                                       getSymbolic().end());
  for (FlatSymbolRefAttr &attr : attrs) {
    auto it = remap.find(attr.getAttr());
    if (it == remap.end()) continue;
    attr = FlatSymbolRefAttr::get(it->second);
  }
  return RankedTensorType::get(getShape(), getElementType(),
                               TensorEncodingAttr::get(getContext(), attrs))
      .cast<TensorWithSymbolicShapes>();
}

void SimpleDepTypeDialect::printAttribute(Attribute attr,
                                          DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected 'simple_deptype' attribute kind");
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

LogicalResult DeassociateOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.size() != 1)
    return emitOptionalError(loc, "invalid number of arguments");
  inferredReturnTypes.resize(1);

  Type type = operands.front().getType();
  auto stype = type.dyn_cast<TensorWithSymbolicShapes>();
  if (!stype) {
    inferredReturnTypes[0] = stype;
    return success();
  }

  inferredReturnTypes[0] =
      RankedTensorType::get(stype.getShape(), stype.getElementType());
  return success();
}

LogicalResult AssociateTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AssociateTensorOpAdaptor op(operands, attributes);
  if (op.getDims().size() != op.getNames().size())
    return emitOptionalError(
        location,
        llvm::formatv(
            "mismatched in count of dims ({0}) and symbol names ({1})",
            op.getDims().size(), op.getNames().size()));
  auto type = op.getValue().getType().dyn_cast<RankedTensorType>();
  if (!type)
    return emitOptionalError(location, "only ranked tensor inputs supported");
  inferredReturnTypes.resize(1);

  auto names =
      llvm::map_range(op.getNames(), [&](Attribute attr) -> FlatSymbolRefAttr {
        return attr.cast<FlatSymbolRefAttr>();
      });
  auto attr = TensorEncodingAttr::get(context, llvm::to_vector(names));
  auto inferredType = TensorWithSymbolicShapes::get(
      type.getShape(), type.getElementType(), attr);
  if (!inferredType) return failure();

  inferredReturnTypes[0] = inferredType;
  return success();
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

struct AssociatedValues {
  AssociatedValues(Operation *function)
      : AssociatedValues(cast<FuncOp>(function)){};
  AssociatedValues(FuncOp function);

  SmallVector<AssociateOp> ops;
  DenseMap<StringAttr, SmallVector<Value>> symbolToAssociated;
  DenseMap<StringAttr, DenseSet<Operation *>> symbolUses;
  DenseMap<Value, DenseSet<StringAttr>> valueToSymbol;
};

AssociatedValues::AssociatedValues(FuncOp function) {
  function.walk([&](Operation *op) {
    if (auto aop = dyn_cast<AssociateOp>(op)) {
      valueToSymbol[aop.getValue()].insert(aop.getSymNameAttr());
      symbolToAssociated[aop.getSymNameAttr()].push_back(aop.getValue());
      ops.push_back(aop);
    }
    for (Value result : op->getResults()) {
      if (auto shape = result.getType().dyn_cast<TensorWithSymbolicShapes>()) {
        if (shape.hasStaticShape()) continue;
        for (auto symbol : shape.getSymbolic()) {
          symbolUses[symbol.getAttr()].insert(op);
        }
      }
    }
  });
  for (Block &block : function.body().getBlocks()) {
    Operation *terminator = block.getTerminator();
    for (Value val : terminator->getOperands()) {
      if (auto shape = val.getType().dyn_cast<TensorWithSymbolicShapes>()) {
        if (shape.hasStaticShape()) continue;
        for (auto symbol : shape.getSymbolic()) {
          symbolUses[symbol.getAttr()].insert(terminator);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

struct VerifyPass : public PassWrapper<VerifyPass, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "sdt-verify"; }
  StringRef getDescription() const final { return "Verify constraints"; }

  void runOnOperation() override {
    verifyFunction(getOperation());
    markAllAnalysesPreserved();
  }

  void verifyFunction(FuncOp func) {
    // We only accept functions with single block. This is both for simplicity
    // and as the expectation is that this is at a higher level where these
    // are used, before lowering to CFG or codegen.

    AssociatedValues &av = getAnalysis<AssociatedValues>();

    // Check that all symbols are uniquely named.
    // This should be expanded to whole module one that does direct verification
    // that the value associated everywhere matches (so inside the function used
    // it is the same symbol value).
    DenseSet<StringAttr> used;
    DenseMap<StringAttr, Value> seen;
    WalkResult res = func.walk([&](Operation *op) {
      if (auto aop = dyn_cast<AssociateOp>(op)) {
        bool inserted =
            seen.insert({aop.getSymNameAttr(), aop.getValue()}).second;
        if (!inserted) {
          if (aop.getValue() != seen[aop.getSymNameAttr()]) {
            InFlightDiagnostic diag = aop.emitOpError("redifinition of symbol ")
                                      << aop.getSymNameAttr() << "value";
            diag.attachNote(aop.getValue().getLoc()) << "new value";
            diag.attachNote(seen[aop.getSymNameAttr()].getLoc())
                << "previous association";
            return WalkResult::interrupt();
          }
        }
      }
      for (Type type : op->getResultTypes()) {
        if (auto shape = type.dyn_cast<TensorWithSymbolicShapes>()) {
          if (shape.hasStaticShape()) continue;
          for (auto symbol : shape.getSymbolic()) {
            used.insert(symbol.getAttr());
          }
        }
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) return signalPassFailure();

    for (auto use : used) {
      if (!seen.count(use)) {
        for (Operation *op : av.symbolUses[use]) {
          op->emitOpError("using un-associated symbolic value: ") << use;
        }
      }
    }

    for (Value val : func.getArguments()) {
      Type type = val.getType();
      if (auto shape = type.dyn_cast<TensorWithSymbolicShapes>()) {
        if (shape.hasStaticShape()) continue;
        for (auto symbol : shape.getSymbolic()) {
          if (!seen.count(symbol.getAttr())) {
            emitError(val.cast<BlockArgument>().getLoc())
                << "symbolic dim '" << symbol
                << "' used that is not associated inside function";
            return signalPassFailure();
          }
        }
      }
    }
  }
};

struct CanonicalizeAssociatePass
    : public PassWrapper<CanonicalizeAssociatePass, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "sdt-canonicalize"; }
  StringRef getDescription() const final {
    return "Canonicalize associate ops";
  }

  void runOnOperation() override {
    AssociatedValues &av = getAnalysis<AssociatedValues>();

    // Move all Associate ops to end of block of the value.
    bool invalidated = false;
    for (AssociateOp op : av.ops) {
      // Remove if no uses of symbol that is associated.
      if (av.symbolUses[op.getSymNameAttr()].empty()) {
        op.erase();
        invalidated = true;
        continue;
      }
      Block *block = op.getValue().getParentBlock();
      op->moveBefore(block->getTerminator());
    }
    if (!invalidated) markAnalysesPreserved<AssociatedValues>();
  }
};

// Fold the values of the associate into the types. This is a module pass we
// allow symbols in function arguments and require that symbol has the same
// value throughout the module.
struct FoldAssociatePass
    : public PassWrapper<FoldAssociatePass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "sdt-fold-associate"; }
  StringRef getDescription() const final { return "Fold associate ops"; }

  FoldAssociatePass() = default;
  FoldAssociatePass(const FoldAssociatePass &other) {
    constantFold = other.constantFold;
    mergeEqual = other.mergeEqual;
  }

  void runOnOperation() override {
    bool changed = false;
    for (FuncOp func : getOperation().getOps<FuncOp>()) {
      if (failed(collectFromFunction(func))) return signalPassFailure();
    }
    for (FuncOp func : getOperation().getOps<FuncOp>())
      changed |= runOnFunction(func);

    if (!changed) markAnalysesPreserved<AssociatedValues>();
  }

  LogicalResult collectFromFunction(FuncOp func) {
    AssociatedValues &av = getChildAnalysis<AssociatedValues>(func);

    if (constantFold) {
      for (AssociateOp aop : av.ops) {
        APInt dim;
        if (!matchPattern(aop.getValue(), m_ConstantInt(&dim))) continue;
        int64_t val = dim.getSExtValue();
        auto it = constantMapping.insert({aop.getSymNameAttr(), val});
        if (it.second) {
          if (it.first->getSecond() != val)
            return LogicalResult(
                aop.emitOpError("non-unique global symbol value: ")
                << it.first->getSecond() << " vs " << val);
        }

        constantMapping[aop.getSymNameAttr()] = dim.getSExtValue();
      }
    }

    if (mergeEqual) {
      // Merge symbols if they are equal.
      for (auto it : av.valueToSymbol) {
        if (it.second.size() == 1) continue;

        // Take lexicographically smallest symbol as top.
        StringAttr top = *it.second.begin();
        for (auto symbol : llvm::drop_begin(it.getSecond())) {
          if (top.strref() > symbol.strref()) {
            drop.insert(top);
            top = symbol;
          } else {
            drop.insert(symbol);
          }
        }
        for (auto symbol : it.getSecond()) equalMapping[symbol] = top;
      }
    }
    return success();
  }

  bool runOnFunction(FuncOp func) {
    AssociatedValues &av = getChildAnalysis<AssociatedValues>(func);

    bool changed = false;
    if (constantFold) {
      for (AssociateOp aop : av.ops) {
        if (!constantMapping.count(aop.getSymNameAttr())) continue;

        for (Operation *use : av.symbolUses[aop.getSymNameAttr()]) {
          for (int i = 0, e = use->getNumResults(); i != e; ++i) {
            for (OpResult result : use->getResults()) {
              Type type = result.getType();
              auto dtt = type.dyn_cast<TensorWithSymbolicShapes>();
              if (!dtt) continue;

              Type newType = dtt.clone(constantMapping);
              result.setType(newType);
            }
          }
        }

        drop.insert(aop.getSymNameAttr());
        changed = true;
      }

      // Handle function args.
      bool funcSigChanged = false;
      for (auto arg : func.getArguments()) {
        Type type = arg.getType();
        auto dtt = type.dyn_cast<TensorWithSymbolicShapes>();
        if (!dtt) continue;

        Type newType = dtt.clone(constantMapping);
        if (type != newType) {
          arg.setType(newType);
          changed = true;
          funcSigChanged = true;
        }
      }
      if (funcSigChanged) {
        auto newType = FunctionType::get(
            &getContext(), func.getArgumentTypes(),
            func.getBody().front().getTerminator()->getOperandTypes());
        func.setType(newType);
      }
    }

    if (mergeEqual) {
      // Update return types.
      for (AssociateOp aop : av.ops) {
        for (Operation *use : av.symbolUses[aop.getSymNameAttr()]) {
          for (int i = 0, e = use->getNumResults(); i != e; ++i) {
            for (OpResult result : use->getResults()) {
              Type type = result.getType();
              auto dtt = type.dyn_cast<TensorWithSymbolicShapes>();
              if (!dtt) continue;

              Type newType = dtt.clone(equalMapping);
              if (result.getType() != newType) {
                result.setType(newType);
                changed = true;
              }
            }
          }
        }
      }

      // Handle function args.
      bool funcSigChanged = false;
      for (auto arg : func.getArguments()) {
        Type type = arg.getType();
        auto dtt = type.dyn_cast<TensorWithSymbolicShapes>();
        if (!dtt) continue;

        Type newType = dtt.clone(equalMapping);
        if (type != newType) {
          arg.setType(newType);
          changed = true;
          funcSigChanged = true;
        }
      }
      if (funcSigChanged) {
        auto newType = FunctionType::get(
            &getContext(), func.getArgumentTypes(),
            func.getBody().front().getTerminator()->getOperandTypes());
        func.setType(newType);
      }
    }

    for (AssociateOp aop : av.ops)
      if (drop.count(aop.getSymNameAttr())) aop.erase();
    return changed;
  }

  // Symbol to constant.
  DenseMap<StringAttr, int> constantMapping;

  // Mapping from equal symbols to single representative one.
  DenseMap<StringAttr, StringAttr> equalMapping;

  // Symbols to drop.
  DenseSet<StringAttr> drop;

  ::mlir::Pass::Option<bool> constantFold{
      *this, "constant-fold",
      ::llvm::cl::desc("Constant fold known dims into types"),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<bool> mergeEqual{*this, "merge-equal",
                                        ::llvm::cl::desc("Merge equal symbols"),
                                        ::llvm::cl::init(false)};
};

// Very simple evaluator that assumes all single result ops. Just enough for
// example.
struct ShapeEvaluator {
  ShapeEvaluator(Operation *op, FuncOp shapeFn) : op(op), shapeFn(shapeFn) {}

  LogicalResult evaluateWith(ValueRange args) {
    for (auto it : llvm::zip(args, shapeFn.getArguments())) {
      mapping[std::get<1>(it)] = {std::get<0>(it)};
    }
    LogicalResult res = success();
    shapeFn.walk([&](Operation *innerOp) {
      if (innerOp->hasTrait<mlir::OpTrait::IsTerminator>()) {
        terminator = innerOp;
        return WalkResult::interrupt();
      }

      ValT eval =
          TypeSwitch<Operation *, ValT>(innerOp)
              .Case<shape::ConstSizeOp>([&](shape::ConstSizeOp cop) -> ValT {
                mapping[cop] = {cop.getValueAttr()};
                return mapping[cop];
              })
              .Case<shape::FromExtentsOp>(
                  [&](shape::FromExtentsOp fop) -> ValT {
                    auto range = llvm::map_range(
                        fop.getExtents(),
                        [&](Value val) { return mapping[val].front(); });
                    mapping[fop] = {range.begin(), range.end()};
                    return mapping[fop];
                  })
              .Case<ReturnOp>(
                  [&](ReturnOp rop) { return mapping[rop.getOperand(0)]; })
              .Default([](Operation *) -> ValT { return {}; });
      if (eval.empty()) {
        innerOp->emitError() << "unable to evaluate " << innerOp;
        res = failure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return success(succeeded(res));
  }

  Type getReturn(DenseMap<Value, DenseSet<StringAttr>> &valueMapping,
                 int index) {
    Type retType = op->getResultTypes()[index];
    RankedTensorType type = retType.dyn_cast<RankedTensorType>();
    if (!type) return retType;

    SmallVector<int64_t> dims;
    SmallVector<FlatSymbolRefAttr> symbols;
    for (auto dim : mapping[terminator->getOperand(index)]) {
      if (auto intAttr = dim.dyn_cast<Attribute>()) {
        dims.push_back(intAttr.cast<IntegerAttr>().getInt());
      } else {
        dims.push_back(ShapedType::kDynamicSize);
        // Just grab the first symbol.
        symbols.push_back(
            FlatSymbolRefAttr::get(*valueMapping[dim.get<Value>()].begin()));
      }
    }
    auto attr = TensorEncodingAttr::get(shapeFn.getContext(), symbols);
    return RankedTensorType::get(dims, type.getElementType(), attr);
  }

 private:
  using ValT = SmallVector<OpFoldResult>;

  DenseMap<Value, ValT> mapping;

  Operation *op;

  FuncOp shapeFn;

  Operation *terminator;
};

struct ShapeReportPass
    : public PassWrapper<ShapeReportPass, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "sdt-shape-report"; }
  StringRef getDescription() const final {
    return "Report potential dependent type using shape function";
  }

  void runOnOperation() override {
    AssociatedValues &av = getAnalysis<AssociatedValues>();

    // This is querying the parent op, but read-only.
    auto module = getOperation()->getParentOfType<ModuleOp>();
    auto shapeFnLibs = module->getAttrOfType<ArrayAttr>("shape.lib");
    if (!shapeFnLibs || shapeFnLibs.size() != 1) {
      module.emitError() << "requires exactly one shape.lib value";
      return signalPassFailure();
    }
    auto shapeFnLib = module.lookupSymbol<shape::FunctionLibraryOp>(
        (*shapeFnLibs.begin()).cast<FlatSymbolRefAttr>());

    auto walkResult = getOperation().walk([&](Operation *op) {
      FuncOp shapeOfOp = shapeFnLib.getShapeFunction(op);
      if (!shapeOfOp) return WalkResult::advance();

      ShapeEvaluator evaluator(op, shapeOfOp);
      if (failed(evaluator.evaluateWith(op->getOperands())))
        return WalkResult::interrupt();

      for (int i = 0, e = op->getNumResults(); i != e; ++i) {
        Type type = op->getResultTypes()[i];
        Type predictedType = evaluator.getReturn(av.valueToSymbol, i);
        op->emitRemark() << "existing type " << type << " and predicted "
                         << predictedType;
      }

      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) return signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

// Two groups of conversion here, the first is to shape dialect/towards codegen,
// while the second is conversion from shape dialect to SDT. We don't really
// expect folks would be converting back to dependent types ones lowered though.

// This uses by default a very direct approach that does not consider
// whether it is needed to materialize constraint or not (e.g., doesn't use
// shape function). The output is quite verbose (more so than
// just how verbose shape dialect is due to current focus).
//
// Alternatively one can rely on known shape functions.
struct ConvertToShapePass
    : public PassWrapper<ConvertToShapePass, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "sdt-to-shape"; }
  StringRef getDescription() const final {
    return "Legalize to shape naively.";
  }

  ConvertToShapePass() = default;
  ConvertToShapePass(const ConvertToShapePass &other) {
    useShapeFns = other.useShapeFns;
  }

  void runOnOperation() override;

  ::mlir::Pass::Option<bool> useShapeFns{
      *this, "use-shape-fns",
      ::llvm::cl::desc("Utilize shape functions to insert less ops"),
      ::llvm::cl::init(false)};
};

void ConvertToShapePass::runOnOperation() {
  // Filter out more advanced cases. This is expected to be running more at
  // higher-level/before codegen or lowering to CFG form.
  if (getOperation().getBlocks().size() != 1) return signalPassFailure();
  WalkResult res = getOperation().walk([&](Operation *op) {
    if (op->getNumRegions() != 0 && !isa<FuncOp>(op)) {
      op->emitWarning() << "unable to convert ops with regions";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();

  shape::FunctionLibraryOp shapeFnLib;
  if (useShapeFns) {
    // This is querying the parent op, but read-only.
    auto module = getOperation()->getParentOfType<ModuleOp>();
    auto shapeFnLibs = module->getAttrOfType<ArrayAttr>("shape.lib");
    if (!shapeFnLibs || shapeFnLibs.size() != 1) {
      module.emitError() << "requires exactly one shape.lib value";
      return signalPassFailure();
    }
    shapeFnLib = module.lookupSymbol<shape::FunctionLibraryOp>(
        (*shapeFnLibs.begin()).cast<FlatSymbolRefAttr>());
  }

  AssociatedValues &av = getAnalysis<AssociatedValues>();
  Operation *terminator = getOperation().getBlocks().front().getTerminator();

  for (AssociateOp aop : av.ops) {
    OpBuilder b(aop);
    auto dim = b.create<shape::IndexToSizeOp>(aop.getLoc(), aop.getValue());

    SmallVector<Value> sinks;
    for (Operation *op : av.symbolUses[aop.getSymNameAttr()]) {
      if (useShapeFns) {
        Operation *shapeOfOp = shapeFnLib.getShapeFunction(op);
        // If the shape function is known, then one can derived the result types
        // again when we need to refine.
        if (shapeOfOp) continue;
      }

      b.setInsertionPointAfter(op);
      SmallVector<Value> toMeet;
      for (OpResult res : op->getResults()) {
        auto type = res.getType().dyn_cast<TensorWithSymbolicShapes>();
        if (!type) continue;

        SymbolicOrConstantDims dims = type.getDims();
        for (int i = 0, e = dims.rank(); i != e; ++i) {
          SymbolOrConstant it = dims.get(i);
          if (it.kind != SymbolOrConstant::Kind::kSymbol) continue;
          if (it.symbol != aop.getSymNameAttr()) continue;
          auto shape = b.create<shape::ShapeOfOp>(op->getLoc(), res);
          auto index = b.create<arith::ConstantIndexOp>(op->getLoc(), i);
          auto extent = b.create<shape::GetExtentOp>(
              op->getLoc(), shape::SizeType::get(&getContext()), shape, index);
          toMeet.push_back(extent);
        }
      }
      // The associate need not happen before symbol is first used.
      if (op->isBeforeInBlock(aop)) b.setInsertionPoint(aop);
      for (auto val : toMeet) {
        sinks.push_back(
            b.create<shape::MeetOp>(val.getLoc(), val, dim, /*error=*/nullptr));
      }
    }

    // Handle args (this is a little brute force, ignoring visibility).
    b.setInsertionPointAfter(dim);
    SmallVector<Value> toMeet;
    for (auto arg : getOperation().getArguments()) {
      auto type = arg.getType().dyn_cast<TensorWithSymbolicShapes>();
      if (!type) continue;

      SymbolicOrConstantDims dims = type.getDims();
      Location loc = getOperation().getLoc();
      for (int i = 0, e = dims.rank(); i != e; ++i) {
        SymbolOrConstant it = dims.get(i);
        if (it.kind != SymbolOrConstant::Kind::kSymbol) continue;
        if (it.symbol != aop.getSymNameAttr()) continue;
        auto shape = b.create<shape::ShapeOfOp>(loc, arg);
        auto index = b.create<arith::ConstantIndexOp>(loc, i);
        auto extent = b.create<shape::GetExtentOp>(
            loc, shape::SizeType::get(&getContext()), shape, index);
        toMeet.push_back(extent);
      }
    }

    for (auto val : toMeet) {
      sinks.push_back(b.create<shape::MeetOp>(
          val.getLoc(), val, dim,
          // TODO: Make this a nicer message.
          StringAttr::get("input constraints unmet", &getContext())));
    }

    b.setInsertionPoint(terminator);
    // For convenience use the name of the symbol as location.
    if (!sinks.empty())
      b.create<WitnessSinkOp>(NameLoc::get(aop.getSymNameAttr()), sinks,
                              aop.getSymNameAttr());
  }

  // Drop the symbolic shape attributes.
  for (AssociateOp aop : av.ops) {
    for (Operation *op : av.symbolUses[aop.getSymNameAttr()]) {
      for (OpResult res : op->getResults()) {
        auto type = res.getType().dyn_cast<TensorWithSymbolicShapes>();
        if (!type) continue;
        res.setType(
            RankedTensorType::get(type.getShape(), type.getElementType()));
      }
    }
    aop.erase();
  }

  SmallVector<Type> argTypes;
  for (auto arg : getOperation().getArguments()) {
    auto type = arg.getType().dyn_cast<TensorWithSymbolicShapes>();
    if (!type) {
      argTypes.push_back(arg.getType());
      continue;
    }

    auto newType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    arg.setType(newType);
    argTypes.push_back(newType);
  }
  auto newType = FunctionType::get(
      &getContext(), argTypes,
      getOperation().getBody().front().getTerminator()->getOperandTypes());
  getOperation().setType(newType);
}

// This only handles the above case where one used the more explicit lowering.
// It could be enhanced by doing an intraprocedural analysis, collecting all
// values that (according to witness sink or shape functions) contribute to
// result shape & uniquely naming them. As this is not the expected direction of
// transformation, only the simple case is shown.
struct ConvertFromShapeNaivePass
    : public PassWrapper<ConvertFromShapeNaivePass, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "shape-naive-to-sdt"; }

  void runOnOperation() override;
};

void ConvertFromShapeNaivePass::runOnOperation() {
  // We are going to cheat here a little bit and assume the format provided by
  // the above for simplicity.
  //
  //  %c0 = arith.constant 0 : index
  //  %0 = shape.index_to_size %arg1
  //  %1 = shape.shape_of %arg0
  //  %2 = shape.get_extent %1, %c0
  //  %3 = shape.meet %2, %0
  //  simple_deptype.witness_sink %3
  //
  // This should be using an interpretation over shapes/shape function still
  // with constraints.
  DenseMap<Value, SmallVector<StringAttr>> rewrite;
  OpBuilder b(getOperation().getBody().front().getTerminator());
  DenseSet<Operation *> toDelete;
  getOperation().walk([&](WitnessSinkOp sink) {
    AssociateOp associate = nullptr;
    // This assumes the name was stashed in the location of the sink (as is done
    // in the conversion above).
    StringAttr name = sink.getSymNameAttr();
    auto meets = sink.getOperands();
    toDelete.insert(sink);
    for (Value val : meets) {
      auto meet = cast<shape::MeetOp>(val.getDefiningOp());
      auto extent = cast<shape::GetExtentOp>(meet.getArg0().getDefiningOp());
      int64_t index = *extent.getConstantDim();
      auto shapeOf = cast<shape::ShapeOfOp>(extent.getShape().getDefiningOp());
      auto indexToSize =
          cast<shape::IndexToSizeOp>(meet.getArg1().getDefiningOp());
      auto &dims = rewrite[shapeOf.getArg()];
      dims.resize(
          shapeOf.getArg().getType().cast<RankedTensorType>().getRank());
      dims[index] = name;
      if (!associate)
        associate =
            b.create<AssociateOp>(meet.getLoc(), indexToSize.getArg(), name);

      for (Operation *op :
           ArrayRef<Operation *>{meet, extent, shapeOf, indexToSize})
        toDelete.insert(op);
      if (extent.getDim().hasOneUse())
        toDelete.insert(extent.getDim().getDefiningOp());
    }
  });

  DenseMap<FuncOp, SmallVector<Type>> fns;
  for (auto it : rewrite) {
    Value val = it.first;
    if (Operation *op = val.getDefiningOp()) {
      auto range = map_range(
          make_filter_range(it.second, [](StringAttr val) { return !!val; }),
          [](StringAttr str) { return FlatSymbolRefAttr::get(str); });
      auto attr =
          TensorEncodingAttr::get(&getContext(), llvm::to_vector(range));
      RankedTensorType type = val.getType().cast<RankedTensorType>();
      val.setType(
          RankedTensorType::get(type.getShape(), type.getElementType(), attr));
    } else {
      // Function argument case.
      auto barg = val.cast<BlockArgument>();
      auto func = cast<FuncOp>(barg.getOwner()->getParentOp());
      if (fns.count(func) == 0) {
        fns[func] = to_vector(func.getType().getInputs());
      }
      auto range = map_range(
          make_filter_range(it.second, [](StringAttr val) { return !!val; }),
          [](StringAttr str) { return FlatSymbolRefAttr::get(str); });
      auto attr =
          TensorEncodingAttr::get(&getContext(), llvm::to_vector(range));
      RankedTensorType type = val.getType().cast<RankedTensorType>();
      fns[func][barg.getArgNumber()] =
          RankedTensorType::get(type.getShape(), type.getElementType(), attr);
      barg.setType(fns[func][barg.getArgNumber()]);
    }
  }

  for (auto it : fns) {
    FuncOp func = it.first;
    Operation *terminator = func.getBody().front().getTerminator();
    auto type = FunctionType::get(&getContext(), it.second,
                                  terminator->getOperandTypes());
    func.setType(type);
  }

  // In the above we could be reusing values, so drop all uses as we know the
  // ordering of toDelete may not be fully consumer-producer but all
  // operations will be finally deleted.
  for (Operation *op : toDelete) {
    op->dropAllUses();
    op->erase();
  }
}

// TODO: Add inference context example (one that covers both the above
// uniformly).

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<arith::ArithmeticDialect, shape::ShapeDialect,
                  SimpleDepTypeDialect, StandardOpsDialect,
                  tensor::TensorDialect>();

  PassRegistration<CanonicalizeAssociatePass>();
  PassRegistration<FoldAssociatePass>();
  PassRegistration<ShapeReportPass>();
  PassRegistration<VerifyPass>();
  registerCanonicalizerPass();
  registerInlinerPass();
  registerSCCPPass();

  PassRegistration<ConvertFromShapeNaivePass>();
  PassRegistration<ConvertToShapePass>();

  return asMainReturnCode(
      MlirOptMain(argc, argv, "Type interface example driver\n", registry));
}
