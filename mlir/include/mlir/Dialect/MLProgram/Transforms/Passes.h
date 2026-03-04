//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Operation;
namespace ml_program {

/// Repopulate constants in `op` from `externalWeightsModule`.
LogicalResult repopulateConstants(Operation *op,
                                  Operation *externalWeightsModule);

#define GEN_PASS_DECL
#include "mlir/Dialect/MLProgram/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MLProgram/Transforms/Passes.h.inc"

} // namespace ml_program
} // namespace mlir

#endif // MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES_H_
