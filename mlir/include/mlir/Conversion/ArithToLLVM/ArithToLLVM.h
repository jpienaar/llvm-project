//===- ArithToLLVM.h - Arith to LLVM dialect conversion ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
#define MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H

#include <memory>

namespace mlir {

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_ARITHTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToLLVMConversionPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);

void registerConvertArithToLLVMInterface(DialectRegistry &registry);
} // namespace arith
} // namespace mlir

// Option 3: Add pass registration in here. This would require that we make the
// general pass registry either not generate the inline registration methods or
// we don't allow both to be included at same time (e.g., guard the inclusion
// here with the header guard of the Passes.h one).
#ifndef MLIR_CONVERSION_PASSES_H
// (this is just as I was showing multiple options as tsame time)
#ifndef MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVMREGISTER_H
#include "mlir/Pass/Pass.h"
namespace mlir {
#define GEN_PASS_REGISTRATION_ARITHTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
}  // namespace mlir
#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVMREGISTER_H
#endif // MLIR_CONVERSION_PASSES_H


#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
