//===- ArithToLLVMRegister.h - Arith to LLVM registration -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVMREGISTER_H
#define MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVMREGISTER_H

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

// Option 1: This groups the pass and its register together. Its separate
// from the main Passes.h and its just drop in. But downside is that we have
// a file like this per pass.
#define GEN_PASS_REGISTRATION_ARITHTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVMREGISTER_H
