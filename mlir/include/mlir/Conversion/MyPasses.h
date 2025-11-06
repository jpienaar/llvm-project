//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MY_PASSES_H
#define MLIR_CONVERSION_MY_PASSES_H

#include "mlir/Conversion/XeVMToLLVM/XeVMToLLVM.h"

namespace mlir {

// Option 2: Just include all the original headders and then define
// here what register functions one needs. Can be easily done downstream
// is just silly to repeat too much and also doesn't feel nice to have a
// h.inc used generally.
#define GEN_PASS_REGISTRATION_XEVMTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_MY_PASSES_H
