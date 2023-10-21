//===- URI.h - URI MLIR Dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the dialect containing the objects pertaining to external resources.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_URI_URI_H
#define MLIR_DIALECT_URI_URI_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/URI/URIDialect.h.inc"

namespace mlir::uri {
struct URIResourceHandle;
} // namespace mlir::uri

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/URI/URIDialectAttrDefs.h.inc"

#endif // MLIR_DIALECT_URI_URI_H
