//===-- RawBytecodeTypes.h - MLIR bytecode basic types ------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_BYTECODETYPES_H
#define MLIRBC_BYTECODETYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Value representing the status of parsing. Status is either a success,
/// failure unhandled or interrupted. Instances of MlirBytecodeStatus must only
/// be inspected using the associated functions.
typedef struct MlirBytecodeStatus MlirBytecodeStatus;

/// Represents a size populated during parsing.
/// This is purely documentative.
typedef uint64_t MlirBytecodeSize;

/// Reader struct for dialect attribute & type parsing should use.
typedef struct MlirBytecodeDialectReader MlirBytecodeDialectReader;

// Handles for the different types.
// Note: These should probably all be explicit structs to go beyond compiler
// warnings and to errors. Currently this isn't done as the type is mostly used
// for documentation, warnings help and avoids needing multiple variant of the
// same read function.
struct MlirBytecodeHandle {
  MlirBytecodeSize id;
};
typedef struct MlirBytecodeHandle MlirBytecodeHandle;
typedef MlirBytecodeHandle MlirBytecodeAttrHandle;
typedef MlirBytecodeHandle MlirBytecodeDialectHandle;
typedef MlirBytecodeHandle MlirBytecodeLocHandle;
typedef MlirBytecodeHandle MlirBytecodeOpHandle;
typedef MlirBytecodeHandle MlirBytecodeResourceHandle;
typedef MlirBytecodeHandle MlirBytecodeStringHandle;
typedef MlirBytecodeHandle MlirBytecodeTypeHandle;
typedef MlirBytecodeHandle MlirBytecodeValueHandle;

/// Reference to section of memory.
// Does not own the underlying string. This is equivalent to llvm::StringRef.
struct MlirBytecodeBytesRef {
  // Pointer to the start memory address.
  const uint8_t *data;
  // Length of the fragment.
  MlirBytecodeSize length;
};
typedef struct MlirBytecodeBytesRef MlirBytecodeBytesRef;

struct MlirBytecodeOpRef {
  // Handle to op dialect.
  MlirBytecodeDialectHandle dialect;
  // Handle to op.
  MlirBytecodeStringHandle op;
};
typedef struct MlirBytecodeOpRef MlirBytecodeOpRef;

// MLIR file structure.
typedef struct MlirBytecodeParserState MlirBytecodeParserState;

// MLIR bytecode stream.
struct MlirBytecodeStream {
  const uint8_t *start;
  const uint8_t *pos;
  const uint8_t *end;
};
typedef struct MlirBytecodeStream MlirBytecodeStream;

#ifdef __cplusplus
}
#endif

#endif // MLIRBC_BYTECODETYPES_H
