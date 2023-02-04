//===-- DialectBytecodeReader.c.inc - Interface dialects use ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// MLIR C bytecode reader for dialects.
//===----------------------------------------------------------------------===//

#ifndef MLIRBC_DIALECTBYTECODEREADER_H
#define MLIRBC_DIALECTBYTECODEREADER_H

#include "mlir/Bytecode/BytecodeTypes.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MlirBytecodeAPInt MlirBytecodeAPInt;

// Parse custom bytecode and associate attr handle with parsed attribute.
typedef MlirBytecodeStatus (*MlirBytecodeAttrParseCallBack)(
    MlirBytecodeDialectReader* reader, MlirBytecodeAttrHandle);

// Parse custom bytecode and associate type handle with parsed type.
typedef MlirBytecodeStatus (*MlirBytecodeTypeParseCallBack)(
    MlirBytecodeDialectReader* reader, MlirBytecodeTypeHandle);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadAttribute(MlirBytecodeDialectReader *reader,
                                       MlirBytecodeAttrHandle *result);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadType(MlirBytecodeDialectReader *reader,
                                  MlirBytecodeTypeHandle *result);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadVarInt(MlirBytecodeDialectReader *reader,
                                    uint64_t *result);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadSignedVarInt(MlirBytecodeDialectReader *reader,
                                          int64_t *result);

static MlirBytecodeStatus mlirBytecodeDialectReaderReadAPIntWithKnownWidth(
    MlirBytecodeDialectReader *reader, unsigned bitWidth,
    void *(*allocFn)(size_t size), MlirBytecodeAPInt *result);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadString(MlirBytecodeDialectReader *reader,
                                    MlirBytecodeBytesRef *result);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadBlob(MlirBytecodeDialectReader *reader,
                                  MlirBytecodeBytesRef *result);

static MlirBytecodeStatus
mlirBytecodeDialectReaderReadResourceHandle(MlirBytecodeDialectReader *reader,
                                            MlirBytecodeResourceHandle *handle);

// Structure that is just sufficient to capture the bits required to materialize
// a APInt.
struct MlirBytecodeAPInt {
  // If bitWidth <= 64 then value is populated, else
  union {
    uint64_t value;
    uint64_t *data;
  } U;

  unsigned bitWidth;
};

#ifdef __cplusplus
}
#endif

struct MlirBytecodeDialectReader {
  void *context;

  /// Stream being parsed.
  MlirBytecodeStream *stream;
};

MlirBytecodeStatus
mlirBytecodeDialectReaderReadAttribute(MlirBytecodeDialectReader *reader,
                                       MlirBytecodeAttrHandle *result) {
  MlirBytecodeStatus ret =
      mlirBytecodeParseHandle(reader->context, reader->stream, result);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  return mlirBytecodeParseAttribute(reader->context, *result);
}

MlirBytecodeStatus
mlirBytecodeDialectReaderReadType(MlirBytecodeDialectReader *reader,
                                  MlirBytecodeTypeHandle *result) {
  MlirBytecodeStatus ret =
      mlirBytecodeParseHandle(reader->context, reader->stream, result);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  return mlirBytecodeParseType(reader->context, *result);
}

MlirBytecodeStatus
mlirBytecodeDialectReaderReadVarInt(MlirBytecodeDialectReader *reader,
                                    uint64_t *result) {
  return mlirBytecodeParseVarInt(reader->context, reader->stream, result);
}

MlirBytecodeStatus
mlirBytecodeDialectReaderReadSignedVarInt(MlirBytecodeDialectReader *reader,
                                          int64_t *result) {
  return mlirBytecodeParseSignedVarInt(reader->context, reader->stream,
                                       result);
}

MlirBytecodeStatus mlirBytecodeDialectReaderReadAPIntWithKnownWidth(
    MlirBytecodeDialectReader *reader, unsigned bitWidth,
    void *(*allocFn)(size_t size), MlirBytecodeAPInt *result) {
  result->bitWidth = bitWidth;

  MlirBytecodeStatus ret;
  // Small values are encoded using a single byte.
  if (bitWidth <= 8) {
    uint8_t value;
    ret = mlirBytecodeParseByte(reader->context, reader->stream, &value);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    return ret;

    result->U.value = value;
    return mlirBytecodeSuccess();
  }

  // Large values up to 64 bits are encoded using a single varint.
  if (bitWidth <= 64) {
    ret = mlirBytecodeParseSignedVarInt(reader->context, reader->stream,
                                        (int64_t *)&result->U.value);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    return mlirBytecodeSuccess();
  }

  // Otherwise, for really big values we encode the array of active words in
  // the value.

  // Check if allocation function was provided.
  if (!allocFn)
    return mlirBytecodeUnhandled();

  uint64_t numActiveWords;
  ret = mlirBytecodeParseVarInt(reader->context, reader->stream,
                                &numActiveWords);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  result->U.data = (uint64_t *)allocFn(numActiveWords * sizeof(uint64_t));
  if (!result->U.data)
    return mlirBytecodeFailure();

  for (uint64_t i = 0; i < numActiveWords; ++i) {
    ret = mlirBytecodeParseSignedVarInt(reader->context, reader->stream,
                                        (int64_t *)&result->U.data[i]);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectReaderReadString(MlirBytecodeDialectReader *reader,
                                    MlirBytecodeBytesRef *result) {
  MlirBytecodeStringHandle handle;
  MlirBytecodeStatus ret =
      mlirBytecodeParseHandle(reader->context, reader->stream, &handle);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  return mlirBytecodeGetStringSectionValue(reader->context, handle, result);
}

MlirBytecodeStatus
mlirBytecodeDialectReaderReadBlob(MlirBytecodeDialectReader *reader,
                                  MlirBytecodeBytesRef *result) {
  MlirBytecodeStatus ret =
      mlirBytecodeDialectReaderReadVarInt(reader, &result->length);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  return mlirBytecodeParseBytes(reader->context, reader->stream,
                                result->length, &result->data);
}

MlirBytecodeStatus mlirBytecodeDialectReaderReadResourceHandle(
    MlirBytecodeDialectReader *reader, MlirBytecodeResourceHandle *handle) {
  return mlirBytecodeParseHandle(reader->context, reader->stream, handle);
}

#pragma GCC diagnostic pop

#endif // MLIRBC_DIALECTBYTECODEREADER_H
