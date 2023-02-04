//===-- Parse.c.inc - MLIR bytecode C parser ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// MLIR C bytecode event-driven parser.
//===----------------------------------------------------------------------===//

// This file contains the full implementation for an event-based MLIR bytecode
// parser. It is defined as a pure include-only implementation.
//
// Define types and functions required for parsing (see below) and simplest
// entry point is `mlirBytecodeParse`.
//
// Callbacks/functions to be implemented in instantiation accept an opaque
// pointer as first argument that gets propagated during parsing and can be used
// by instantiation for additional/parse local state capture.

// TODO: Support for big-endian architectures.

#include "BytecodeTypes.h"

#include <assert.h>
#include <inttypes.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(MLIRBC_DEBUG) || defined(MLIRBC_VERBOSE_ERROR)
#include <stdarg.h>
#endif
#if defined(MLIRBC_DEBUG)
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// - Configs
#define MLIRBC_IR_STACK_MAX_DEPTH 15 // TODO: add dynamic option.

// - Types needed to be defined before including this file.
typedef MlirBytecodeOperationState *MlirBytecodeOperationStateHandle;
typedef MlirBytecodeOperation *MlirBytecodeOperationHandle;

//===----------------------------------------------------------------------===//
// MLIR file parsing methods.

/// Populates the MlirBytecodeParserState contents for given in memory bytes.
/// Returns an empty file if population failed.
static MlirBytecodeParserState
mlirBytecodePopulateParserState(void *context, MlirBytecodeBytesRef bytes);

/// Parses the given MLIR file represented in memory `bytes`, calls the
/// appropriate callbacks during parsing and returns the parsed top-level block.
static MlirBytecodeStatus
mlirBytecodeParse(void *context, MlirBytecodeParserState *parserState,
                  void *retBlock);

/// Returns whether the given MlirBytecodeParserState structure is empty.
static bool mlirBytecodeParserStateEmpty(MlirBytecodeParserState *parserState);

//===----------------------------------------------------------------------===//
// Functions required to be defined:
#define MLIRBC_DEF static

// - IR section hooks.

/// Called upon entry of IR section with argumentless block to populate.
MLIRBC_DEF void mlirBytecodeIRSectionEnter(void *context, void *retBlock);

/// Hook to populate handle to a operation state or equivalent helper structure
// to / collect fields required to define operation.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePush(
    void *context, MlirBytecodeOpHandle name, MlirBytecodeLocHandle loc,
    MlirBytecodeOperationStateHandle *opState);

/// Called for an operation with attributes.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *context, MlirBytecodeOperationStateHandle, MlirBytecodeAttrHandle);

/// Called for operation with results.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeSize numResults);
/// Called per operation result type.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddResultType(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeTypeHandle type);

/// Called for operation with operands.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeSize numOperands);
/// Called per operation operand.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddOperand(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeValueHandle value);

/// Called for operations with regions.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *context, MlirBytecodeOperationStateHandle, uint64_t numRegions,
    bool isIsolatedFromAbove);

/// Called for operations with successors.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeSize numSuccessors);
/// Called per operation operand.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessor(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeHandle successor);

/// Called when finalizing population all operation state excluding
/// regions/blocks.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePop(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeOperationHandle *opHandle);

/// Called when entering a region with numBlocks blocks and numValues Values
/// (including values due to block args).
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeOperationRegionPush(void *context, MlirBytecodeOperationHandle,
                                size_t numBlocks, size_t numValues);

/// Called when entering block, the argument range consists of numArgs
/// pairs of type and location.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationBlockPush(
    void *context, MlirBytecodeOperationHandle, MlirBytecodeSize numArgs);
/// Called per block argument with type and location.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationBlockAddArgument(
    void *context, MlirBytecodeOperationHandle, MlirBytecodeTypeHandle,
    MlirBytecodeLocHandle);

/// Called when exiting the block.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeOperationBlockPop(void *context, MlirBytecodeOperationHandle);

/// Called when exiting a region.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeOperationRegionPop(void *context, MlirBytecodeOperationHandle);

/// Invoked when entering attributes subsection.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeAttributesPush(void *context, MlirBytecodeSize numAttributes);

/// Invoked with dialect and attribute with the attribute's range in memory.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeAssociateAttributeRange(
    void *context, MlirBytecodeAttrHandle, MlirBytecodeDialectHandle,
    MlirBytecodeBytesRef, bool hasCustom);

/// Invoked when entering types subsection.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeTypesPush(void *context,
                                                    MlirBytecodeSize numTypes);

/// Invoked with dialect and type with the type's range in memory.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeAssociateTypeRange(
    void *context, MlirBytecodeTypeHandle, MlirBytecodeDialectHandle,
    MlirBytecodeBytesRef, bool hasCustom);

/// Invoked when processing dialects.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeDialectsPush(void *context, MlirBytecodeSize numDialects);

/// Invoked with dialect handle and its corresponding string handle.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeDialectCallBack(
    void *context, MlirBytecodeDialectHandle, MlirBytecodeStringHandle);

/// Invoked when entering opname association section.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeDialectOpNames(void *context, MlirBytecodeSize numOps);

/// Invoked per opname with its dialect and string handle corresponding to its
/// name.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeDialectOpCallBack(
    void *context, MlirBytecodeOpHandle, MlirBytecodeDialectHandle,
    MlirBytecodeStringHandle);

/// Invoked when starting parsing of group of resources.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceDialectGroupEnter(
    void *context, MlirBytecodeDialectHandle dialect,
    MlirBytecodeSize numResources);

/// Invoked when starting parsing of group of resources.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceExternalGroupEnter(
    void *context, MlirBytecodeStringHandle groupKey,
    MlirBytecodeSize numResources);

/// Invoked per blob resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *context, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeSize alignment, MlirBytecodeBytesRef blob);
/// Invoked per bool resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *context, MlirBytecodeStringHandle resourceKey, const uint8_t);
/// Invoked per string resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceStringCallBack(
    void *context, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle);

/// Invoked when entering strings section with total number of strings.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeStringsPush(void *context, MlirBytecodeSize numStrings);

/// Invoked with string handle in string section and
/// bytes corresponding to the string per entry in string section.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeAssociateStringRange(
    void *context, MlirBytecodeStringHandle, MlirBytecodeBytesRef);

/// Called to process an attribute handle and convert it into an attribute.
/// This may result in triggering additional parsing and processing of
/// attributes and types.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeParseAttribute(void *context, MlirBytecodeAttrHandle handle);

/// Called to process a type handle and convert it into an attribute.
/// This may result in triggering additional parsing and processing of
/// attributes and types.
MLIRBC_DEF MlirBytecodeStatus
mlirBytecodeParseType(void *context, MlirBytecodeTypeHandle handle);

/// Returns the requested string from the string section.
// This isn't used here but in dialect type & attribute parsing.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeGetStringSectionValue(
    void *context, MlirBytecodeStringHandle idx, MlirBytecodeBytesRef *result);

#ifdef MLIRBC_VERBOSE_ERROR
/// Returns failure status post emitting error using same format message as
/// printf.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeEmitErrorImpl(void *context,
                                                        const char *fmt, ...);
#endif

//===----------------------------------------------------------------------===//
// MlirBytecodeStatus.

// Value representing the status of parsing. Status is either a success, failure
// unhandled or interrupted. Instances of MlirBytecodeStatus must only be
// inspected using the associated functions.
struct MlirBytecodeStatus {
  int8_t value;
};

/// Creates a status representing a success.
inline static MlirBytecodeStatus mlirBytecodeSuccess() {
  MlirBytecodeStatus res = {1};
  return res;
}

/// Creates a status representing a failure.
inline static MlirBytecodeStatus mlirBytecodeFailure() {
  MlirBytecodeStatus res = {0};
  return res;
}

/// Creates a status representing an unhandled case.
inline static MlirBytecodeStatus mlirBytecodeUnhandled() {
  MlirBytecodeStatus res = {2};
  return res;
}

/// Creates a status representing an interrupted iteration.
inline static MlirBytecodeStatus mlirBytecodeIterationInterrupt() {
  MlirBytecodeStatus res = {3};
  return res;
}

/// Checks if the given status represents a success.
inline static bool mlirBytecodeSucceeded(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeSuccess().value;
}

/// Checks if the given status represents a failure.
inline static bool mlirBytecodeFailed(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeFailure().value;
}

/// Checks if the given status represents a failure.
inline static bool mlirBytecodeInterrupted(MlirBytecodeStatus res) {
  return res.value == mlirBytecodeIterationInterrupt().value;
}

/// Checks if the given status represents a handled state.
inline static bool mlirBytecodeHandled(MlirBytecodeStatus res) {
  return mlirBytecodeSucceeded(res) || mlirBytecodeFailed(res);
}

/// Helper function to emit debugging info.
extern void mlirBytecodeEmitDebugImpl(const char *file, int line,
                                      const char *fmt, ...);

#ifdef __cplusplus
}
#endif

//===----------------------------------------------------------------------===//
// Implementation/implementation details.

//===----------------------------------------------------------------------===//
// Section parsing entry points.

/// Iterators over attributes and types, calling MlirBytecodeAttrCallBack and
/// MlirBytecodeTypeCallBack upon encountering Attribute or Type respectively.
/// Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseAttributeAndTypeSection(void *context,
                                         MlirBytecodeParserState *parserState);

/// Parses the dialect section, invoking MlirBytecodeDialectCallBack upon
/// dialect encountered and MlirBytecodeDialectOpCallBack per operation type in
/// dialect. Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseDialectSection(void *context,
                                MlirBytecodeParserState *parserState);

/// Parse IR section and returns populated block. The block args, operation and
/// region callback are invoked during bytecode in-order walk. Additionally
/// allows for passing in an opaque state.
///
/// The IR section parsing follows the nesting order:
///   op ->* regions ->* blocks
/// The caller is required to keep track of when all operations/blocks in
/// block/region have been processed and so parsing resumes at parent level.
/// Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseIRSection(void *context, MlirBytecodeParserState *parserState,
                           void *retBlock);

/// Parse the resource section, calling MlirBytecodeResourceCallBack upon
/// resources encountered. Returns whether failed.
static MlirBytecodeStatus
mlirBytecodeParseResourceSection(void *context,
                                 MlirBytecodeParserState *parserState);

//===----------------------------------------------------------------------===//
// Parsing utility methods.

/// Parse the next handle on the stream and increment stream.
static MlirBytecodeStatus mlirBytecodeParseHandle(void *context,
                                                  MlirBytecodeStream *stream,
                                                  MlirBytecodeHandle *result);

/// Skips the next n handles.
static MlirBytecodeStatus
mlirBytecodeSkipHandles(void *context, MlirBytecodeStream *stream, uint64_t n);

/// Parse byte from stream.
static MlirBytecodeStatus
mlirBytecodeParseByte(void *context, MlirBytecodeStream *stream, uint8_t *val);

/// Parse n bytes from stream.
static MlirBytecodeStatus mlirBytecodeParseBytes(void *context,
                                                 MlirBytecodeStream *stream,
                                                 size_t n, const uint8_t **val);

/// Parse uint64 on the stream and increment stream.
static MlirBytecodeStatus mlirBytecodeParseVarInt(void *context,
                                                  MlirBytecodeStream *stream,
                                                  uint64_t *result);

/// Parse int64 on the stream and increment stream.
static MlirBytecodeStatus
mlirBytecodeParseSignedVarInt(void *context, MlirBytecodeStream *stream,
                              int64_t *result);

/// Parse a variable length encoded integer whose low bit is used to encode a
/// flag, i.e: `(integerValue << 1) | (flag ? 1 : 0)`.
static MlirBytecodeStatus
mlirBytecodeParseVarIntWithFlag(void *context, MlirBytecodeStream *stream,
                                uint64_t *result, bool *flag);

/// Parse a null-terminated string.
static MlirBytecodeStatus
mlirBytecodeParseNullTerminatedString(void *context, MlirBytecodeStream *stream,
                                      MlirBytecodeBytesRef *str);

//===----------------------------------------------------------------------===//
// MlirBytecodeStream.

/// Create a stream from a range of bytes.
static MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef bytes);

/// Return whether stream is empty.
static bool mlirBytecodeStreamEmpty(MlirBytecodeStream *stream);

// Macros to allow capturing file and line from which invoked.
#define mlirBytecodeEmitDebug(...)                                             \
  mlirBytecodeEmitDebugImpl(__FILE__, __LINE__, __VA_ARGS__)

#ifdef MLIRBC_VERBOSE_ERROR
#define mlirBytecodeEmitError(context, ...)                                    \
  mlirBytecodeEmitErrorImpl(context, __VA_ARGS__)
#else
#define mlirBytecodeEmitError(context, ...) mlirBytecodeFailure()
#endif

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define MLIRBC_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define MLIRBC_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define MLIRBC_LIKELY(EXPR) (EXPR)
#define MLIRBC_UNLIKELY(EXPR) (EXPR)
#endif

// MLIRBC_ATTRIBUTE_NOINLINE - On compilers where we have a directive to do so,
// mark a method "not for inlining".
#if __has_attribute(noinline)
#define MLIRBC_ATTRIBUTE_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define MLIRBC_ATTRIBUTE_NOINLINE __declspec(noinline)
#else
#define MLIRBC_ATTRIBUTE_NOINLINE
#endif

// Convention followed in this file is to prefix implementation details with
// mbci_. But there is more to do to differentiate between internal and
// user-facing methods.

//===----------------------------------------------------------------------===//
// Bytecode constants.

enum {
  // This section contains strings referenced within the bytecode.
  mbci_kString = 0,

  // This section contains the dialects referenced within an IR module.
  mbci_kDialect = 1,

  // This section contains the attributes and types referenced within an IR
  // module.
  mbci_kAttrType = 2,

  // This section contains the offsets for the attribute and types within the
  // AttrType section.
  mbci_kAttrTypeOffset = 3,

  // This section contains the list of operations serialized into the bytecode,
  // and their nested regions/operations.
  mbci_kIR = 4,

  // This section contains the resources of the bytecode.
  mbci_kResource = 5,

  // This section contains the offsets of resources within the Resource
  // section.
  mbci_kResourceOffset = 6,

  // The total number of section types.
  mbci_kNumSections = 7,
};

#ifdef MLIRBC_VERBOSE_ERROR
static const char *mbci_sectionIDToString(uint8_t id) {
  const char *arr[] = {
      "String (0)",         //
      "Dialect (1)",        //
      "AttrType (2)",       //
      "AttrTypeOffset (3)", //
      "IR (4)",             //
      "Resource (5)",       //
      "ResourceOffset (6)"  //
  };
  assert(id < sizeof(arr) / sizeof(arr[0]));
  return arr[id];
}
#endif

static bool mbci_isSectionOptional(int id) {
  switch (id) {
  case mbci_kResource:
  case mbci_kResourceOffset:
    return true;
  case mbci_kString:
  case mbci_kDialect:
  case mbci_kAttrType:
  case mbci_kAttrTypeOffset:
  case mbci_kIR:
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Parser state definition and helpers.

struct MlirBytecodeParserState {
  uint64_t version;
  MlirBytecodeBytesRef producer;

  MlirBytecodeBytesRef sectionData[mbci_kNumSections];
};

//===----------------------------------------------------------------------===//
// Default implementations for verbose logging and debugging

#ifdef MLIRBC_DEBUG
__attribute__((weak)) void mlirBytecodeEmitDebugImpl(const char *file, int line,
                                                     const char *fmt, ...) {
  va_list args;
  fprintf(stderr, "%s:%d: ", file, line);
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
}
#else
void mlirBytecodeEmitDebugImpl(const char *file, int line, const char *fmt,
                               ...) {
  (void)fmt;
}
#endif

MlirBytecodeStream mlirBytecodeStreamCreate(MlirBytecodeBytesRef bytes) {
  MlirBytecodeStream ret;
  ret.start = ret.pos = bytes.data;
  ret.end = ret.start + bytes.length;
  return ret;
}

// Represents simple parser state.
static bool mlirBytecodeStreamEmpty(MlirBytecodeStream *stream) {
  return stream->pos >= stream->end;
}

static MlirBytecodeStatus
mlirBytecodeParseByte(void *context, MlirBytecodeStream *stream, uint8_t *val) {
  if (MLIRBC_UNLIKELY(mlirBytecodeStreamEmpty(stream)))
    return mlirBytecodeFailure();
  *val = *stream->pos++;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mlirBytecodeParseBytes(void *context,
                                                 MlirBytecodeStream *stream,
                                                 size_t n,
                                                 const uint8_t **result) {
  *result = stream->pos;
  stream->pos += n;
  if (MLIRBC_UNLIKELY(stream->pos > stream->end))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_skipBytes(void *context,
                                         MlirBytecodeStream *stream, size_t n) {
  stream->pos += n;
  if (MLIRBC_UNLIKELY(stream->pos > stream->end))
    return mlirBytecodeFailure();
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_skipVarInts(void *context, MlirBytecodeStream *stream, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t head;
    MlirBytecodeStatus ret = mlirBytecodeParseByte(context, stream, &head);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeFailure();
    int numBytes = (MLIRBC_UNLIKELY(head == 0)) ? 8 : __builtin_ctz(head);
    ret = mbci_skipBytes(context, stream, numBytes);
    if (mlirBytecodeFailed(ret))
      return mlirBytecodeFailure();
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseHandle(void *context,
                                           MlirBytecodeStream *stream,
                                           MlirBytecodeAttrHandle *result) {
  return mlirBytecodeParseVarInt(context, stream, &result->id);
}

__attribute__((used)) MlirBytecodeStatus
mlirBytecodeSkipHandles(void *context, MlirBytecodeStream *stream, uint64_t n) {
  return mbci_skipVarInts(context, stream, n);
}

MLIRBC_ATTRIBUTE_NOINLINE MlirBytecodeStatus mbci_parseVarInts(
    void *context, MlirBytecodeStream *stream, uint8_t head, uint64_t *result) {

  uint32_t numBytes = __builtin_ctz(head);
  assert(numBytes > 0 && numBytes <= 7 &&
         "unexpected number of trailing zeros in varint encoding");

  const uint8_t *res;
  MlirBytecodeStatus ret =
      mlirBytecodeParseBytes(context, stream, numBytes, &res);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  uint8_t *ptr = (uint8_t *)(result) + 1;
  for (uint32_t i = 0; i < numBytes; ++i) {
    ptr[i] = res[i];
  }

  // Shift out the low-order bits that were used to mark how the value was
  // encoded.
  *result >>= (numBytes + 1);
  return ret;
}

MlirBytecodeStatus mlirBytecodeParseVarInt(void *context,
                                           MlirBytecodeStream *stream,
                                           uint64_t *result) {
  // Compute the number of bytes needed to encode the value. Each byte can hold
  // up to 7-bits of data. We only check up to the number of bits we can encode
  // in the first byte (8).
  uint8_t head;
  if (mlirBytecodeFailed(mlirBytecodeParseByte(context, stream, &head)))
    return mlirBytecodeFailure();
  *result = head;
  if (MLIRBC_LIKELY(*result & 1)) {
    *result >>= 1;
    return mlirBytecodeSuccess();
  }

  if (MLIRBC_UNLIKELY(*result == 0)) {
    uint32_t numBytes = sizeof(*result);
    const uint8_t *res;
    MlirBytecodeStatus ret =
        mlirBytecodeParseBytes(context, stream, numBytes, &res);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeFailure();
    uint8_t *ptr = (uint8_t *)(result);
    for (uint32_t i = 0; i < numBytes; ++i) {
      ptr[i] = res[i];
    }
    return mlirBytecodeSuccess();
  }
  return mbci_parseVarInts(context, stream, head, result);
}

MlirBytecodeStatus mlirBytecodeParseVarIntWithFlag(void *context,
                                                   MlirBytecodeStream *stream,
                                                   uint64_t *result,
                                                   bool *flag) {
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(context, stream, result);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  *flag = *result & 1;
  *result >>= 1;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeParseSignedVarInt(void *context,
                                                 MlirBytecodeStream *stream,
                                                 int64_t *result) {
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(context, stream, (uint64_t *)result);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  *result = (*result >> 1) ^ (~(*result & 1) + 1);
  return mlirBytecodeSuccess();
}

// Align the current reader position to the specified alignment.
static MlirBytecodeStatus
mbci_alignTo(void *context, MlirBytecodeStream *stream, uint32_t alignment) {
  bool isPowerOf2 = (alignment != 0) && ((alignment & (alignment - 1)) == 0);
  if (MLIRBC_UNLIKELY(!isPowerOf2))
    return mlirBytecodeEmitError(context,
                                 "expected alignment to be a power-of-two");

  // An arbitrary value used to fill alignment padding.
  const uint8_t kAlignmentByte = 0xCB;

  // Shift the reader position to the next alignment boundary.
  while ((uintptr_t)stream->pos & ((uintptr_t)alignment - 1)) {
    uint8_t padding;
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(
            mlirBytecodeParseByte(context, stream, &padding))))
      return mlirBytecodeFailure();
    if (padding != kAlignmentByte) {
      return mlirBytecodeEmitError(
          context, "expected alignment byte (0x%x), but got: '0x%x'",
          kAlignmentByte, padding);
    }
  }

  if (MLIRBC_UNLIKELY((uint64_t)(stream->pos) % alignment != 0))
    return mlirBytecodeEmitError(
        context,
        "expected data iterator aligned to %d, but got pointer '0x" PRIx64 "'",
        alignment, (void *)stream->pos);
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_parseSections(void *context, MlirBytecodeStream *stream,
                   MlirBytecodeBytesRef sectionData[mbci_kNumSections]) {
  uint8_t byte;
  uint64_t length;
  MlirBytecodeStatus ret = mlirBytecodeParseByte(context, stream, &byte);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return ret;
  ret = mlirBytecodeParseVarInt(context, stream, &length);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return ret;
  uint8_t sectionID = byte & 0x7f;

  mlirBytecodeEmitDebug("parsing section %d of size %" PRIu64, sectionID, length);
  if (MLIRBC_UNLIKELY(sectionID >= mbci_kNumSections))
    return mlirBytecodeEmitError(context, "invalid section ID: %d", sectionID);
  if (MLIRBC_UNLIKELY(sectionData[sectionID].data != NULL)) {
    return mlirBytecodeEmitError(context, "duplicate top-level section: %s",
                                 mbci_sectionIDToString(sectionID));
  }

  bool isAligned = byte >> 7;
  if (isAligned) {
    uint64_t alignment;
    ret = mlirBytecodeParseVarInt(context, stream, &alignment);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return ret;
    ret = mbci_alignTo(context, stream, alignment);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return ret;
  }

  sectionData[sectionID].data = stream->pos;
  sectionData[sectionID].length = length;
  return mbci_skipBytes(context, stream, length);
}

MlirBytecodeStatus
mlirBytecodeParseNullTerminatedString(void *context, MlirBytecodeStream *stream,
                                      MlirBytecodeBytesRef *str) {
  const uint8_t *startIt = (const uint8_t *)stream->pos;
  const uint8_t *nulIt =
      (const uint8_t *)memchr(startIt, 0, stream->end - stream->pos);
  if (MLIRBC_UNLIKELY(!nulIt)) {
    return mlirBytecodeEmitError(
        context, "malformed null-terminated string, no null character found");
  }

  str->data = startIt;
  str->length = nulIt - startIt;
  stream->pos += str->length + 1;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStream
mbci_populateStreamForSection(MlirBytecodeBytesRef ref) {
  MlirBytecodeStream stream;
  stream.start = stream.pos = ref.data;
  stream.end = stream.start + ref.length;
  return stream;
}

static MlirBytecodeBytesRef
mbci_getSection(const MlirBytecodeParserState *const file, int index) {
  return file->sectionData[index];
}

MlirBytecodeStatus
mlirBytecodeForEachString(void *context,
                          const MlirBytecodeParserState *const parserState) {
  const MlirBytecodeBytesRef stringSection =
      mbci_getSection(parserState, mbci_kString);
  MlirBytecodeStream stream = mbci_populateStreamForSection(stringSection);

  uint64_t numStrings;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(context, &stream, &numStrings);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "failed to parse number of strings");
  ret = mlirBytecodeStringsPush(context, numStrings);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  // Parse each of the strings. The sizes of the strings are encoded in reverse
  // order, so that's the order we populate the table.
  size_t stringDataEndOffset = stringSection.length;
  for (uint64_t i = numStrings; i > 0; --i) {
    uint64_t stringSize;
    ret = mlirBytecodeParseVarInt(context, &stream, &stringSize);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "failed to parse string size of string %d", i - 1);
    if (MLIRBC_UNLIKELY(stringDataEndOffset < stringSize)) {
      return mlirBytecodeEmitError(
          context, "string size exceeds the available data size");
    }

    // Extract the string from the data, dropping the null character.
    size_t stringOffset = stringDataEndOffset - stringSize;
    MlirBytecodeBytesRef str;
    str.data = stringSection.data + stringOffset;
    str.length = stringSize - 1;
    ret = mlirBytecodeAssociateStringRange(
        context, (MlirBytecodeStringHandle){.id = i - 1}, str);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context, "string callback failed");
    stringDataEndOffset = stringOffset;
  }

  // Check that the only remaining data was for the strings, i.e. the reader
  // should be at the same offset as the first string.
  if (MLIRBC_UNLIKELY(stream.pos != (stream.start + stringDataEndOffset))) {
    return mlirBytecodeEmitError(
        context, "unexpected trailing data between the offsets for strings "
                 "and their data");
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseDialectSection(void *context,
                                MlirBytecodeParserState *parserState) {
  const MlirBytecodeBytesRef dialectSection =
      mbci_getSection(parserState, mbci_kDialect);
  MlirBytecodeStream stream = mbci_populateStreamForSection(dialectSection);
  mlirBytecodeEmitDebug("parsing dialect section of length %" PRIu64 "",
                        dialectSection.length);

  uint64_t numDialects;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(context, &stream, &numDialects);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "unable to parse number of dialects");
  mlirBytecodeEmitDebug("number of dialects = %d", numDialects);
  ret = mlirBytecodeDialectsPush(context, numDialects);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return ret;

  uint64_t dialectName;
  for (uint64_t i = 0; i < numDialects; ++i) {
    ret = mlirBytecodeParseVarInt(context, &stream, &dialectName);
    if (mlirBytecodeFailed(ret))
      return mlirBytecodeEmitError(context, "unable to parse dialect %d", i);
    if (mlirBytecodeFailed(mlirBytecodeDialectCallBack(
            context, (MlirBytecodeDialectHandle){.id = i},
            (MlirBytecodeStringHandle){.id = dialectName})))
      return mlirBytecodeFailure();
  }

  const uint8_t *resetPos = stream.pos;
  // TODO: See if possible to avoid reparsing.
  MlirBytecodeSize numOps = 0;
  while (!mlirBytecodeStreamEmpty(&stream)) {
    uint64_t dialect, numOpNames;
    mlirBytecodeParseVarInt(context, &stream, &dialect);
    mlirBytecodeParseVarInt(context, &stream, &numOpNames);
    ret = mbci_skipVarInts(context, &stream, numOpNames);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    numOps += numOpNames;
  }
  ret = mlirBytecodeDialectOpNames(context, numOps);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  stream.pos = resetPos;

  MlirBytecodeOpHandle op = {.id = 0};
  while (!mlirBytecodeStreamEmpty(&stream)) {
    uint64_t dialect, numOpNames;
    mlirBytecodeParseVarInt(context, &stream, &dialect);
    mlirBytecodeParseVarInt(context, &stream, &numOpNames);

    mlirBytecodeEmitDebug("parsing for dialect %d %d ops", dialect, numOpNames);
    for (uint64_t j = 0; j < numOpNames; ++j) {
      uint64_t opName;
      ret = mlirBytecodeParseVarInt(context, &stream, &opName);
      assert(mlirBytecodeSucceeded(ret) &&
             "impossible; would have failed above");
      ret = mlirBytecodeDialectOpCallBack(
          context, op, (MlirBytecodeDialectHandle){.id = dialect},
          (MlirBytecodeStringHandle){.id = opName});
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      ++op.id;
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseAttributeAndTypeSection(void *context,
                                         MlirBytecodeParserState *parserState) {
  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(parserState, mbci_kAttrTypeOffset);
  MlirBytecodeStream offsetStream =
      mbci_populateStreamForSection(offsetSection);
  const MlirBytecodeBytesRef atSection =
      mbci_getSection(parserState, mbci_kAttrType);
  MlirBytecodeStream atPP = mbci_populateStreamForSection(atSection);

  uint64_t numAttrs, numTypes;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(context, &offsetStream, &numAttrs);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "invalid number of attributes");
  ret = mlirBytecodeParseVarInt(context, &offsetStream, &numTypes);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "invalid number of types");
  mlirBytecodeEmitDebug("parsing %" PRIu64 " attributes", numAttrs);
  ret = mlirBytecodeAttributesPush(context, numAttrs);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  uint64_t i = 0;
  while (i < numAttrs) {
    uint64_t dialect;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &dialect);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "invalid dialect handle while parsing attributes");

    uint64_t numElements;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &numElements);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "invalid number of elements in attr offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t length;
      bool hasCustomEncoding;
      ret = mlirBytecodeParseVarIntWithFlag(context, &offsetStream, &length,
                                            &hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid attr offset");

      MlirBytecodeBytesRef attr = {.data = atPP.pos, .length = length};

      // Verify that the offset is actually valid.
      if (MLIRBC_UNLIKELY(atPP.pos + length > atPP.end)) {
        return mlirBytecodeEmitError(
            context,
            "attribute or Type entry offset points past the end of section");
      }

      // Parse & associate dialect.attr[j] with `attr`
      ret = mlirBytecodeAssociateAttributeRange(
          context, (MlirBytecodeAttrHandle){.id = i++},
          (MlirBytecodeDialectHandle){.id = dialect}, attr, hasCustomEncoding);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret))) {
        mlirBytecodeEmitDebug("attr callback failed");
        return ret;
      }
      if (MLIRBC_UNLIKELY(mlirBytecodeInterrupted(ret)))
        return ret;
      // Unhandled attributes are not considered error.

      if (mlirBytecodeFailed(mbci_skipBytes(context, &atPP, length)))
        return mlirBytecodeEmitError(context, "invalid attr offset");
    }
  }

  mlirBytecodeEmitDebug("parsing %" PRIu64 " types", numTypes);
  i = 0;
  ret = mlirBytecodeTypesPush(context, numTypes);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  while (i < numTypes) {
    uint64_t dialect;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &dialect);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "invalid dialect handle while parsing types");

    uint64_t numElements;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &numElements);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "invalid number of elements in type offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t offset;
      bool hasCustomEncoding;
      ret = mlirBytecodeParseVarIntWithFlag(context, &offsetStream, &offset,
                                            &hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid type offset");

      MlirBytecodeBytesRef type = {.data = atPP.pos, .length = offset};
      // Verify that the offset is actually valid.
      if (MLIRBC_UNLIKELY(atPP.pos + offset > atPP.end)) {
        return mlirBytecodeEmitError(
            context,
            "attribute or Type entry offset points past the end of section");
      }

      ret = mlirBytecodeAssociateTypeRange(
          context, (MlirBytecodeTypeHandle){.id = i++},
          (MlirBytecodeDialectHandle){.id = dialect}, type, hasCustomEncoding);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret))) {
        mlirBytecodeEmitDebug("type callback failed");
        return ret;
      }
      if (MLIRBC_UNLIKELY(mlirBytecodeInterrupted(ret)))
        return ret;
      // Unhandled attributes are not considered error.

      ret = mbci_skipBytes(context, &atPP, offset);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid type offset");
    }
  }

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeParseResourceSection(void *context,
                                 MlirBytecodeParserState *const parserState) {
  mlirBytecodeEmitDebug("entering resource section");

  enum MlirBytecodeAsmResourceEntryKind {
    /// A blob of data with an accompanying alignment.
    kMlirBytecodeResourceEntryBlob,
    /// A boolean value.
    kMlirBytecodeResourceEntryBool,
    /// A string value.
    kMlirBytecodeResourceEntryString,
  };
  typedef enum MlirBytecodeAsmResourceEntryKind
      MlirBytecodeAsmResourceEntryKind;

  const MlirBytecodeBytesRef offsetSection =
      mbci_getSection(parserState, mbci_kResourceOffset);
  MlirBytecodeStream offsetStream =
      mbci_populateStreamForSection(offsetSection);
  const MlirBytecodeBytesRef resSection =
      mbci_getSection(parserState, mbci_kResource);
  MlirBytecodeStream resPP = mbci_populateStreamForSection(resSection);

  uint64_t numExternalResourceGroups;
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(context, &offsetStream,
                                                   &numExternalResourceGroups);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(
        context, "invalid/missing number of external resource groups");

  uint64_t i = 0;
  while (!mlirBytecodeStreamEmpty(&offsetStream)) {
    MlirBytecodeStringHandle groupKey;
    ret = mlirBytecodeParseHandle(context, &offsetStream, &groupKey);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context, "invalid/missing "
                                            "resource group key");

    uint64_t numResources;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &numResources);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context,
                                   "invalid/missing number of resources");

    ret = i < numExternalResourceGroups
              ? mlirBytecodeResourceExternalGroupEnter(context, groupKey,
                                                       numResources)
              : mlirBytecodeResourceDialectGroupEnter(context, groupKey,
                                                      numResources);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;

    for (uint64_t j = 0; j < numResources; ++j) {
      uint64_t resourceKey;
      ret = mlirBytecodeParseVarInt(context, &offsetStream, &resourceKey);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid/missing resource key");
      uint64_t size;
      ret = mlirBytecodeParseVarInt(context, &offsetStream, &size);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid/missing resource size");
      uint8_t kind;
      ret = mlirBytecodeParseByte(context, &offsetStream, &kind);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid/missing resource kind");

      MlirBytecodeAsmResourceEntryKind resKind =
          (MlirBytecodeAsmResourceEntryKind)kind;
      switch (resKind) {
      case kMlirBytecodeResourceEntryBlob: {
        uint64_t alignment;
        ret = mlirBytecodeParseVarInt(context, &resPP, &alignment);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid/missing resource alignment");
        uint64_t size;
        ret = mlirBytecodeParseVarInt(context, &resPP, &size);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid/missing resource size");
        ret = mbci_alignTo(context, &resPP, alignment);
        const uint8_t *start = resPP.pos;
        if (mlirBytecodeFailed(ret) ||
            mlirBytecodeFailed(mbci_skipBytes(context, &resPP, size)))
          return mlirBytecodeFailure();
        MlirBytecodeBytesRef blob = {.data = start, .length = size};
        ret = mlirBytecodeResourceBlobCallBack(
            context, (MlirBytecodeStringHandle){.id = resourceKey}, alignment,
            blob);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
        break;
      }
      case kMlirBytecodeResourceEntryBool: {
        uint8_t value;
        ret = mlirBytecodeParseByte(context, &resPP, &value);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid/missing resource entry");
        ret = mlirBytecodeResourceBoolCallBack(
            context, (MlirBytecodeStringHandle){.id = resourceKey}, value);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
        break;
      }
      case kMlirBytecodeResourceEntryString: {
        MlirBytecodeStringHandle value;
        ret = mlirBytecodeParseHandle(context, &resPP, &value);
        mlirBytecodeEmitDebug("string = %" PRIu64, value.id);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid/missing resource value");
        ret = mlirBytecodeResourceStringCallBack(
            context, (MlirBytecodeStringHandle){.id = resourceKey}, value);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
        break;
      }
      }
    }
  }
  return mlirBytecodeSuccess();
}

enum {
  // Sentinel value to indicate unset number of blocks & ops remaining.
  mbci_MlirIRSectionStackUnset = -1,
};

// We keep the following on the stack for a op with regions when descending into
// regions:
struct mbci_MlirIRSectionStackEntry {
  union {
    // State of parent op as currently populated.
    MlirBytecodeOperationStateHandle opState;
    MlirBytecodeOperationHandle op;
  };

  // Number of regions to still complete parsing.
  // Number of regions is assumed to be relatively small (<= 32767).
  uint16_t numRegionsRemaining : 15;

  // Number of blocks to still complete parsing.
  // Number of regions is assumed to be relatively small (<= 65536).
  uint16_t numBlocksRemaining;

  // Number of ops remaining.
  uint32_t numOpsRemaining;
};
typedef struct mbci_MlirIRSectionStackEntry mbci_MlirIRSectionStackEntry;

// IR section parsing stack. Instruction without region does not result in
// pushing anything on stack, for operations with regions:
struct mbci_MlirIRSectionStack {
  int top;

  mbci_MlirIRSectionStackEntry data[MLIRBC_IR_STACK_MAX_DEPTH + 1];
};
typedef struct mbci_MlirIRSectionStack mbci_MlirIRSectionStack;

static mbci_MlirIRSectionStackEntry *
mbci_getStackTop(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top];
}
static mbci_MlirIRSectionStackEntry *
mbci_getStackWIP(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top + 1];
}

// Initialize stack entry with required state to resume. `op` is initialized
// already and so not passed in.
static MlirBytecodeStatus
mbci_mlirIRSectionStackPush(void *context, mbci_MlirIRSectionStack *stack) {
  if (MLIRBC_UNLIKELY(stack->top == MLIRBC_IR_STACK_MAX_DEPTH)) {
    mlirBytecodeEmitError(context, "IR stack max depth exceeded");
    return mlirBytecodeIterationInterrupt();
  }
  ++stack->top;
  mbci_MlirIRSectionStackEntry *it = mbci_getStackTop(stack);
  it->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
  it->numOpsRemaining = mbci_MlirIRSectionStackUnset;

  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mbci_mlirIRSectionStackPop(void *context, mbci_MlirIRSectionStack *stack) {
  --stack->top;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mbci_parseBlock(void *context,
                                          MlirBytecodeParserState *parserState,
                                          MlirBytecodeStream *stream,
                                          mbci_MlirIRSectionStack *stack) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);
  bool hasArgs;
  uint64_t numOpsRemaining;
  MlirBytecodeStatus ret = mlirBytecodeParseVarIntWithFlag(
      context, stream, &numOpsRemaining, &hasArgs);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context,
                                 "unable to parse block's number of args");
  top->numOpsRemaining = numOpsRemaining;
  mlirBytecodeEmitDebug("num ops = %d and block %s", numOpsRemaining,
                        hasArgs ? "has args" : "has no args");

  uint64_t numArgs = 0;
  if (hasArgs) {
    ret = mlirBytecodeParseVarInt(context, stream, &numArgs);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context,
                                   "invalid/missing number of block args");
    ret = mlirBytecodeOperationBlockPush(context, top->op, numArgs);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context,
                                   "invalid/missing number of block args");
  }

  for (MlirBytecodeSize i = 0; i < numArgs; ++i) {
    MlirBytecodeTypeHandle type;
    ret = mlirBytecodeParseHandle(context, stream, &type);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    MlirBytecodeLocHandle loc;
    ret = mlirBytecodeParseHandle(context, stream, &loc);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;

    ret = mlirBytecodeOperationBlockAddArgument(context, top->op, type, loc);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }
  return ret;
}

static MlirBytecodeStatus mbci_parseRegion(void *context,
                                           MlirBytecodeStream *stream,
                                           mbci_MlirIRSectionStack *stack,
                                           void *state) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);
  uint64_t numBlocksRemaining;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(context, stream, &numBlocksRemaining);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "invalid number of blocks");
  top->numBlocksRemaining = numBlocksRemaining;
  top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
  if (top->numBlocksRemaining == 0)
    return mlirBytecodeSuccess();

  uint64_t numValues;
  ret = mlirBytecodeParseVarInt(context, stream, &numValues);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context,
                                 "invalid/missing number of values in region");
  mlirBytecodeEmitDebug("numBlocks = %" PRIu64 " numValues = %" PRIu64,
                        numBlocksRemaining, numValues);

  return mlirBytecodeOperationRegionPush(state, top->op,
                                         top->numBlocksRemaining, numValues);
}

static MlirBytecodeStatus
mbci_parseOperation(void *context, MlirBytecodeParserState *parserState,
                    MlirBytecodeStream *stream,
                    mbci_MlirIRSectionStack *stack) {
  enum OpEncodingMask {
    kHasAttrs = 0x01,
    kHasResults = 0x02,
    kHasOperands = 0x04,
    kHasSuccessors = 0x08,
    kHasInlineRegions = 0x10,
  };

  MlirBytecodeOpHandle name;
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(context, stream, &name.id);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "invalid operation");

  uint8_t encodingMask;
  ret = mlirBytecodeParseByte(context, stream, &encodingMask);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "invalid encoding mask");

  MlirBytecodeLocHandle loc;
  ret = mlirBytecodeParseVarInt(context, stream, &loc.id);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(context, "invalid operation location");

  mbci_MlirIRSectionStackEntry *cur = mbci_getStackWIP(stack);
  ret = mlirBytecodeOperationStatePush(context, name, loc, &cur->opState);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;

  if (encodingMask & kHasAttrs) {
    MlirBytecodeAttrHandle attrDict;
    ret = mlirBytecodeParseHandle(context, stream, &attrDict);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret

                                           ))) {
      return mlirBytecodeEmitError(context, "invalid op attribute handle");
    }
    ret = mlirBytecodeOperationStateAddAttributeDictionary(
        context, cur->opState, attrDict);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitDebug("unable to add attributes"), ret;
  }

  // Parsing all the variadic sizes.
  if (encodingMask & kHasResults) {
    uint64_t numResults = 0;
    ret = mlirBytecodeParseVarInt(context, stream, &numResults);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitError(context, "invalid result types"), ret;
    ret = mlirBytecodeOperationStateAddResultTypes(context, cur->opState,
                                                   numResults);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitError(context, "invalid result types"), ret;
    for (MlirBytecodeSize i = 0; i < numResults; ++i) {
      MlirBytecodeTypeHandle type;
      ret = mlirBytecodeParseHandle(context, stream, &type);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      // TODO: Verify in range here already.
      ret =
          mlirBytecodeOperationStateAddResultType(context, cur->opState, type);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
    }
  }

  if (encodingMask & kHasOperands) {
    uint64_t numOperands = 0;
    ret = mlirBytecodeParseVarInt(context, stream, &numOperands);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitError(context, "invalid operands"), ret;
    ret = mlirBytecodeOperationStateAddOperands(context, cur->opState,
                                                numOperands);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitError(context, "invalid operands"), ret;
    for (MlirBytecodeSize i = 0; i < numOperands; ++i) {
      MlirBytecodeValueHandle operand;
      ret = mlirBytecodeParseHandle(context, stream, &operand);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      // TODO: Verify in range here already.
      ret =
          mlirBytecodeOperationStateAddOperand(context, cur->opState, operand);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
    }
  }

  if (encodingMask & kHasSuccessors) {
    uint64_t numSuccessors = 0;
    ret = mlirBytecodeParseVarInt(context, stream, &numSuccessors);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitError(context, "invalid successors"), ret;
    ret = mlirBytecodeOperationStateAddSuccessors(context, cur->opState,
                                                  numSuccessors);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return mlirBytecodeEmitError(context, "invalid successors"), ret;
    for (MlirBytecodeSize i = 0; i < numSuccessors; ++i) {
      MlirBytecodeHandle successor;
      ret = mlirBytecodeParseHandle(context, stream, &successor);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      // TODO: Verify in range here already.
      ret = mlirBytecodeOperationStateAddSuccessor(context, cur->opState,
                                                   successor);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
    }
  }

  if (encodingMask & kHasInlineRegions) {
    uint64_t numRegions = 0;
    bool isIsolatedFromAbove = false;
    ret = mlirBytecodeParseVarIntWithFlag(context, stream, &numRegions,
                                          &isIsolatedFromAbove);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
      return mlirBytecodeEmitError(context, "invalid number of regions");
    }
    cur->numRegionsRemaining = numRegions;
    ret = mlirBytecodeOperationStateAddRegions(context, cur->opState,
                                               numRegions, isIsolatedFromAbove);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }

  // Completed parsing of op proper, regions potentially remaining.
  MlirBytecodeOperationHandle opHandle;
  ret = mlirBytecodeOperationStatePop(context, cur->opState, &opHandle);
  if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
    return ret;
  cur->op = opHandle;

  if (encodingMask & kHasInlineRegions)
    return mbci_mlirIRSectionStackPush(context, stack);

  // `cur` is the scratchpad for parsing op without region while `top` is where
  // it is being parsed.
  --mbci_getStackTop(stack)->numOpsRemaining;
  return ret;
}

MlirBytecodeStatus
mlirBytecodeParseIRSection(void *context, MlirBytecodeParserState *parserState,
                           void *retBlock) {
  mlirBytecodeIRSectionEnter(context, retBlock);
  const MlirBytecodeBytesRef irSection = mbci_getSection(parserState, mbci_kIR);
  MlirBytecodeStream stream = mbci_populateStreamForSection(irSection);

  bool hasArgs;
  uint64_t numOps;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarIntWithFlag(context, &stream, &numOps, &hasArgs);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  if (MLIRBC_UNLIKELY(hasArgs))
    return mlirBytecodeEmitError(
        context, "IR section top-level block may not have block args");

  // Initialize with parsing of top-level.
  mbci_MlirIRSectionStack stack = {.top = 0};
  stack.data[0].numRegionsRemaining = 1;
  stack.data[0].numBlocksRemaining = 1;
  stack.data[0].numOpsRemaining = numOps;

  while (stack.data[0].numOpsRemaining) {
    mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(&stack);

    MlirBytecodeStatus ret;
    bool next = false;
    if (top->numOpsRemaining == 0) {
      mlirBytecodeEmitDebug("no ops remaining in block");
      ret = mlirBytecodeOperationBlockPop(context, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;

      --top->numBlocksRemaining;
      top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numBlocksRemaining == 0) {
      mlirBytecodeEmitDebug("no blocks remaining in region");
      ret = mlirBytecodeOperationRegionPop(context, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;

      --top->numRegionsRemaining;
      top->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numRegionsRemaining == 0) {
      mlirBytecodeEmitDebug("no regions remaining in op");
      mbci_mlirIRSectionStackPop(context, &stack);
      --mbci_getStackTop(&stack)->numOpsRemaining;
      next = true;
    }
    if (next)
      continue;

    if (top->numBlocksRemaining == (uint16_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting region parsing");
      ret = mbci_parseRegion(context, &stream, &stack, context);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    if (top->numOpsRemaining == (uint32_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting block parsing");
      ret = mbci_parseBlock(context, parserState, &stream, &stack);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    mlirBytecodeEmitDebug("parsing operation (remaining regions: %" PRIu64
                          " blocks: %" PRIu64 " ops: %" PRIu64 ")",
                          mbci_getStackTop(&stack)->numRegionsRemaining,
                          mbci_getStackTop(&stack)->numBlocksRemaining,
                          mbci_getStackTop(&stack)->numOpsRemaining);
    ret = mbci_parseOperation(context, parserState, &stream, &stack);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }
  return mlirBytecodeSuccess();
}

bool mlirBytecodeParserStateEmpty(MlirBytecodeParserState *file) {
  return file->producer.length == 0;
}

MlirBytecodeParserState
mlirBytecodePopulateParserState(void *context,
                                const MlirBytecodeBytesRef bytes) {
  MlirBytecodeParserState ret;
  memset(&ret, 0, sizeof(MlirBytecodeParserState));
  MlirBytecodeStream stream = mbci_populateStreamForSection(bytes);

  const uint8_t *magic;
  if (mlirBytecodeFailed(mlirBytecodeParseBytes(context, &stream, 4, &magic)))
    return mlirBytecodeEmitError(context, "unable to read 4 byte magic code"),
           ret;
  if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 0xef || magic[3] != 'R')
    return mlirBytecodeEmitError(context, "invalid file magic code"), ret;

  // Parse the bytecode version and producer.
  MlirBytecodeBytesRef producer;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarInt(context, &stream, &ret.version)) ||
      mlirBytecodeFailed(
          mlirBytecodeParseNullTerminatedString(context, &stream, &producer)))
    return mlirBytecodeEmitError(context, "invalid version or producer"), ret;
  mlirBytecodeEmitDebug("bytecode producer: %s", producer.data);

  while (!mlirBytecodeStreamEmpty(&stream)) {
    // Read the next section from the bytecode.
    if (mlirBytecodeFailed(
            mbci_parseSections(context, &stream, ret.sectionData)))
      return mlirBytecodeEmitError(context, "invalid section"), ret;
  }

  // Check that all of the required sections were found
  for (int i = 0; i < mbci_kNumSections; ++i) {
    if (mbci_getSection(&ret, i).data == NULL && !mbci_isSectionOptional(i)) {
      return mlirBytecodeEmitError(context,
                                   "missing data for top-level section: ",
                                   mbci_sectionIDToString(i)),
             ret;
    }
  }

  // Mark as valid.
  ret.producer = producer;
  return ret;
}

MlirBytecodeStatus mlirBytecodeParse(void *context,
                                     MlirBytecodeParserState *parserState,
                                     void *retBlock) {
  if (mlirBytecodeParserStateEmpty(parserState))
    return mlirBytecodeSuccess();

  // Process the string section first.
  MlirBytecodeStatus ret = mlirBytecodeForEachString(context, parserState);

  if (!mlirBytecodeSucceeded(ret))
    return ret;

  // Process the dialect section.
  ret = mlirBytecodeParseDialectSection(context, parserState);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  // Process the resource section if present.
  ret = mlirBytecodeParseResourceSection(context, parserState);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  // Process the attribute and type section.
  ret = mlirBytecodeParseAttributeAndTypeSection(context, parserState);
  if (!mlirBytecodeSucceeded(ret))
    return ret;

  // Finally, process the IR section.
  return mlirBytecodeParseIRSection(context, parserState, retBlock);
}
