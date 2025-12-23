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

// Debug output disabled for production use
// #define MLIRBC_DEBUG
// #define MLIRBC_VERBOSE_ERROR  // Required for proper error diagnostics

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

// Use-list order storage limits (inline storage to avoid dynamic allocation).
#define MLIRBC_MAX_PENDING_USE_LISTS 4
#define MLIRBC_MAX_USE_LIST_INDICES 16

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
#ifndef MLIRBC_DEF
#define MLIRBC_DEF static
#endif

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

/// Called when adding properties to operation state (version 5+).
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStateAddProperties(
    void *context, MlirBytecodeOperationStateHandle,
    MlirBytecodeHandle propertiesIndex);

/// Called when finalizing population all operation state excluding
/// regions/blocks.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeOperationStatePop(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeOperationHandle *opHandle);

#ifndef MLIRBC_NO_LAZY_LOADING
/// Check if an operation was lazily loaded (regions deferred for later
/// parsing).
MLIRBC_DEF bool
mlirBytecodeOperationWasLazyLoaded(void *context,
                                   MlirBytecodeOperationHandle opHandle);

/// Store deferred region IR data for later materialization.
MLIRBC_DEF void
mlirBytecodeStoreDeferredRegionData(void *context,
                                    MlirBytecodeOperationHandle opHandle,
                                    const uint8_t *data, uint64_t length);

/// Get the number of regions in an operation.
MLIRBC_DEF uint64_t
mlirBytecodeGetOperationNumRegions(MlirBytecodeOperationHandle opHandle);
#endif // MLIRBC_NO_LAZY_LOADING

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

/// Called to add use-list order for a block argument.
/// - valueIndex: index of the value within current scope
/// - indexPairEncoding: if true, indices are (src, dst) pairs
/// - indices: array of use-list order indices
/// - numIndices: number of indices
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeBlockArgAddUseListOrder(
    void *context, uint64_t valueIndex, bool indexPairEncoding,
    const uint64_t *indices, uint64_t numIndices);

/// Called to add use-list order for an operation result.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResultAddUseListOrder(
    void *context, MlirBytecodeOperationHandle op, uint64_t resultIndex,
    bool indexPairEncoding, const uint64_t *indices, uint64_t numIndices);

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

/// Invoked with dialect version information (version 1+).
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeDialectVersionCallBack(
    void *context, MlirBytecodeDialectHandle, MlirBytecodeBytesRef version);

/// Invoked per opname with registration flag (version 5+).
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeDialectOpWithRegisteredCallBack(
    void *context, MlirBytecodeOpHandle, MlirBytecodeDialectHandle,
    MlirBytecodeStringHandle, bool isRegistered);

#ifndef MLIRBC_NO_RESOURCES
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
/// Invoked per empty dialect resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceEmptyCallBack(
    void *context, MlirBytecodeStringHandle resourceKey);
/// Invoked per string resource in group.
MLIRBC_DEF MlirBytecodeStatus mlirBytecodeResourceStringCallBack(
    void *context, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeStringHandle);
#endif // MLIRBC_NO_RESOURCES

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

/// Allocate temporary memory for parsing. Returns NULL if allocation is not
/// supported or fails. Implementations can:
/// - Return NULL to indicate no dynamic allocation (parse will fail for large
///   use-lists)
/// - Use malloc() or custom allocator
/// If this always returns NULL, the compiler can optimize away the call.
MLIRBC_DEF void *mlirBytecodeAllocateTemp(void *context, size_t bytes);

/// Free temporary memory allocated by mlirBytecodeAllocateTemp.
/// Only called if mlirBytecodeAllocateTemp returned non-NULL.
MLIRBC_DEF void mlirBytecodeFreeTemp(void *context, void *ptr);

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
                              uint64_t *result);

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
#ifndef MLIRBC_ATTRIBUTE_NOINLINE
#if __has_attribute(noinline)
#define MLIRBC_ATTRIBUTE_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define MLIRBC_ATTRIBUTE_NOINLINE __declspec(noinline)
#else
#define MLIRBC_ATTRIBUTE_NOINLINE
#endif
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

  // This section contains the versions of each dialect (version 1+).
  mbci_kDialectVersions = 7,

  // This section contains the properties for operations (version 5+).
  mbci_kProperties = 8,

  // The total number of section types.
  mbci_kNumSections = 9,
};

// Bytecode version constants.
enum {
  // The minimum supported version of the bytecode.
  mbci_kMinSupportedVersion = 0,

  // Dialects versioning was added in version 1.
  mbci_kDialectVersioning = 1,

  // Support for lazy-loading of isolated region was added in version 2.
  mbci_kLazyLoading = 2,

  // Use-list ordering started to be encoded in version 3.
  mbci_kUseListOrdering = 3,

  // Avoid recording unknown locations on block arguments (compression) started
  // in version 4.
  mbci_kElideUnknownBlockArgLocation = 4,

  // Support for encoding properties natively in bytecode instead of merged
  // with the discardable attributes.
  mbci_kNativePropertiesEncoding = 5,

  // ODS emits operand/result segment_size as native properties instead of
  // an attribute.
  mbci_kNativePropertiesODSSegmentSize = 6,

  // The current bytecode version.
  mbci_kVersion = 6,
};

#ifdef MLIRBC_VERBOSE_ERROR
static const char *mbci_sectionIDToString(uint8_t id) {
  const char *arr[] = {
      "String (0)",          //
      "Dialect (1)",         //
      "AttrType (2)",        //
      "AttrTypeOffset (3)",  //
      "IR (4)",              //
      "Resource (5)",        //
      "ResourceOffset (6)",  //
      "DialectVersions (7)", //
      "Properties (8)"       //
  };
  assert(id < sizeof(arr) / sizeof(arr[0]));
  return arr[id];
}
#endif

static bool mbci_isSectionOptional(int id, uint64_t version) {
  switch (id) {
  case mbci_kResource:
  case mbci_kResourceOffset:
  case mbci_kDialectVersions:
    return true;
  case mbci_kProperties:
    // Properties are required starting from version 5.
    return version < mbci_kNativePropertiesEncoding;
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
    return mlirBytecodeEmitError(
        context, "attempting to parse a byte at the end of the bytecode");
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
    return mlirBytecodeEmitError(
        context, "attempting to parse a byte at the end of the bytecode");
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

MlirBytecodeStatus
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

  // Fast path: Single byte encoding.
  if (MLIRBC_LIKELY(*result & 1)) {
    *result >>= 1;
    return mlirBytecodeSuccess();
  }

  // Fast path: Two byte encoding.
  // Encoding: head has pattern 0bxxxxxx10, meaning 1 additional byte.
  if (MLIRBC_LIKELY(*result & 0x02)) {
    uint8_t byte1;
    if (MLIRBC_UNLIKELY(
            mlirBytecodeFailed(mlirBytecodeParseByte(context, stream, &byte1))))
      return mlirBytecodeFailure();
    // Decode: (head >> 2) gives low 6 bits, byte1 << 6 gives high bits.
    *result = (head >> 2) | ((uint64_t)byte1 << 6);
    return mlirBytecodeSuccess();
  }

  // Slow path: 8-byte or multi-byte encoding.
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
                                                 uint64_t *result) {
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(context, stream, result);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeFailure();
  *result = (*result >> 1) ^ (~(*result & 1) + 1);
  return mlirBytecodeSuccess();
}

// Align the current reader position to the specified alignment.
// Per BytecodeFormat.md: "The input bytecode buffer must satisfy the same
// alignment requirements as those of every section."
static MlirBytecodeStatus
mbci_alignTo(void *context, MlirBytecodeStream *stream, uint32_t alignment,
             int sectionID) {
  (void)sectionID; // Used by caller for error reporting
  bool isPowerOf2 = (alignment != 0) && ((alignment & (alignment - 1)) == 0);
  if (MLIRBC_UNLIKELY(!isPowerOf2))
    return mlirBytecodeEmitError(context,
                                 "expected alignment to be a power-of-two");

  // An arbitrary value used to fill alignment padding.
  const uint8_t kAlignmentByte = 0xCB;

  // Skip the padding bytes. The bytecode writer adds padding relative to the
  // file start to align sections. We need to consume those padding bytes.
  while ((stream->pos - stream->start) & (alignment - 1)) {
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

  // Check that the absolute memory address is properly aligned.
  // This is required for mmap compatibility - the buffer must be loaded at
  // an aligned address so that aligned data can be accessed directly.
  if (MLIRBC_UNLIKELY((uintptr_t)(stream->pos) % alignment != 0)) {
    return mlirBytecodeEmitError(
        context,
        "expected section alignment %d but bytecode buffer not aligned",
        alignment);
  }
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

  mlirBytecodeEmitDebug("parsing section %d of size %" PRIu64, sectionID,
                        length);
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
    ret = mbci_alignTo(context, stream, alignment, sectionID);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
      // Emit secondary error about which section failed
      mlirBytecodeEmitError(context, "failed to align section ID: %d", sectionID);
      return ret;
    }
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
    MlirBytecodeStringHandle strHandle;
    strHandle.id = i - 1;
    ret = mlirBytecodeAssociateStringRange(context, strHandle, str);
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

  MlirBytecodeDialectHandle dialect;
  for (dialect.id = 0; dialect.id < numDialects; ++dialect.id) {
    // Before version kDialectVersioning, there wasn't any versioning available
    // for dialects, and the entry represents the string itself.
    if (parserState->version < mbci_kDialectVersioning) {
      MlirBytecodeStringHandle dialectStr;
      ret = mlirBytecodeParseVarInt(context, &stream, &dialectStr.id);
      if (mlirBytecodeFailed(ret))
        return mlirBytecodeEmitError(
            context, "unable to parse dialect %" PRIu64, dialect.id);
      if (mlirBytecodeFailed(
              mlirBytecodeDialectCallBack(context, dialect, dialectStr)))
        return mlirBytecodeFailure();
      continue;
    }

    // Parse ID representing dialect and version flag.
    uint64_t dialectNameIdxWithFlag;
    ret = mlirBytecodeParseVarInt(context, &stream, &dialectNameIdxWithFlag);
    if (mlirBytecodeFailed(ret))
      return mlirBytecodeEmitError(context, "unable to parse dialect %" PRIu64,
                                   dialect.id);

    MlirBytecodeStringHandle dialectStr;
    dialectStr.id = dialectNameIdxWithFlag >> 1;
    bool versionAvailable = dialectNameIdxWithFlag & 1;

    if (mlirBytecodeFailed(
            mlirBytecodeDialectCallBack(context, dialect, dialectStr)))
      return mlirBytecodeFailure();

    // If a version is available, parse and skip the embedded version section.
    if (versionAvailable) {
      // Parse the section header (id + length).
      uint8_t sectionId;
      ret = mlirBytecodeParseByte(context, &stream, &sectionId);
      if (mlirBytecodeFailed(ret))
        return mlirBytecodeEmitError(
            context, "unable to parse dialect version section id");
      if (sectionId != mbci_kDialectVersions)
        return mlirBytecodeEmitError(
            context, "expected dialect version section, got %d", sectionId);

      uint64_t versionLength;
      ret = mlirBytecodeParseVarInt(context, &stream, &versionLength);
      if (mlirBytecodeFailed(ret))
        return mlirBytecodeEmitError(context,
                                     "unable to parse dialect version length");

      // Extract the version bytes as a reference.
      const uint8_t *versionData;
      ret =
          mlirBytecodeParseBytes(context, &stream, versionLength, &versionData);
      if (mlirBytecodeFailed(ret))
        return mlirBytecodeEmitError(context,
                                     "unable to parse dialect version data");

      // Create version reference using explicit assignment for C89
      // compatibility.
      MlirBytecodeBytesRef versionRef;
      versionRef.data = versionData;
      versionRef.length = versionLength;

      // Call the version callback (can be a no-op in Printer example).
      ret = mlirBytecodeDialectVersionCallBack(context, dialect, versionRef);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
    }
  }

  // For version 4+, the total number of ops is pre-parsed.
  MlirBytecodeSize totalOps = 0;
  if (parserState->version >= mbci_kElideUnknownBlockArgLocation) {
    uint64_t numOps;
    ret = mlirBytecodeParseVarInt(context, &stream, &numOps);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context, "unable to parse total op count");
    totalOps = numOps;
    ret = mlirBytecodeDialectOpNames(context, totalOps);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  } else {
    // Pre-v4: Need to re-parse to count ops first.
    const uint8_t *resetPos = stream.pos;
    while (!mlirBytecodeStreamEmpty(&stream)) {
      uint64_t dialectId, numOpNames;
      mlirBytecodeParseVarInt(context, &stream, &dialectId);
      mlirBytecodeParseVarInt(context, &stream, &numOpNames);
      ret = mbci_skipVarInts(context, &stream, numOpNames);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      totalOps += numOpNames;
    }
    ret = mlirBytecodeDialectOpNames(context, totalOps);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
    stream.pos = resetPos;
  }

  // Parse operation names grouped by dialect.
  MlirBytecodeOpHandle op;
  op.id = 0;
  while (!mlirBytecodeStreamEmpty(&stream)) {
    MlirBytecodeDialectHandle opDialect;
    uint64_t numOpNames;
    mlirBytecodeParseVarInt(context, &stream, &opDialect.id);
    mlirBytecodeParseVarInt(context, &stream, &numOpNames);

    mlirBytecodeEmitDebug("parsing for dialect %d %d ops", opDialect,
                          numOpNames);
    for (uint64_t j = 0; j < numOpNames; ++j) {
      // For version 5+, op names include registration flag.
      if (parserState->version >= mbci_kNativePropertiesEncoding) {
        uint64_t opNameIdxWithFlag;
        ret = mlirBytecodeParseVarInt(context, &stream, &opNameIdxWithFlag);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context, "unable to parse op name");

        MlirBytecodeStringHandle opName;
        opName.id = opNameIdxWithFlag >> 1;
        bool isRegistered = opNameIdxWithFlag & 1;

        ret = mlirBytecodeDialectOpWithRegisteredCallBack(
            context, op, opDialect, opName, isRegistered);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
      } else {
        MlirBytecodeStringHandle opName;
        ret = mlirBytecodeParseVarInt(context, &stream, &opName.id);
        assert(mlirBytecodeSucceeded(ret) &&
               "impossible; would have failed above");
        ret = mlirBytecodeDialectOpCallBack(context, op, opDialect, opName);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
      }
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
    MlirBytecodeDialectHandle dialect;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &dialect.id);
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

      MlirBytecodeBytesRef attr;
      attr.data = atPP.pos;
      attr.length = length;

      // Verify that the offset is actually valid.
      if (MLIRBC_UNLIKELY(atPP.pos + length > atPP.end)) {
        return mlirBytecodeEmitError(
            context,
            "Attribute or Type entry offset points past the end of section");
      }

      // Parse & associate dialect.attr[j] with `attr`
      MlirBytecodeAttrHandle handle;
      handle.id = i++;
      ret = mlirBytecodeAssociateAttributeRange(context, handle, dialect, attr,
                                                hasCustomEncoding);
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
    MlirBytecodeDialectHandle dialect;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &dialect.id);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "invalid dialect handle while parsing types");

    uint64_t numElements;
    ret = mlirBytecodeParseVarInt(context, &offsetStream, &numElements);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(
          context, "invalid number of elements in type offset group");

    for (uint64_t j = 0; j < numElements; ++j) {
      uint64_t length;
      bool hasCustomEncoding;
      ret = mlirBytecodeParseVarIntWithFlag(context, &offsetStream, &length,
                                            &hasCustomEncoding);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid type offset");

      MlirBytecodeBytesRef type;
      type.data = atPP.pos;
      type.length = length;
      // Verify that the offset is actually valid.
      if (MLIRBC_UNLIKELY(atPP.pos + length > atPP.end)) {
        return mlirBytecodeEmitError(
            context,
            "Attribute or Type entry offset points past the end of section");
      }

      MlirBytecodeTypeHandle handle;
      handle.id = i++;
      ret = mlirBytecodeAssociateTypeRange(context, handle, dialect, type,
                                           hasCustomEncoding);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret))) {
        mlirBytecodeEmitDebug("type callback failed");
        return ret;
      }
      if (MLIRBC_UNLIKELY(mlirBytecodeInterrupted(ret)))
        return ret;
      // Unhandled attributes are not considered error.

      ret = mbci_skipBytes(context, &atPP, length);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid type offset");
    }
  }

  return mlirBytecodeSuccess();
}

#ifndef MLIRBC_NO_RESOURCES
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
  const MlirBytecodeBytesRef resSection =
      mbci_getSection(parserState, mbci_kResource);
  // Offset section is empty if either it has no elements or it has a single
  // element which is the 0 number of external resource elements.
  bool emptyOffsetSection = offsetSection.length <= 1;
  assert(!emptyOffsetSection ||
         (offsetSection.length == 0 ||
          (offsetSection.length == 1 && offsetSection.data[0] == 1)));

  // If offset section is empty but resource section is not, that's an error.
  // If offset section is present but resource section is empty, that's OK
  // (resources were elided with -elide-resource-data-from-bytecode).
  if (emptyOffsetSection && resSection.length != 0) {
    return mlirBytecodeEmitError(context,
                                 "unexpected resource section when "
                                 "resource offset section is not present");
  }
  if (offsetSection.length == 0)
    return mlirBytecodeSuccess();

  MlirBytecodeStream offsetStream =
      mbci_populateStreamForSection(offsetSection);
  MlirBytecodeStream resPP = mbci_populateStreamForSection(resSection);

  uint64_t numExternalResourceGroups;
  MlirBytecodeStatus ret = mlirBytecodeParseVarInt(context, &offsetStream,
                                                   &numExternalResourceGroups);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(
        context, "invalid/missing number of external resource groups");
  mlirBytecodeEmitDebug("number of external section %" PRIu64,
                        numExternalResourceGroups);

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
      MlirBytecodeStringHandle resourceKey;
      ret = mlirBytecodeParseVarInt(context, &offsetStream, &resourceKey.id);
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

      // If the resource data is empty and we allow it, don't error out when
      // parsing below, just skip it.
      if (i >= numExternalResourceGroups && size == 0) {
        ret = mlirBytecodeResourceEmptyCallBack(context, resourceKey);
        if (mlirBytecodeFailed(ret))
          return ret;
        continue;
      }

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
        ret = mbci_alignTo(context, &resPP, alignment, mbci_kResource);
        const uint8_t *start = resPP.pos;
        if (mlirBytecodeFailed(ret) ||
            mlirBytecodeFailed(mbci_skipBytes(context, &resPP, size)))
          return mlirBytecodeFailure();
        MlirBytecodeBytesRef blob;
        blob.data = start;
        blob.length = size;
        ret = mlirBytecodeResourceBlobCallBack(context, resourceKey, alignment,
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
        ret = mlirBytecodeResourceBoolCallBack(context, resourceKey, value);
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
        ret = mlirBytecodeResourceStringCallBack(context, resourceKey, value);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
        break;
      }
      }
    }
  }
  return mlirBytecodeSuccess();
}
#endif // MLIRBC_NO_RESOURCES

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
#ifndef MLIRBC_NO_OVERFLOW_IR_STACK
    // For the last fixed stack entry: pointer to overflow entries when depth
    // exceeds MLIRBC_IR_STACK_MAX_DEPTH. Size is 2*MLIRBC_IR_STACK_MAX_DEPTH.
    // Sentinel: numRegionsRemaining == 0x7FFF indicates overflow active.
    struct mbci_MlirIRSectionStackEntry *overflowEntries;
#endif
  };

  // Number of regions to still complete parsing.
  // Number of regions is assumed to be relatively small (< 32767).
  uint16_t numRegionsRemaining : 15;

  // Whether this region is isolated from above (v2+ uses separate IR section).
  uint16_t isIsolatedFromAbove : 1;

  // Number of blocks to still complete parsing.
  // Number of regions is assumed to be relatively small (<= 65536).
  uint16_t numBlocksRemaining;

  // Number of ops remaining.
  uint32_t numOpsRemaining;

  // For v2+ isolated regions: stream pointing to the isolated region's IR
  // section. When null, uses the parent stream.
  MlirBytecodeStream *isolatedStream;

#ifndef MLIRBC_NO_USE_LIST_ORDERING
  // Pending use-list orders for operation results (stored during parsing,
  // applied after op creation).
  uint8_t numPendingUseListOrders;
  struct {
    uint8_t resultIdx;
    uint8_t indexPairEncoding;
    uint8_t numIndices;
    uint16_t indices[MLIRBC_MAX_USE_LIST_INDICES];
  } pendingUseListOrders[MLIRBC_MAX_PENDING_USE_LISTS];
#endif
};
typedef struct mbci_MlirIRSectionStackEntry mbci_MlirIRSectionStackEntry;

// IR section parsing stack. Instruction without region does not result in
// pushing anything on stack, for operations with regions:
struct mbci_MlirIRSectionStack {
  int top;

  mbci_MlirIRSectionStackEntry data[MLIRBC_IR_STACK_MAX_DEPTH + 1];

  // Inline storage for isolated region streams (v2+ lazy loading).
  // Each stack level may have its own isolated stream.
  MlirBytecodeStream isolatedStreamStorage[MLIRBC_IR_STACK_MAX_DEPTH + 1];
};
typedef struct mbci_MlirIRSectionStack mbci_MlirIRSectionStack;

#ifndef MLIRBC_NO_OVERFLOW_IR_STACK
// Sentinel value to indicate overflow array is active.
#define MLIRBC_STACK_OVERFLOW_SENTINEL 0x7FFF

// Get entry at specified index, handling overflow.
static mbci_MlirIRSectionStackEntry *
mbci_getStackEntry(mbci_MlirIRSectionStack *stack, int idx) {
  if (idx < MLIRBC_IR_STACK_MAX_DEPTH) {
    return &stack->data[idx];
  }
  // Overflow: last fixed entry holds pointer to overflow array.
  mbci_MlirIRSectionStackEntry *last = &stack->data[MLIRBC_IR_STACK_MAX_DEPTH];
  if (last->numRegionsRemaining == MLIRBC_STACK_OVERFLOW_SENTINEL &&
      last->overflowEntries != NULL) {
    return &last->overflowEntries[idx - MLIRBC_IR_STACK_MAX_DEPTH];
  }
  return NULL; // Should not happen if stack management is correct.
}

// Get isolated stream storage at specified index, handling overflow.
// Uses start==NULL as sentinel in last entry, with pos holding overflow pointer.
static MlirBytecodeStream *
mbci_getIsolatedStream(mbci_MlirIRSectionStack *stack, int idx) {
  if (idx < MLIRBC_IR_STACK_MAX_DEPTH) {
    return &stack->isolatedStreamStorage[idx];
  }
  // Overflow: last entry uses start==NULL sentinel, pos holds overflow array.
  MlirBytecodeStream *last = &stack->isolatedStreamStorage[MLIRBC_IR_STACK_MAX_DEPTH];
  if (last->start == NULL && last->pos != NULL) {
    MlirBytecodeStream *overflow = (MlirBytecodeStream *)(void *)last->pos;
    return &overflow[idx - MLIRBC_IR_STACK_MAX_DEPTH];
  }
  return NULL; // Should not happen if stack management is correct.
}

static mbci_MlirIRSectionStackEntry *
mbci_getStackTop(mbci_MlirIRSectionStack *stack) {
  return mbci_getStackEntry(stack, stack->top);
}
static mbci_MlirIRSectionStackEntry *
mbci_getStackWIP(mbci_MlirIRSectionStack *stack) {
  return mbci_getStackEntry(stack, stack->top + 1);
}

// Initialize stack entry with required state to resume. `op` is initialized
// already and so not passed in.
static MlirBytecodeStatus
mbci_mlirIRSectionStackPush(void *context, mbci_MlirIRSectionStack *stack) {
  if (MLIRBC_UNLIKELY(stack->top + 1 >= MLIRBC_IR_STACK_MAX_DEPTH)) {
    // Check if we can use overflow.
    mbci_MlirIRSectionStackEntry *last =
        &stack->data[MLIRBC_IR_STACK_MAX_DEPTH];
    if (stack->top + 1 == MLIRBC_IR_STACK_MAX_DEPTH) {
      // First overflow: allocate if possible.
      mbci_MlirIRSectionStackEntry *overflow =
          (mbci_MlirIRSectionStackEntry *)mlirBytecodeAllocateTemp(
              context, 2 * MLIRBC_IR_STACK_MAX_DEPTH *
                           sizeof(mbci_MlirIRSectionStackEntry));
      if (overflow == NULL) {
        mlirBytecodeEmitError(context, "IR stack max depth exceeded");
        return mlirBytecodeIterationInterrupt();
      }
      last->overflowEntries = overflow;
      last->numRegionsRemaining = MLIRBC_STACK_OVERFLOW_SENTINEL;
      
      // Also allocate overflow for isolated stream storage.
      MlirBytecodeStream *streamOverflow =
          (MlirBytecodeStream *)mlirBytecodeAllocateTemp(
              context, 2 * MLIRBC_IR_STACK_MAX_DEPTH *
                           sizeof(MlirBytecodeStream));
      if (streamOverflow == NULL) {
        // Clean up previously allocated overflow.
        mlirBytecodeFreeTemp(context, overflow);
        last->overflowEntries = NULL;
        last->numRegionsRemaining = 0;
        mlirBytecodeEmitError(context, "IR stack overflow allocation failed");
        return mlirBytecodeIterationInterrupt();
      }
      // Use NULL sentinel: start==NULL, pos holds pointer.
      MlirBytecodeStream *streamLast = &stack->isolatedStreamStorage[MLIRBC_IR_STACK_MAX_DEPTH];
      streamLast->start = NULL;
      streamLast->pos = (const uint8_t *)(void *)streamOverflow;
      streamLast->end = NULL;
    } else if (last->numRegionsRemaining != MLIRBC_STACK_OVERFLOW_SENTINEL ||
               last->overflowEntries == NULL) {
      mlirBytecodeEmitError(context,
                            "IR stack max depth exceeded (no allocator)");
      return mlirBytecodeIterationInterrupt();
    }
    // Check overflow bounds.
    int overflowIdx = stack->top + 1 - MLIRBC_IR_STACK_MAX_DEPTH;
    if (overflowIdx >= 2 * MLIRBC_IR_STACK_MAX_DEPTH) {
      mlirBytecodeEmitError(context, "IR stack overflow exceeded");
      return mlirBytecodeIterationInterrupt();
    }
  }
  ++stack->top;
  mbci_MlirIRSectionStackEntry *it = mbci_getStackTop(stack);
  it->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
  it->numOpsRemaining = mbci_MlirIRSectionStackUnset;
  it->isolatedStream = NULL;
  // Note: isIsolatedFromAbove is already set on this entry from
  // mbci_parseOperation.

  return mlirBytecodeSuccess();
}
#else
// Simple non-overflow stack access.
static mbci_MlirIRSectionStackEntry *
mbci_getStackTop(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top];
}
static mbci_MlirIRSectionStackEntry *
mbci_getStackWIP(mbci_MlirIRSectionStack *stack) {
  return &stack->data[stack->top + 1];
}

// Isolated stream access (no overflow).
static MlirBytecodeStream *
mbci_getIsolatedStream(mbci_MlirIRSectionStack *stack, int idx) {
  return &stack->isolatedStreamStorage[idx];
}

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
  it->isolatedStream = NULL;
  return mlirBytecodeSuccess();
}
#endif

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
    MlirBytecodeLocHandle loc;

    // Version 4+ uses (type << 1) | hasLocation encoding for compression.
    if (parserState->version >= mbci_kElideUnknownBlockArgLocation) {
      uint64_t typeIdxWithFlag;
      ret = mlirBytecodeParseVarInt(context, stream, &typeIdxWithFlag);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;

      type.id = typeIdxWithFlag >> 1;
      bool hasLoc = typeIdxWithFlag & 1;

      if (hasLoc) {
        ret = mlirBytecodeParseHandle(context, stream, &loc);
        if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
          return ret;
      } else {
        // Unknown location - use sentinel value (0xFFFFFFFFFFFFFFFF).
        loc.id = UINT64_MAX;
      }
    } else {
      // Pre-v4: type and location are separate varints.
      ret = mlirBytecodeParseHandle(context, stream, &type);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      ret = mlirBytecodeParseHandle(context, stream, &loc);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
    }

    ret = mlirBytecodeOperationBlockAddArgument(context, top->op, type, loc);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }

  // Parse use-list orders for block arguments (v3+).
  if (hasArgs && parserState->version >= mbci_kUseListOrdering) {
    uint8_t hasUseListOrders = 0;
    ret = mlirBytecodeParseByte(context, stream, &hasUseListOrders);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return ret;

    if (hasUseListOrders) {
      uint64_t numValuesToRead = 1;
      if (numArgs > 1) {
        ret = mlirBytecodeParseVarInt(context, stream, &numValuesToRead);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid block arg use-list count");
      }

      for (uint64_t v = 0; v < numValuesToRead; ++v) {
        uint64_t argIdx = 0;
        if (numArgs > 1) {
          ret = mlirBytecodeParseVarInt(context, stream, &argIdx);
          if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
            return mlirBytecodeEmitError(context,
                                         "invalid block arg use-list index");
        }

        uint64_t numIndicesWithFlag;
        ret = mlirBytecodeParseVarInt(context, stream, &numIndicesWithFlag);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid block arg use-list encoding");

        bool indexPairEncoding = numIndicesWithFlag & 1;
        uint64_t numIndices = numIndicesWithFlag >> 1;

        // Parse all indices into a buffer.
        uint64_t indicesBuf[64]; // Stack buffer for small cases
        uint64_t *indices = indicesBuf;
        if (numIndices > 64) {
          // Try to allocate via user-provided callback. If it returns NULL,
          // we cannot handle this use-list (too large for stack buffer).
          indices = (uint64_t *)mlirBytecodeAllocateTemp(
              context, numIndices * sizeof(uint64_t));
          if (!indices)
            return mlirBytecodeEmitError(context,
                                         "use-list too large for stack buffer "
                                         "and allocation not supported");
        }

        for (uint64_t i = 0; i < numIndices; ++i) {
          ret = mlirBytecodeParseVarInt(context, stream, &indices[i]);
          if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
            if (indices != indicesBuf)
              mlirBytecodeFreeTemp(context, indices);
            return mlirBytecodeEmitError(
                context, "invalid block arg use-list index value");
          }
        }

        // Call the callback to store the use-list order.
        ret = mlirBytecodeBlockArgAddUseListOrder(
            context, argIdx, indexPairEncoding, indices, numIndices);
        if (indices != indicesBuf)
          mlirBytecodeFreeTemp(context, indices);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return ret;
      }
    }
  }
  return ret;
}

static MlirBytecodeStatus mbci_parseRegion(void *context,
                                           MlirBytecodeParserState *parserState,
                                           MlirBytecodeStream *stream,
                                           mbci_MlirIRSectionStack *stack,
                                           void *state) {
  mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(stack);

#ifndef MLIRBC_NO_LAZY_LOADING
  // For v2+ with isolated regions, parse the inline IR section first.
  // The isIsolatedFromAbove flag is set on the current stack top (the operation
  // with regions).
  // The bytecode writer wraps all regions of an isolated operation in a single
  // IR section. We only read the section header once (for the first region)
  // while subsequent regions use the already-parsed isolated stream.
  if (parserState->version >= mbci_kLazyLoading && top->isIsolatedFromAbove) {
    // Check if we already have an isolated stream from parsing a previous
    // region of this same operation.
    if (top->isolatedStream == NULL) {
      // First region: parse the inline section header: section ID + length + data
      uint8_t sectionId;
      MlirBytecodeStatus ret = mlirBytecodeParseByte(context, stream, &sectionId);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context,
                                     "unable to parse inline IR section id");
      if (sectionId != mbci_kIR)
        return mlirBytecodeEmitError(
            context, "expected IR section for isolated region, got %d",
            sectionId);

      uint64_t sectionLength;
      ret = mlirBytecodeParseVarInt(context, stream, &sectionLength);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context,
                                     "unable to parse inline IR section length");

      // Create a new stream for the isolated region's IR content.
      const uint8_t *sectionData;
      ret = mlirBytecodeParseBytes(context, stream, sectionLength, &sectionData);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context,
                                     "unable to parse inline IR section data");

      // Store the isolated region's stream in the stack entry.
      // Use stack-local storage for thread-safety and reentrancy.
      MlirBytecodeStream *isolatedStorage = mbci_getIsolatedStream(stack, stack->top);
      isolatedStorage->pos = sectionData;
      isolatedStorage->end = sectionData + sectionLength;
      isolatedStorage->start = sectionData;
      top->isolatedStream = isolatedStorage;
    }

    // Now parse region content from the isolated stream.
    stream = top->isolatedStream;
  }
#endif // MLIRBC_NO_LAZY_LOADING

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
    kHasUseListOrders = 0x20, // v3+
    kHasProperties = 0x40,    // v5+
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
  // Initialize isIsolatedFromAbove and pending use-lists to clear stale values.
  cur->isIsolatedFromAbove = 0;
#ifndef MLIRBC_NO_USE_LIST_ORDERING
  cur->numPendingUseListOrders = 0;
#endif
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

  // Parse properties for v5+.
  if (parserState->version >= mbci_kNativePropertiesEncoding &&
      (encodingMask & kHasProperties)) {
    uint64_t propertiesIdx;
    ret = mlirBytecodeParseVarInt(context, stream, &propertiesIdx);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
      return mlirBytecodeEmitError(context, "invalid properties index");
    MlirBytecodeHandle propsHandle;
    propsHandle.id = propertiesIdx;
    ret = mlirBytecodeOperationStateAddProperties(
        context, cur->opState, propsHandle);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }

  // Parsing all the variadic sizes. numResults is used for use-list parsing.
  uint64_t numResults = 0;
  if (encodingMask & kHasResults) {
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

  // Parse use-list orders for v3+ if present.
#ifndef MLIRBC_NO_USE_LIST_ORDERING
  // Store data in pending buffer, then call callback after op creation.
  if (parserState->version >= mbci_kUseListOrdering &&
      (encodingMask & kHasUseListOrders)) {
    uint64_t numValuesToRead = 1;
    if (numResults > 1) {
      ret = mlirBytecodeParseVarInt(context, stream, &numValuesToRead);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid use-list count");
    }

    for (uint64_t v = 0; v < numValuesToRead; ++v) {
      uint64_t resultIdx = 0;
      if (numResults > 1) {
        ret = mlirBytecodeParseVarInt(context, stream, &resultIdx);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "invalid use-list result index");
      }

      uint64_t numIndicesWithFlag;
      ret = mlirBytecodeParseVarInt(context, stream, &numIndicesWithFlag);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "invalid use-list encoding");

      bool indexPairEncoding = numIndicesWithFlag & 1;
      uint64_t numIndices = numIndicesWithFlag >> 1;

      // Store in pending buffer if there's space.
      if (cur->numPendingUseListOrders < MLIRBC_MAX_PENDING_USE_LISTS &&
          numIndices <= MLIRBC_MAX_USE_LIST_INDICES) {
        uint8_t idx = cur->numPendingUseListOrders++;
        cur->pendingUseListOrders[idx].resultIdx = (uint8_t)resultIdx;
        cur->pendingUseListOrders[idx].indexPairEncoding =
            indexPairEncoding ? 1 : 0;
        cur->pendingUseListOrders[idx].numIndices = (uint8_t)numIndices;
        for (uint64_t i = 0; i < numIndices; ++i) {
          uint64_t index;
          ret = mlirBytecodeParseVarInt(context, stream, &index);
          if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
            return mlirBytecodeEmitError(context, "invalid use-list index");
          cur->pendingUseListOrders[idx].indices[i] = (uint16_t)index;
        }
      } else {
        // Skip indices we can't store (too many results or too many indices).
        for (uint64_t i = 0; i < numIndices; ++i) {
          uint64_t index;
          ret = mlirBytecodeParseVarInt(context, stream, &index);
          if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
            return mlirBytecodeEmitError(context, "invalid use-list index");
        }
      }
    }
  }
#else
  // Skip use-list orders entirely when disabled.
  if (parserState->version >= mbci_kUseListOrdering &&
      (encodingMask & kHasUseListOrders)) {
    uint64_t numValuesToRead = 1;
    if (numResults > 1) {
      ret = mlirBytecodeParseVarInt(context, stream, &numValuesToRead);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "skipping use-list count");
    }
    for (uint64_t v = 0; v < numValuesToRead; ++v) {
      if (numResults > 1) {
        uint64_t resultIdx;
        ret = mlirBytecodeParseVarInt(context, stream, &resultIdx);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context,
                                       "skipping use-list result index");
      }
      uint64_t numIndicesWithFlag;
      ret = mlirBytecodeParseVarInt(context, stream, &numIndicesWithFlag);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(context, "skipping use-list encoding");
      uint64_t numIndices = numIndicesWithFlag >> 1;
      for (uint64_t i = 0; i < numIndices; ++i) {
        uint64_t index;
        ret = mlirBytecodeParseVarInt(context, stream, &index);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return mlirBytecodeEmitError(context, "skipping use-list index");
      }
    }
  }
#endif

  if (encodingMask & kHasInlineRegions) {
    uint64_t numRegions = 0;
    bool isIsolatedFromAbove = false;
    ret = mlirBytecodeParseVarIntWithFlag(context, stream, &numRegions,
                                          &isIsolatedFromAbove);
    if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret))) {
      return mlirBytecodeEmitError(context, "invalid number of regions");
    }
    // Validate numRegions fits in 15-bit field (max 32766, reserve 0x7FFF for
    // overflow sentinel).
    if (MLIRBC_UNLIKELY(numRegions >= 0x7FFF)) {
      return mlirBytecodeEmitError(
          context, "number of regions (%llu) exceeds maximum (32766)",
          (unsigned long long)numRegions);
    }
    cur->numRegionsRemaining = numRegions;
    cur->isIsolatedFromAbove = isIsolatedFromAbove;
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

  // Apply pending use-list orders now that the operation exists.
#ifndef MLIRBC_NO_USE_LIST_ORDERING
  for (uint8_t i = 0; i < cur->numPendingUseListOrders; ++i) {
    uint64_t indices64[16];
    for (uint8_t j = 0; j < cur->pendingUseListOrders[i].numIndices; ++j)
      indices64[j] = cur->pendingUseListOrders[i].indices[j];
    ret = mlirBytecodeResultAddUseListOrder(
        context, opHandle, cur->pendingUseListOrders[i].resultIdx,
        cur->pendingUseListOrders[i].indexPairEncoding != 0, indices64,
        cur->pendingUseListOrders[i].numIndices);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }
#endif

  if (encodingMask & kHasInlineRegions) {
#ifndef MLIRBC_NO_LAZY_LOADING
    // If the operation was lazily loaded, skip region processing.
    // The regions will be parsed later when materialized.
    if (mlirBytecodeOperationWasLazyLoaded(context, opHandle)) {
      // Skip the inline IR section data for isolated regions (v2+).
      if (parserState->version >= mbci_kLazyLoading &&
          cur->isIsolatedFromAbove) {
        // Parse and skip the section header.
        uint8_t sectionId;
        ret = mlirBytecodeParseByte(context, stream, &sectionId);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return ret;
        uint64_t sectionLength;
        ret = mlirBytecodeParseVarInt(context, stream, &sectionLength);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return ret;
        // Skip the section data.
        const uint8_t *sectionData;
        ret = mlirBytecodeParseBytes(context, stream, sectionLength,
                                     &sectionData);
        if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
          return ret;
        // Store the deferred region data for later materialization.
        mlirBytecodeStoreDeferredRegionData(context, opHandle, sectionData,
                                            sectionLength);
      }
      // Decrement op count since we won't push to region stack.
      --mbci_getStackTop(stack)->numOpsRemaining;
      return ret;
    }
#endif // MLIRBC_NO_LAZY_LOADING
    return mbci_mlirIRSectionStackPush(context, stack);
  }

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
  mbci_MlirIRSectionStack stack;
  stack.top = 0;
  stack.data[0].numRegionsRemaining = 1;
  stack.data[0].numBlocksRemaining = 1;
  stack.data[0].numOpsRemaining = numOps;
  stack.data[0].isIsolatedFromAbove = 0;
  stack.data[0].isolatedStream = NULL;

  while (stack.data[0].numOpsRemaining) {
    mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(&stack);
    // Find the active stream by walking up the stack to find the closest
    // isolated stream.
    MlirBytecodeStream *activeStream = &stream;
    for (int i = stack.top; i >= 0; --i) {
      if (stack.data[i].isolatedStream) {
        activeStream = stack.data[i].isolatedStream;
        break;
      }
    }

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
      ret =
          mbci_parseRegion(context, parserState, activeStream, &stack, context);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    if (top->numOpsRemaining == (uint32_t)mbci_MlirIRSectionStackUnset) {
      mlirBytecodeEmitDebug("starting block parsing");
      ret = mbci_parseBlock(context, parserState, activeStream, &stack);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    mlirBytecodeEmitDebug("parsing operation (remaining regions: %" PRIu64
                          " blocks: %" PRIu64 " ops: %" PRIu64 ")",
                          mbci_getStackTop(&stack)->numRegionsRemaining,
                          mbci_getStackTop(&stack)->numBlocksRemaining,
                          mbci_getStackTop(&stack)->numOpsRemaining);
    ret = mbci_parseOperation(context, parserState, activeStream, &stack);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }

#ifndef MLIRBC_NO_OVERFLOW_IR_STACK
  // Clean up overflow arrays if allocated.
  mbci_MlirIRSectionStackEntry *last = &stack.data[MLIRBC_IR_STACK_MAX_DEPTH];
  if (last->numRegionsRemaining == MLIRBC_STACK_OVERFLOW_SENTINEL) {
    if (last->overflowEntries) {
      mlirBytecodeFreeTemp(context, last->overflowEntries);
    }
    // Clean up isolated stream overflow.
    MlirBytecodeStream *streamLast = &stack.isolatedStreamStorage[MLIRBC_IR_STACK_MAX_DEPTH];
    if (streamLast->start == NULL && streamLast->pos != NULL) {
      mlirBytecodeFreeTemp(context, (void *)streamLast->pos);
    }
  }
#endif

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
  mlirBytecodeEmitDebug("bytecode top");
  ;

  const uint8_t *magic;
  if (mlirBytecodeFailed(mlirBytecodeParseBytes(context, &stream, 4, &magic)))
    return mlirBytecodeEmitError(context, "unable to read 4 byte magic code"),
           ret;
  if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 0xef || magic[3] != 'R')
    return mlirBytecodeEmitError(context, "invalid file magic code"), ret;

  // Parse the bytecode version and producer.
  MlirBytecodeBytesRef producer;
  if (mlirBytecodeFailed(
          mlirBytecodeParseVarInt(context, &stream, &ret.version)))
    return mlirBytecodeEmitError(context, "invalid version"), ret;
  // Validate the bytecode version.
  if (ret.version > mbci_kVersion)
    return mlirBytecodeEmitError(context,
                                 "bytecode version %" PRIu64
                                 " is newer than the current version %d",
                                 ret.version, mbci_kVersion),
           ret;
  if (mlirBytecodeFailed(
          mlirBytecodeParseNullTerminatedString(context, &stream, &producer)))
    return mlirBytecodeEmitError(context, "invalid producer"), ret;

  while (!mlirBytecodeStreamEmpty(&stream)) {
    // Read the next section from the bytecode.
    if (mlirBytecodeFailed(
            mbci_parseSections(context, &stream, ret.sectionData)))
      return mlirBytecodeEmitError(context, "invalid section"), ret;
  }

  // Check that all of the required sections were found.
  for (int i = 0; i < mbci_kNumSections; ++i) {
    if (mbci_getSection(&ret, i).data == NULL &&
        !mbci_isSectionOptional(i, ret.version)) {
      return mlirBytecodeEmitError(context,
                                   "missing data for top-level section: %s",
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
    return mlirBytecodeFailure();

  // Process the string section first.
  MlirBytecodeStatus ret = mlirBytecodeForEachString(context, parserState);

  if (!mlirBytecodeSucceeded(ret))
    return ret;

  // Process the dialect section.
  ret = mlirBytecodeParseDialectSection(context, parserState);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
#ifndef MLIRBC_NO_RESOURCES
  // Process the resource section if present.
  ret = mlirBytecodeParseResourceSection(context, parserState);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
#endif
  // Process the attribute and type section.
  ret = mlirBytecodeParseAttributeAndTypeSection(context, parserState);
  if (!mlirBytecodeSucceeded(ret))
    return ret;

  // Finally, process the IR section.
  return mlirBytecodeParseIRSection(context, parserState, retBlock);
}

#ifndef MLIRBC_NO_LAZY_LOADING
/// Parse a deferred region from stored bytecode data.
/// This is used for lazy loading materialization.
MlirBytecodeStatus mlirBytecodeParseDeferredRegion(
    void *context, MlirBytecodeParserState *parserState,
    const uint8_t *regionData, uint64_t regionDataLength,
    MlirBytecodeOperationHandle opHandle) {
  if (regionData == NULL || regionDataLength == 0)
    return mlirBytecodeSuccess(); // Empty region data - nothing to parse

  // Create stream from region data
  MlirBytecodeBytesRef ref = {regionData, regionDataLength};
  MlirBytecodeStream stream = mlirBytecodeStreamCreate(ref);

  // Region format: numBlocks (varint), numValues (varint), then blocks...
  uint64_t numBlocks;
  MlirBytecodeStatus ret =
      mlirBytecodeParseVarInt(context, &stream, &numBlocks);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(
        context, "unable to parse number of blocks in deferred region");

  if (numBlocks == 0)
    return mlirBytecodeSuccess(); // Empty region

  uint64_t numValues;
  ret = mlirBytecodeParseVarInt(context, &stream, &numValues);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return mlirBytecodeEmitError(
        context, "unable to parse number of values in deferred region");

  // Set up the parsing state - push region first
  ret =
      mlirBytecodeOperationRegionPush(context, opHandle, numBlocks, numValues);
  if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
    return ret;

  // Get the actual number of regions from the operation
  uint64_t numRegions = mlirBytecodeGetOperationNumRegions(opHandle);

  // Initialize parsing stack
  mbci_MlirIRSectionStack stack;
  stack.top = 0;
  stack.data[0].op = opHandle;
  stack.data[0].numRegionsRemaining = numRegions;
  stack.data[0].numBlocksRemaining = numBlocks;
  stack.data[0].numOpsRemaining = mbci_MlirIRSectionStackUnset;
  stack.data[0].isIsolatedFromAbove = 1;
  stack.data[0].isolatedStream = &stream;
#ifndef MLIRBC_NO_USE_LIST_ORDERING
  stack.data[0].numPendingUseListOrders = 0;
#endif

  // Parse using the standard IR parsing loop
  while (stack.data[0].numRegionsRemaining > 0 ||
         stack.data[0].numBlocksRemaining > 0 ||
         stack.data[0].numOpsRemaining > 0) {
    mbci_MlirIRSectionStackEntry *top = mbci_getStackTop(&stack);
    MlirBytecodeStream *activeStream = &stream;
    for (int i = stack.top; i >= 0; --i) {
      if (stack.data[i].isolatedStream) {
        activeStream = stack.data[i].isolatedStream;
        break;
      }
    }

    bool next = false;
    if (top->numOpsRemaining == 0) {
      ret = mlirBytecodeOperationBlockPop(context, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      --top->numBlocksRemaining;
      top->numOpsRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numBlocksRemaining == 0) {
      ret = mlirBytecodeOperationRegionPop(context, top->op);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      --top->numRegionsRemaining;
      top->numBlocksRemaining = mbci_MlirIRSectionStackUnset;
      next = true;
    }
    if (top->numRegionsRemaining == 0) {
      mbci_mlirIRSectionStackPop(context, &stack);
      if (stack.top < 0)
        break; // Done parsing
      --mbci_getStackTop(&stack)->numOpsRemaining;
      next = true;
    }
    if (next)
      continue;

    if (top->numBlocksRemaining == (uint16_t)mbci_MlirIRSectionStackUnset) {
      // Parse region content directly without inline IR section handling
      // (the inline section was already consumed when we stored the irData)
      uint64_t numBlocksRegion;
      ret = mlirBytecodeParseVarInt(context, activeStream, &numBlocksRegion);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(
            context, "unable to parse numBlocks in deferred region");

      top->numBlocksRemaining = numBlocksRegion;
      top->numOpsRemaining = mbci_MlirIRSectionStackUnset;

      if (numBlocksRegion == 0)
        continue;

      uint64_t numValuesRegion;
      ret = mlirBytecodeParseVarInt(context, activeStream, &numValuesRegion);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return mlirBytecodeEmitError(
            context, "unable to parse numValues in deferred region");

      ret = mlirBytecodeOperationRegionPush(context, top->op, numBlocksRegion,
                                            numValuesRegion);
      if (MLIRBC_UNLIKELY(mlirBytecodeFailed(ret)))
        return ret;
      continue;
    }
    if (top->numOpsRemaining == (uint32_t)mbci_MlirIRSectionStackUnset) {
      ret = mbci_parseBlock(context, parserState, activeStream, &stack);
      if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
        return ret;
      continue;
    }
    ret = mbci_parseOperation(context, parserState, activeStream, &stack);
    if (MLIRBC_UNLIKELY(!mlirBytecodeSucceeded(ret)))
      return ret;
  }

#ifndef MLIRBC_NO_OVERFLOW_IR_STACK
  // Clean up overflow arrays if allocated.
  mbci_MlirIRSectionStackEntry *last = &stack.data[MLIRBC_IR_STACK_MAX_DEPTH];
  if (last->numRegionsRemaining == MLIRBC_STACK_OVERFLOW_SENTINEL) {
    if (last->overflowEntries) {
      mlirBytecodeFreeTemp(context, last->overflowEntries);
    }
    // Clean up isolated stream overflow.
    MlirBytecodeStream *streamLast = &stack.isolatedStreamStorage[MLIRBC_IR_STACK_MAX_DEPTH];
    if (streamLast->start == NULL && streamLast->pos != NULL) {
      mlirBytecodeFreeTemp(context, (void *)streamLast->pos);
    }
  }
#endif

  return mlirBytecodeSuccess();
}
#endif // MLIRBC_NO_LAZY_LOADING
