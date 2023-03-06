//===-- PrintParse.c - Parser instantiation that just prints --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Example parser instantiation that just prints IR.
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Parsing inc configuration.
typedef struct MlirBytecodeOperationState MlirBytecodeOperationState;
typedef struct MlirBytecodeOperationState MlirBytecodeOperation;
// Include bytecode parsing implementation.
#include "mlir/Bytecode/BytecodeTypes.h"
#include "mlir/Bytecode/Parse.c.inc"
// Dialect attribute and types parsing hooks.
#include "mlir/Bytecode/DialectBytecodeReader.c.inc"
// Dialects.
#include "BuiltinParse.c.inc"

// Example that prints as one parses.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// Define struct that captures operation state during parsing.
struct MlirBytecodeOperationState {
  MlirBytecodeOpHandle name;
  MlirBytecodeAttrHandle attrDict;
  MlirBytecodeLocHandle loc;
  bool isIsolated;
  bool hasOperands;
  bool hasRegions;
  bool hasResults;
  bool hasSuccessors;
  int numArgsRemaining;
};

typedef struct MlirMutableBytesRef MlirMutableBytesRef;
typedef MlirMutableBytesRef MlirBytecodeAttribute;
typedef MlirMutableBytesRef MlirBytecodeType;

// Just using some globals here for the example, this should be part of the
// user/instantiation state and globals avoided in general.

struct MlirMutableBytesRef {
  // Pointer to the start memory address.
  uint8_t *data;
  // Length of the fragment.
  size_t length;
};

// Effectively set up a cached dialect behavior.
#define MAX_DIALECTS 10
#define MAX_DEPTH 10

struct MlirBytecodeAttributeOrTypeRange {
  MlirBytecodeBytesRef bytes;
  MlirBytecodeDialectHandle dialectHandle;
  bool hasCustom;
};
typedef struct MlirBytecodeAttributeOrTypeRange
    MlirBytecodeAttributeOrTypeRange;

struct ParsingState {
  MlirBytecodeStringHandle *dialectStr;

  MlirBytecodeAttrParseCallBack attrCallbacks[MAX_DIALECTS];
  MlirBytecodeTypeParseCallBack typeCallbacks[MAX_DIALECTS];

  // Mapping from string handle to section.
  MlirBytecodeBytesRef *strings;

  // Mapping from op handle to op.
  MlirBytecodeOpRef *ops;

  // Attribute handle to string representation.
  MlirBytecodeAttribute *attributes;
  // Mapping from attribute handle to range.
  MlirBytecodeAttributeOrTypeRange *attributeRange;
  int numAttributes;

  // Type handle to string representation.
  MlirMutableBytesRef *types;
  // Mapping from type handle to range.
  MlirBytecodeAttributeOrTypeRange *typeRange;
  int numTypes;

  // Last used SSA at current depth.
  int ssaIdStack[MAX_DEPTH];

  // Number of values in regions nested.
  int regionNumValues[MAX_DEPTH];

  // Current depth.
  int depth;

  // Indent size for formatting.
  int indentSize;
};
typedef struct ParsingState ParsingState;

#ifdef MLIRBC_VERBOSE_ERROR
MlirBytecodeStatus mlirBytecodeEmitErrorImpl(void *context, const char *fmt,
                                             ...) {
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");

  return mlirBytecodeFailure();
}
#endif

void *mlirBytecodeAllocate(void *context, size_t bytes) {
  return malloc(bytes);
}

void mlirBytecodeFree(void *context, void *ptr) { free(ptr); }

MlirBytecodeBytesRef getString(void *context, MlirBytecodeStringHandle hdl) {
  ParsingState *state = context;
  assert(state->strings);
  return state->strings[hdl.id];
}

MlirBytecodeStatus
mlirBytecodeGetStringSectionValue(void *context, MlirBytecodeStringHandle idx,
                                  MlirBytecodeBytesRef *result) {
  *result = getString(context, idx);
  return mlirBytecodeSuccess();
}

MlirBytecodeOpRef mlirBytecodeGetOpName(void *context,
                                        MlirBytecodeOpHandle hdl) {
  ParsingState *state = context;
  assert(state->ops);
  return state->ops[hdl.id];
}

MlirBytecodeStatus mlirBytecodeParseAttribute(void *context,
                                              MlirBytecodeAttrHandle handle) {
  ParsingState *state = context;
  if (!state->attributes)
    return mlirBytecodeUnhandled();

  if (state->attributes[handle.id].length)
    return mlirBytecodeSuccess();

  MlirBytecodeAttributeOrTypeRange attr = state->attributeRange[handle.id];
  if (!attr.hasCustom) {
    int len = attr.bytes.length + sizeof("Textual()");
    state->attributes[handle.id].data = (uint8_t *)malloc(len);
    state->attributes[handle.id].length =
        snprintf((char *)state->attributes[handle.id].data, len,
                 "Textual(%.*s)", (int)attr.bytes.length, attr.bytes.data);

    mlirBytecodeEmitDebug("%.*s", (int)state->attributes[handle.id].length,
                          state->attributes[handle.id].data);
    return mlirBytecodeSuccess();
  }

  // Attribute parsing etc should happen here.
  if (attr.dialectHandle.id < MAX_DIALECTS &&
      state->attrCallbacks[attr.dialectHandle.id]) {
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(attr.bytes);
    MlirBytecodeDialectReader reader = {.context = context, .stream = &stream};
    return state->attrCallbacks[attr.dialectHandle.id](&reader, handle);
  }

  mlirBytecodeEmitDebug("attr unhandled");
  return mlirBytecodeUnhandled();
}

MlirMutableBytesRef getAttribute(void *context, MlirBytecodeAttrHandle attr) {
  static char empty[] = "<<unknown>>";
  ParsingState *state = context;
  if (mlirBytecodeSucceeded(mlirBytecodeParseAttribute(context, attr)))
    return state->attributes[attr.id];
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

MlirBytecodeStatus mlirBytecodeParseType(void *context,
                                         MlirBytecodeTypeHandle handle) {
  ParsingState *state = context;
  if (!state->types)
    return mlirBytecodeUnhandled();

  if (state->types[handle.id].length)
    return mlirBytecodeSuccess();

  MlirBytecodeAttributeOrTypeRange type = state->typeRange[handle.id];
  mlirBytecodeEmitDebug("type %d %d", type.dialectHandle.id, handle.id);
  if (!type.hasCustom) {
    int len = type.bytes.length + sizeof("Textual()");
    state->types[handle.id].data = (uint8_t *)malloc(len);
    state->types[handle.id].length =
        snprintf((char *)state->types[handle.id].data, len, "Textual(%.*s)",
                 (int)type.bytes.length, type.bytes.data);

    mlirBytecodeEmitDebug("%.*s", (int)state->types[handle.id].length,
                          state->types[handle.id].data);
    return mlirBytecodeSuccess();
  }

  // Type parsing etc should happen here.
  if (type.dialectHandle.id < MAX_DIALECTS &&
      state->typeCallbacks[type.dialectHandle.id]) {
    MlirBytecodeStream stream = mlirBytecodeStreamCreate(type.bytes);
    MlirBytecodeDialectReader reader = {.context = context, .stream = &stream};
    return state->typeCallbacks[type.dialectHandle.id](&reader, handle);
  }

  return mlirBytecodeUnhandled();
}

MlirMutableBytesRef getType(void *context, MlirBytecodeTypeHandle type) {
  static char empty[] = "<<unknown>>";
  ParsingState *state = context;
  if (mlirBytecodeSucceeded(mlirBytecodeParseType(context, type)))
    return state->types[type.id];
  return (MlirMutableBytesRef){.data = (uint8_t *)&empty[0],
                               .length = sizeof(empty)};
}

// -----------------------------------
// Define dialect construction methods.

MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *context, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col) {
  ParsingState *state = context;
  MlirBytecodeStatus ret = mlirBytecodeParseAttribute(context, filename);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  MlirMutableBytesRef str = getAttribute(context, filename);
  // Should avoid log here ...
  int len = str.length + sizeof("FileLineColLoc(::)") + ceil(log10(line + 1)) +
            ceil(log10(col + 1)) + 1;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length =
      snprintf((char *)state->attributes[attrHandle.id].data, len,
               "FileLineColLoc(%.*s:%" PRId64 ":%" PRId64 ")", (int)str.length,
               str.data, line, col);

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnitAttr(void *context,
                                  MlirBytecodeAttrHandle attrHandle) {
  ParsingState *state = context;
  int len = sizeof("Unit");
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length =
      snprintf((char *)state->attributes[attrHandle.id].data, len, "Unit");

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *context,
                                    MlirBytecodeAttrHandle attrHandle) {
  ParsingState *state = context;
  int len = sizeof("UnknownLoc");
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length = snprintf(
      (char *)state->attributes[attrHandle.id].data, len, "UnknownLoc");

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinStringAttr(void *context,
                                    MlirBytecodeAttrHandle attrHandle,
                                    MlirBytecodeBytesRef str) {
  ParsingState *state = context;
  int len = str.length + 3;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[attrHandle.id].length =
      snprintf((char *)state->attributes[attrHandle.id].data, len, "\"%.*s\"",
               (int)str.length, str.data);

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinStringAttrWithType(
    void *context, MlirBytecodeAttrHandle bcAttrHandle,
    MlirBytecodeBytesRef str, MlirBytecodeTypeHandle type) {
  ParsingState *state = context;

  MlirMutableBytesRef typestr = getType(context, type);
  if (typestr.data == 0)
    return mlirBytecodeFailure();
  int len = str.length + 30;
  state->attributes[bcAttrHandle.id].data = (uint8_t *)malloc(len);
  state->attributes[bcAttrHandle.id].length = snprintf(
      (char *)state->attributes[bcAttrHandle.id].data, len, "\"%.*s, %.*s\"",
      (int)str.length, str.data, (int)typestr.length, typestr.data);

  mlirBytecodeEmitDebug("%s", state->attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *context,
                                        MlirBytecodeAttrHandle attrHandle,
                                        MlirDictionaryHandleIterator *range) {
  ParsingState *state = context;
  int kMax = 200;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  int len = 0;

  len += snprintf((char *)state->attributes[attrHandle.id].data + len,
                  kMax - len, "{");
  MlirBytecodeAttrHandle name;
  MlirBytecodeAttrHandle value;
  while (mlirBytecodeSucceeded(
      mlirBytecodeGetNextDictionaryHandles(context, range, &name, &value))) {
    MlirBytecodeStatus ret = mlirBytecodeParseAttribute(context, name);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    ret = mlirBytecodeParseAttribute(context, value);
    if (!mlirBytecodeSucceeded(ret))
      return ret;
    MlirMutableBytesRef nameStr = state->attributes[name.id];
    MlirMutableBytesRef valueStr = state->attributes[value.id];
    len += snprintf((char *)state->attributes[attrHandle.id].data + len,
                    kMax - len, " %.*s = %.*s;", (int)nameStr.length,
                    nameStr.data, (int)valueStr.length, valueStr.data);
  }
  len += snprintf((char *)state->attributes[attrHandle.id].data + len,
                  kMax - len, " }");

  state->attributes[attrHandle.id].length = len;
  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *context, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeTypeHandle typeHandle, uint64_t value) {
  ParsingState *state = context;
  char kMax = 30;
  state->attributes[attrHandle.id].data = (uint8_t *)malloc(kMax);
  state->attributes[attrHandle.id].length = snprintf(
      (char *)state->attributes[attrHandle.id].data, kMax,
      "Int(%.*s, %" PRId64 ")", (int)state->types[typeHandle.id].length,
      state->types[typeHandle.id].data, value);

  mlirBytecodeEmitDebug("%s", state->attributes[attrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *context,
                                  MlirBytecodeAttrHandle bcAttrHandle,
                                  MlirBytecodeTypeHandle value) {
  ParsingState *state = context;
  MlirMutableBytesRef type = getType(context, value);
  state->attributes[bcAttrHandle.id].data = malloc(type.length + 1);
  memcpy(state->attributes[bcAttrHandle.id].data, type.data, type.length + 1);
  state->attributes[bcAttrHandle.id].length = type.length;

  mlirBytecodeEmitDebug("%s", state->attributes[bcAttrHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *context,
                                     MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = context;
  int len = sizeof("f32");
  state->types[typeHandle.id].data = (uint8_t *)malloc(len);
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data, len, "f32");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *context,
                                   MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = context;
  int len = sizeof("index");
  state->types[typeHandle.id].data = (uint8_t *)malloc(len);
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data, len, "index");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *context, MlirBytecodeTypeHandle typeHandle,
    MlirBuiltinSignednessSemantics signedness, int width) {
  ParsingState *state = context;
  int len = ceil(log10(width)) + 4;
  state->types[typeHandle.id].data = (uint8_t *)malloc(len);
  int i = 0;
  if (signedness == kBuiltinIntegerTypeSigned) {
    state->types[typeHandle.id].data[i++] = 's';
  } else if (signedness == kBuiltinIntegerTypeUnsigned) {
    state->types[typeHandle.id].data[i++] = 'u';
  }
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data + i, len, "i%d",
               width) +
      i + 1;

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *context,
                                      MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = context;
  if (!state->types)
    return mlirBytecodeUnhandled();
  if (state->types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  state->types[typeHandle.id].data = (uint8_t *)malloc(26);
  state->types[typeHandle.id].length = snprintf(
      (char *)state->types[typeHandle.id].data, 26, "function (...) -> (...)");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *context,
                                          MlirBytecodeTypeHandle typeHandle) {
  ParsingState *state = context;
  if (!state->types)
    return mlirBytecodeUnhandled();
  if (state->types[typeHandle.id].data)
    return mlirBytecodeSuccess();

  state->types[typeHandle.id].data = (uint8_t *)malloc(20);
  state->types[typeHandle.id].length =
      snprintf((char *)state->types[typeHandle.id].data, 20, "tensor<...>");

  mlirBytecodeEmitDebug("%s", state->types[typeHandle.id].data);
  return mlirBytecodeSuccess();
}

// ----

MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *context, MlirBytecodeTypeHandle typeHandle, unsigned *width) {
  ParsingState *state = context;
  if (!state->types)
    return mlirBytecodeUnhandled();

  MlirBytecodeStatus ret = mlirBytecodeParseType(context, typeHandle);
  if (!mlirBytecodeSucceeded(ret))
    return ret;
  MlirMutableBytesRef type = state->types[typeHandle.id];
  if (!type.length)
    return mlirBytecodeFailure();

  if (strncmp((char *)type.data, "index", type.length) == 0) {
    *width = 64;
    return mlirBytecodeSuccess();
  }

  char *c = (char *)type.data;
  for (unsigned i = 0; i < type.length; ++i) {
    if ('0' <= *c && *c <= '9')
      break;
    ++c;
  }
  int res = sscanf(c, "%u", width);
  return res == 1 ? mlirBytecodeSuccess() : mlirBytecodeFailure();
}

// ----

static MlirBytecodeStatus mlirBytecodeAttributesPush(void *context,
                                                     MlirBytecodeSize total) {
  ParsingState *state = context;
  // Note: this currently assumes that number of state->attributes don't change.
  if (state->attributes) {
    free(state->attributes);
  }

  state->attributes =
      (MlirMutableBytesRef *)malloc(total * sizeof(*state->attributes));
  memset(state->attributes, 0, total * sizeof(*state->attributes));
  state->attributeRange = (MlirBytecodeAttributeOrTypeRange *)malloc(
      total * sizeof(*state->attributeRange));
  mlirBytecodeEmitDebug("alloc'd state->attributes / %d", (int)total);
  state->numAttributes = total;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mlirBytecodeAssociateAttributeRange(void *context,
                                    MlirBytecodeAttrHandle attrHandle,
                                    MlirBytecodeDialectHandle dialectHandle,
                                    MlirBytecodeBytesRef str, bool hasCustom) {
  ParsingState *state = context;
  state->attributeRange[attrHandle.id].bytes = str;
  state->attributeRange[attrHandle.id].hasCustom = hasCustom;
  state->attributeRange[attrHandle.id].dialectHandle = dialectHandle;
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus mlirBytecodeTypesPush(void *context,
                                                MlirBytecodeSize total) {
  ParsingState *state = context;
  if (state->types) {
    free(state->types);
  }
  state->types = (MlirMutableBytesRef *)malloc(total * sizeof(*state->types));
  memset(state->types, 0, total * sizeof(*state->types));
  state->typeRange = (MlirBytecodeAttributeOrTypeRange *)malloc(
      total * sizeof(*state->typeRange));
  state->numTypes = total;
  mlirBytecodeEmitDebug("alloc'd state->types");
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus
mlirBytecodeAssociateTypeRange(void *context, MlirBytecodeTypeHandle typeHandle,
                               MlirBytecodeDialectHandle dialectHandle,
                               MlirBytecodeBytesRef str, bool hasCustom) {
  ParsingState *state = context;
  state->typeRange[typeHandle.id].bytes = str;
  state->typeRange[typeHandle.id].hasCustom = hasCustom;
  state->typeRange[typeHandle.id].dialectHandle = dialectHandle;
  mlirBytecodeEmitDebug("type[%d] %d dialect = %d", typeHandle.id, hasCustom,
                        dialectHandle.id);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectOpNames(void *context,
                                              MlirBytecodeSize numOps) {
  ParsingState *state = context;
  mlirBytecodeEmitDebug("number of ops %" PRIu64, numOps);
  if (state->ops)
    free(state->ops);
  state->ops = malloc(numOps * sizeof(*state->ops));
  memset(state->ops, 0, numOps * sizeof(*state->ops));
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectOpCallBack(void *context, MlirBytecodeOpHandle opHdl,
                              MlirBytecodeDialectHandle dialectHandle,
                              MlirBytecodeStringHandle stringHdl) {
  ParsingState *state = context;
  mlirBytecodeEmitDebug("\t\tdialect[%d] :: op[%d] = %d", (int)dialectHandle.id,
                        (int)opHdl.id, (int)stringHdl.id);

  state->ops[opHdl.id] =
      (MlirBytecodeOpRef){.dialect = dialectHandle, .op = stringHdl};
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeDialectsPush(void *context,
                                            MlirBytecodeSize total) {
  ParsingState *state = context;
  if (state->dialectStr) {
    free(state->dialectStr);
  }
  state->dialectStr = malloc(total * sizeof(*state->dialectStr));
  memset(state->dialectStr, 0, total * sizeof(*state->dialectStr));
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeDialectCallBack(void *context,
                            MlirBytecodeDialectHandle dialectHandle,
                            MlirBytecodeStringHandle stringHdl) {
  ParsingState *state = context;
  MlirBytecodeBytesRef dialect = getString(context, stringHdl);
  mlirBytecodeEmitDebug("\t\tdialect[%d] = %s", (int)dialectHandle.id,
                        dialect.data);

  if (dialectHandle.id < MAX_DIALECTS) {
    if (strncmp((char *)dialect.data, "builtin", dialect.length) == 0) {
      mlirBytecodeEmitDebug("builtin handler set for %d", dialectHandle.id);
      state->attrCallbacks[dialectHandle.id] = &mlirBytecodeParseBuiltinAttr;
      state->typeCallbacks[dialectHandle.id] = &mlirBytecodeParseBuiltinType;
    }
  }

  state->dialectStr[dialectHandle.id] = stringHdl;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeStringsPush(void *context,
                                           MlirBytecodeSize numStrings) {
  ParsingState *state = context;
  if (state->strings) {
    free(state->strings);
  }
  state->strings =
      (MlirBytecodeBytesRef *)malloc(numStrings * sizeof(*state->strings));
  memset(state->strings, 0, numStrings * sizeof(*state->strings));
  mlirBytecodeEmitDebug("alloc'd state->strings");
  return mlirBytecodeSuccess();
}
MlirBytecodeStatus
mlirBytecodeAssociateStringRange(void *context, MlirBytecodeStringHandle hdl,
                                 MlirBytecodeBytesRef bytes) {
  ParsingState *state = context;
  state->strings[hdl.id] = bytes;
  return mlirBytecodeSuccess();
}

enum {
  /// Sentinel to indicate unset/unknown handle.
  kMlirBytecodeHandleSentinel = -1,
};
static bool hasAttrDict(MlirBytecodeAttrHandle attr) {
  return attr.id == (uint64_t)kMlirBytecodeHandleSentinel;
}

MlirBytecodeOperationState states[MLIRBC_IR_STACK_MAX_DEPTH + 1];
int stateDepth;

void mlirBytecodeIRSectionEnter(void *context, void *retBlock) {}

MlirBytecodeStatus mlirBytecodeOperationStatePush(
    void *context, MlirBytecodeOpHandle name, MlirBytecodeLocHandle loc,
    MlirBytecodeOperationStateHandle *opStateHandle) {
  MlirBytecodeOperationState *opState = &states[stateDepth++];
  mlirBytecodeEmitDebug("operation state push [depth = %d]", stateDepth);
  *opStateHandle = opState;
  opState->name = name;
  opState->loc = loc;
  opState->attrDict.id = kMlirBytecodeHandleSentinel;
  opState->hasOperands = false;
  opState->hasRegions = false;
  opState->hasResults = false;
  opState->hasSuccessors = false;
  opState->isIsolated = false;

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPush(void *context,
                               MlirBytecodeOperationHandle opHandle,
                               MlirBytecodeSize numArgs) {
  ParsingState *state = context;
  MlirBytecodeOperationState *opState = opHandle;
  opState->numArgsRemaining = numArgs;
  printf("%*cblock", state->indentSize, '_');
  if (numArgs)
    printf(" ");
  state->indentSize += 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationBlockAddArgument(
    void *context, MlirBytecodeOperationHandle opHandle,
    MlirBytecodeTypeHandle type, MlirBytecodeLocHandle loc) {
  ParsingState *state = context;
  MlirBytecodeOperationState *opState = opHandle;

  // Note: block args locs push the print form too much.
  printf("%%%d : %s ", state->ssaIdStack[state->depth]++,
         getType(context, type).data);
  if (!(--opState->numArgsRemaining))
    printf("\n");
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationBlockPop(void *context,
                              MlirBytecodeOperationHandle opHandle) {
  ParsingState *state = context;
  state->indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPush(void *context,
                                MlirBytecodeOperationHandle opHandle,
                                size_t numBlocks, size_t numValues) {
  ParsingState *state = context;
  MlirBytecodeOperationState *opState = opHandle;
  if (state->depth + 1 == MAX_DEPTH)
    return mlirBytecodeUnhandled();

  if (opState->isIsolated)
    state->ssaIdStack[++state->depth] = 0;
  else {
    state->ssaIdStack[state->depth + 1] =
        state->regionNumValues[state->depth] +
        (state->depth > 1 ? state->ssaIdStack[state->depth - 1] : 0);
    ++state->depth;
  }
  state->regionNumValues[state->depth] = numValues;
  state->indentSize += 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationRegionPop(void *context,
                               MlirBytecodeOperationStateHandle opStateHandle) {
  ParsingState *state = context;
  --state->depth;
  state->indentSize -= 2;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddAttributeDictionary(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeAttrHandle attrs) {
  MlirBytecodeOperationState *opState = opStateHandle;
  opState->attrDict = attrs;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus printOperationPrefix(void *context,
                                        MlirBytecodeOperationState *opState) {
  ParsingState *state = context;
  MlirBytecodeOpRef opRef = mlirBytecodeGetOpName(context, opState->name);
  MlirBytecodeBytesRef opName = getString(context, opRef.op);
  MlirBytecodeBytesRef dialectName = getString(context, opRef.dialect);

  if (opState->hasResults)
    printf("= ");
  else
    printf("%*c", state->indentSize, ' ');

  printf("%.*s.%.*s", (int)dialectName.length, dialectName.data,
         (int)opName.length, opName.data);

  if (!hasAttrDict(opState->attrDict)) {
    MlirBytecodeStatus ret =
        mlirBytecodeParseAttribute(context, opState->attrDict);
    if (mlirBytecodeFailed(ret)) {
      return ret;
    } else if (mlirBytecodeSucceeded(ret)) {
      MlirMutableBytesRef attrDictVal = state->attributes[opState->attrDict.id];
      printf(" %.*s ", (int)attrDictVal.length, attrDictVal.data);
    } else {
      printf(" << unhandled >> ");
    }
  }

  if (opState->hasOperands)
    printf(" ");

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultTypes(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSize numResults) {
  ParsingState *state = context;
  MlirBytecodeOperationState *opState = opStateHandle;
  opState->hasResults = true;
  printf("%*c", state->indentSize, ' ');
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddResultType(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeTypeHandle type) {
  ParsingState *state = context;
  printf("%%%d", state->ssaIdStack[state->depth]++);
  MlirBytecodeStatus ret = mlirBytecodeParseType(context, type);
  if (mlirBytecodeSucceeded(ret)) {
    printf(":%.*s ", (int)state->types[type.id].length,
           state->types[type.id].data);
  } else if (mlirBytecodeFailed(ret)) {
    return ret;
  } else {
    printf(":<<unknown type>> ");
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperands(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSize numOperands) {
  MlirBytecodeOperationState *opState = opStateHandle;
  opState->hasOperands = true;
  return printOperationPrefix(context, opState);
}

MlirBytecodeStatus mlirBytecodeOperationStateAddOperand(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeValueHandle value) {
  printf("%%%d ", (int)value.id);

  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddRegions(
    void *context, MlirBytecodeOperationStateHandle opStateHandle, uint64_t n,
    bool isIsolatedFromAbove) {
  MlirBytecodeOperationState *opState = opStateHandle;
  mlirBytecodeEmitDebug("numRegions = %d", (int)n);
  if (!opState->hasOperands)
    if (!mlirBytecodeSucceeded(printOperationPrefix(context, opState)))
      return mlirBytecodeSuccess();
  opState->hasRegions = true;
  opState->isIsolated = isIsolatedFromAbove;
  printf("\n");
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessors(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeSize numSuccessors) {
  if (numSuccessors > 0) {
    printf(" // successors:");
    MlirBytecodeOperationState *opState = opStateHandle;
    if (!opState->hasOperands && !opState->hasRegions)
      if (!mlirBytecodeSucceeded(printOperationPrefix(context, opState)))
        return mlirBytecodeSuccess();
    opState->hasSuccessors = true;
  }
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeOperationStateAddSuccessor(
    void *context, MlirBytecodeOperationStateHandle opStateHandle,
    MlirBytecodeHandle successor) {
  printf(" ^%" PRIu64, successor.id);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeOperationStatePop(void *context,
                              MlirBytecodeOperationStateHandle opStateHandle,
                              MlirBytecodeOperationHandle *opHandle) {
  MlirBytecodeOperationState *opState = opStateHandle;
  mlirBytecodeEmitDebug("operation state pop [from depth = %d]", stateDepth);
  --stateDepth;
  if (!opState->hasOperands && !opState->hasRegions && !opState->hasSuccessors)
    if (!mlirBytecodeSucceeded(printOperationPrefix(context, opState)))
      return mlirBytecodeFailure();
  if (!opState->hasRegions)
    printf("\n");
  *opHandle = opStateHandle;
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceEmptyCallBack(void *context,
                                  MlirBytecodeStringHandle resourceKey) {
  return mlirBytecodeUnhandled();
}

MlirBytecodeStatus
mlirBytecodeResourceDialectGroupEnter(void *context,
                                      MlirBytecodeDialectHandle dialect,
                                      MlirBytecodeSize numResources) {
  printf("\tgroup %d\n", (int)numResources);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceExternalGroupEnter(void *context,
                                       MlirBytecodeStringHandle groupKey,
                                       MlirBytecodeSize numResources) {
  printf("\tgroup '%s'\n", getString(context, groupKey).data);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBlobCallBack(
    void *context, MlirBytecodeStringHandle resourceKey,
    MlirBytecodeSize alignment, MlirBytecodeBytesRef blob) {
  printf("\t\tblob size(resource %s) = %" PRIu64 "\n",
         getString(context, resourceKey).data, blob.length);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus mlirBytecodeResourceBoolCallBack(
    void *context, MlirBytecodeStringHandle resourceKey, uint8_t boolResource) {
  printf("\t\tbool resource %s = %d\n", getString(context, resourceKey).data,
         boolResource);
  return mlirBytecodeSuccess();
}

MlirBytecodeStatus
mlirBytecodeResourceStringCallBack(void *context,
                                   MlirBytecodeStringHandle resourceKey,
                                   MlirBytecodeStringHandle strHandle) {
  printf("\t\tstring resource %s = %s\n", getString(context, resourceKey).data,
         getString(context, strHandle).data);
  return mlirBytecodeSuccess();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s file.mlirbc\n", argv[0]);
    return 1;
  }

  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    return mlirBytecodeEmitError(NULL, "MlirBytecodeFailed to open file '%s'",
                                 argv[1]),
           1;
  }

  struct stat fileInfo;
  if (fstat(fd, &fileInfo) == -1) {
    return mlirBytecodeEmitError(NULL, "error getting the file size"), 1;
  }

  uint8_t *stream =
      (uint8_t *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (stream == MAP_FAILED) {
    close(fd);
    return mlirBytecodeEmitError(NULL, "error mmapping the file"), 1;
  }

  // Set parsing state
  ParsingState state;
  state.attributes = 0;
  state.depth = 0;
  state.dialectStr = 0;
  state.indentSize = 0;
  state.ops = 0;
  state.regionNumValues[0] = 0;
  state.ssaIdStack[0] = 0;
  state.strings = 0;
  state.types = 0;

  MlirBytecodeBytesRef ref = {.data = stream, .length = fileInfo.st_size};
  MlirBytecodeParserState parserState =
      mlirBytecodePopulateParserState(&state, ref);

  if (!mlirBytecodeParserStateEmpty(&parserState) &&
      mlirBytecodeFailed(mlirBytecodeParse(&state, &parserState, NULL))) {
    return mlirBytecodeEmitError(NULL, "MlirBytecodeFailed to parse file"), 1;
  }

  free(state.ops);
  free(state.attributeRange);
  for (int i = 0, e = state.numAttributes; i != e; ++i)
    free(state.attributes[i].data);
  free(state.attributes);
  free(state.typeRange);
  for (int i = 0, e = state.numTypes; i != e; ++i)
    free(state.types[i].data);
  free(state.types);
  free(state.dialectStr);
  free(state.strings);
  if (munmap(stream, fileInfo.st_size) == -1) {
    close(fd);
    perror("Error un-mmapping the file");
    exit(EXIT_FAILURE);
  }
  return 0;
}

#pragma GCC diagnostic pop
