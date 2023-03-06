//===-- BuiltinParse.c.inc - Builtin attr & type parsers ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Builtin Attribute & Type driver.
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Entry point for Builtin dialect Attribute parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(MlirBytecodeDialectReader *reader,
                             MlirBytecodeAttrHandle attrHandle);

// Entry point for Builtin dialect Type parsing.
MlirBytecodeStatus
mlirBytecodeParseBuiltinType(MlirBytecodeDialectReader *reader,
                             MlirBytecodeTypeHandle typeHandle);

// NOTE: this is proof-of-concept and just enough working to show end-to-end.
// The parsing of attributes/types (especially those with arrays) will change.

//===----------------------------------------------------------------------===//
// Create attributes.

// Functions defined by folks adding support of builtin dialect parsing. The
// `attrHandle` is the handle corresponding to the attribute being created. The
// defined function will/can set attribute[attrHandle] if successfully decoded.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerAttr(
    void *context, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeTypeHandle typeHandle, uint64_t value);
static MlirBytecodeStatus mlirBytecodeCreateBuiltinFileLineColLoc(
    void *context, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeAttrHandle filename, uint64_t line, uint64_t col);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnknownLoc(void *context,
                                    MlirBytecodeAttrHandle attrHandle);

// Handle iterator.
struct MlirBytecodeHandleIterator {
  // Stream over block args.
  MlirBytecodeStream stream;

  // Number of handles in stream.
  const MlirBytecodeSize count;
};
typedef struct MlirBytecodeHandleIterator MlirBytecodeHandleIterator;
typedef struct MlirBytecodeHandleIterator MlirDictionaryHandleIterator;

// Populate the next key and value and increment the iterator.
// Returns whether there was an element.
MlirBytecodeStatus mlirBytecodeGetNextDictionaryHandles(
    void *context, MlirDictionaryHandleIterator *iterator,
    MlirBytecodeAttrHandle *name, MlirBytecodeAttrHandle *value);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinDictionaryAttr(void *context,
                                        MlirBytecodeAttrHandle attrHandle,
                                        MlirDictionaryHandleIterator *range);

// Create method for TypeAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *context,
                                  MlirBytecodeAttrHandle attrHandle,
                                  MlirBytecodeTypeHandle value);

//===----------------------------------------------------------------------===//
// Create types.

static MlirBytecodeStatus
mlirBytecodeCreateBuiltinFloat32Type(void *context,
                                     MlirBytecodeTypeHandle typeHandle);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinIndexType(void *context,
                                   MlirBytecodeTypeHandle typeHandle);

enum MlirBuiltinSignednessSemantics {
  kBuiltinIntegerTypeSignless, /// No signedness semantics
  kBuiltinIntegerTypeSigned,   /// Signed integer
  kBuiltinIntegerTypeUnsigned, /// Unsigned integer
};
typedef enum MlirBuiltinSignednessSemantics MlirBuiltinSignednessSemantics;

static MlirBytecodeStatus mlirBytecodeCreateBuiltinIntegerType(
    void *context, MlirBytecodeTypeHandle typeHandle,
    MlirBuiltinSignednessSemantics signedness, int width);

// TODO
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinFunctionType(void *context,
                                      MlirBytecodeTypeHandle typeHandle);
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinRankedTensorType(void *context,
                                          MlirBytecodeTypeHandle typeHandle);

//===----------------------------------------------------------------------===//
// Query types.

// Queries the bitwidth of the builtin integer type.
static MlirBytecodeStatus mlirBytecodeQueryBuiltinIntegerTypeWidth(
    void *context, MlirBytecodeTypeHandle typeHandle, unsigned *width);

// Create method for StringAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinStringAttr(void *context,
                                    MlirBytecodeAttrHandle attrHandle,
                                    MlirBytecodeBytesRef value);

// Create method for StringAttrWithType.
static MlirBytecodeStatus mlirBytecodeCreateBuiltinStringAttrWithType(
    void *context, MlirBytecodeAttrHandle attrHandle,
    MlirBytecodeBytesRef value, MlirBytecodeTypeHandle type);

// Create method for TypeAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinTypeAttr(void *context,
                                  MlirBytecodeAttrHandle attrHandle,
                                  MlirBytecodeTypeHandle value);

// Create method for UnitAttr.
static MlirBytecodeStatus
mlirBytecodeCreateBuiltinUnitAttr(void *context,
                                  MlirBytecodeAttrHandle attrHandle);

#ifdef __cplusplus
}
#endif

enum AttributeCode {
  ///   ArrayAttr {
  ///     elements: Attribute[]
  ///   }
  ///
  kBuiltinAttrArrayAttrCode = 0,

  ///   DictionaryAttr {
  ///     attrs: <StringAttr, Attribute>[]
  ///   }
  kBuiltinAttrDictionaryAttrCode = 1,

  ///   StringAttr {
  ///     value: string
  ///   }
  kBuiltinAttrStringAttrCode = 2,

  ///   StringAttrWithType {
  ///     value: string,
  ///     type: Type
  ///   }
  /// A variant of StringAttr with a type.
  kBuiltinAttrStringAttrWithTypeCode = 3,

  ///   FlatSymbolRefAttr {
  ///     rootReference: StringAttr
  ///   }
  /// A variant of SymbolRefAttr with no leaf references.
  kBuiltinAttrFlatSymbolRefAttrCode = 4,

  ///   SymbolRefAttr {
  ///     rootReference: StringAttr,
  ///     leafReferences: FlatSymbolRefAttr[]
  ///   }
  kBuiltinAttrSymbolRefAttrCode = 5,

  ///   TypeAttr {
  ///     value: Type
  ///   }
  kBuiltinAttrTypeAttrCode = 6,

  ///   UnitAttr {
  ///   }
  kBuiltinAttrUnitAttrCode = 7,

  ///   IntegerAttr {
  ///     type: Type
  ///     value: APInt,
  ///   }
  kBuiltinAttrIntegerAttrCode = 8,

  ///   FloatAttr {
  ///     type: FloatType
  ///     value: APFloat
  ///   }
  kBuiltinAttrFloatAttrCode = 9,

  ///   CallSiteLoc {
  ///    callee: LocationAttr,
  ///    caller: LocationAttr
  ///   }
  kBuiltinAttrCallSiteLocCode = 10,

  ///   FileLineColLoc {
  ///     file: StringAttr,
  ///     line: varint,
  ///     column: varint
  ///   }
  kBuiltinAttrFileLineColLocCode = 11,

  ///   FusedLoc {
  ///     locations: LocationAttr[]
  ///   }
  kBuiltinAttrFusedLocCode = 12,

  ///   FusedLocWithMetadata {
  ///     locations: LocationAttr[],
  ///     metadata: Attribute
  ///   }
  /// A variant of FusedLoc with metadata.
  kBuiltinAttrFusedLocWithMetadataCode = 13,

  ///   NameLoc {
  ///     name: StringAttr,
  ///     childLoc: LocationAttr
  ///   }
  kBuiltinAttrNameLocCode = 14,

  ///   UnknownLoc {
  ///   }
  kBuiltinAttrUnknownLocCode = 15,

  ///   DenseResourceElementsAttr {
  ///     type: Type,
  ///     handle: ResourceHandle
  ///   }
  kBuiltinAttrDenseResourceElementsAttrCode = 16,

  ///   DenseArrayAttr {
  ///     type: RankedTensorType,
  ///     data: blob
  ///   }
  kBuiltinAttrDenseArrayAttrCode = 17,

  ///   DenseIntOrFPElementsAttr {
  ///     type: ShapedType,
  ///     data: blob
  ///   }
  kBuiltinAttrDenseIntOrFPElementsAttrCode = 18,

  ///   DenseStringElementsAttr {
  ///     type: ShapedType,
  ///     isSplat: varint,
  ///     data: string[]
  ///   }
  kBuiltinAttrDenseStringElementsAttrCode = 19,

  ///   SparseElementsAttr {
  ///     type: ShapedType,
  ///     indices: DenseIntElementsAttr,
  ///     values: DenseElementsAttr
  ///   }
  kBuiltinAttrSparseElementsAttrCode = 20,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
  ///   IntegerType {
  ///     widthAndSignedness: varint // (width << 2) | (signedness)
  ///   }
  ///
  kBuiltinTypeIntegerTypeCode = 0,

  ///   IndexType {
  ///   }
  ///
  kBuiltinTypeIndexTypeCode = 1,

  ///   FunctionType {
  ///     inputs: Type[],
  ///     results: Type[]
  ///   }
  ///
  kBuiltinTypeFunctionTypeCode = 2,

  ///   BFloat16Type {
  ///   }
  ///
  kBuiltinTypeBFloat16TypeCode = 3,

  ///   Float16Type {
  ///   }
  ///
  kBuiltinTypeFloat16TypeCode = 4,

  ///   Float32Type {
  ///   }
  ///
  kBuiltinTypeFloat32TypeCode = 5,

  ///   Float64Type {
  ///   }
  ///
  kBuiltinTypeFloat64TypeCode = 6,

  ///   Float80Type {
  ///   }
  ///
  kBuiltinTypeFloat80TypeCode = 7,

  ///   Float128Type {
  ///   }
  ///
  kBuiltinTypeFloat128TypeCode = 8,

  ///   ComplexType {
  ///     elementType: Type
  ///   }
  ///
  kBuiltinTypeComplexTypeCode = 9,

  ///   MemRefType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  ///
  kBuiltinTypeMemRefTypeCode = 10,

  ///   MemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type,
  ///     layout: Attribute
  ///   }
  /// Variant of MemRefType with non-default memory space.
  kBuiltinTypeMemRefTypeWithMemSpaceCode = 11,

  ///   NoneType {
  ///   }
  ///
  kBuiltinTypeNoneTypeCode = 12,

  ///   RankedTensorType {
  ///     shape: svarint[],
  ///     elementType: Type,
  ///   }
  ///
  kBuiltinTypeRankedTensorTypeCode = 13,

  ///   RankedTensorTypeWithEncoding {
  ///     encoding: Attribute,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of RankedTensorType with an encoding.
  kBuiltinTypeRankedTensorTypeWithEncodingCode = 14,

  ///   TupleType {
  ///     elementTypes: Type[]
  ///   }
  kBuiltinTypeTupleTypeCode = 15,

  ///   UnrankedMemRefType {
  ///     shape: svarint[]
  ///   }
  ///
  kBuiltinTypeUnrankedMemRefTypeCode = 16,

  ///   UnrankedMemRefTypeWithMemSpace {
  ///     memorySpace: Attribute,
  ///     shape: svarint[]
  ///   }
  /// Variant of UnrankedMemRefType with non-default memory space.
  kBuiltinTypeUnrankedMemRefTypeWithMemSpaceCode = 17,

  ///   UnrankedTensorType {
  ///     elementType: Type
  ///   }
  ///
  kBuiltinTypeUnrankedTensorTypeCode = 18,

  ///   VectorType {
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  ///
  kBuiltinTypeVectorTypeCode = 19,

  ///   VectorTypeWithScalableDims {
  ///     numScalableDims: varint,
  ///     shape: svarint[],
  ///     elementType: Type
  ///   }
  /// Variant of VectorType with scalable dimensions.
  kBuiltinTypeVectorTypeWithScalableDimsCode = 20,
};

static MlirBytecodeStatus parseIntegerAttr(MlirBytecodeDialectReader *reader,
                                           MlirBytecodeAttrHandle attrHandle) {
  MlirBytecodeTypeHandle typeHandle;
  if (mlirBytecodeFailed(mlirBytecodeDialectReaderReadType(reader, &typeHandle)))
    return mlirBytecodeFailure();
  // Example of tricky case: without being able to construct the type, there
  // is nothing to query the bitwidth of the type except to ask the
  // implementation what width the type handle has.
  unsigned width;
  MlirBytecodeStatus ret = 
      mlirBytecodeQueryBuiltinIntegerTypeWidth(reader->context, typeHandle, &width);
  if (!mlirBytecodeHandled(ret))
    return ret;
  if (mlirBytecodeFailed(ret))
    return mlirBytecodeEmitError(
        reader->context, "expected integer or index type for IntegerAttr");
  // TODO: Handle arbitrary integer sizes.
  if (width > 64)
    return mlirBytecodeUnhandled();

  MlirBytecodeAPInt val;
  ret = mlirBytecodeDialectReaderReadAPIntWithKnownWidth(reader, width, NULL, &val);
  return mlirBytecodeCreateBuiltinIntegerAttr(reader->context, attrHandle,
                                              typeHandle, val.U.value);
}

MlirBytecodeStatus mlirBytecodeGetNextDictionaryHandles(
    void *context, MlirDictionaryHandleIterator *iterator,
    MlirBytecodeAttrHandle *name, MlirBytecodeAttrHandle *value) {
  if (mlirBytecodeFailed(
          mlirBytecodeParseHandle(context, &iterator->stream, name)))
    return mlirBytecodeFailure();
  if (mlirBytecodeFailed(
          mlirBytecodeParseHandle(context, &iterator->stream, value)))
    return mlirBytecodeEmitError(context, "can't parse value");
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseDictionaryAttr(MlirBytecodeDialectReader *reader,
                                              MlirBytecodeAttrHandle attrHandle) {
  uint64_t size;
  if (mlirBytecodeFailed(
    mlirBytecodeDialectReaderReadVarInt(reader, &size)))
    return mlirBytecodeFailure();
  MlirDictionaryHandleIterator iterator = {.stream = *reader->stream, .count = size};
  return mlirBytecodeCreateBuiltinDictionaryAttr(reader->context, attrHandle,
                                                 &iterator);
}

static MlirBytecodeStatus parseFileLineColLoc(MlirBytecodeDialectReader *reader,
                                              MlirBytecodeAttrHandle attrHandle) {
  uint64_t filename, line, col;
  if (mlirBytecodeFailed(mlirBytecodeDialectReaderReadVarInt(reader, &filename)) ||
      mlirBytecodeFailed(mlirBytecodeDialectReaderReadVarInt(reader, &line)) ||
      mlirBytecodeFailed(mlirBytecodeDialectReaderReadVarInt(reader, &col)))
    return mlirBytecodeFailure();
  return mlirBytecodeCreateBuiltinFileLineColLoc(
      reader->context, attrHandle, (MlirBytecodeAttrHandle){.id = filename}, line,
      col);
}

static MlirBytecodeStatus parseStringAttr(MlirBytecodeDialectReader *reader,
                                              MlirBytecodeAttrHandle attrHandle) {
  MlirBytecodeBytesRef bytes;
  if (mlirBytecodeFailed(
    mlirBytecodeDialectReaderReadString(reader,&bytes)))
    return mlirBytecodeFailure();
  return mlirBytecodeCreateBuiltinStringAttr(reader->context, attrHandle, bytes);
}

static MlirBytecodeStatus
parseStringAttrWithType(MlirBytecodeDialectReader *reader,
                             MlirBytecodeAttrHandle attrHandle
) {
  MlirBytecodeTypeHandle type;
  MlirBytecodeBytesRef value;
  if (mlirBytecodeSucceeded(
    mlirBytecodeDialectReaderReadType(reader, &type)) &&
      mlirBytecodeSucceeded(
        mlirBytecodeDialectReaderReadString(reader, &value)))
    return mlirBytecodeCreateBuiltinStringAttrWithType(
        reader->context, attrHandle, value, type);
  return mlirBytecodeEmitError(reader->context, "invalid StringAttrWithType");
}

static MlirBytecodeStatus parseTypeAttr(MlirBytecodeDialectReader *reader,
                             MlirBytecodeAttrHandle attrHandle) {
  MlirBytecodeTypeHandle value;
  if (mlirBytecodeSucceeded(
    mlirBytecodeDialectReaderReadType(reader, &value)))
    return mlirBytecodeCreateBuiltinTypeAttr(reader->context, attrHandle, value);

  return mlirBytecodeEmitError(reader->context, "invalid TypeAttr");
}

MlirBytecodeStatus
mlirBytecodeParseBuiltinAttr(MlirBytecodeDialectReader *reader,
                             MlirBytecodeAttrHandle attrHandle) {
  uint64_t kind;
  if (mlirBytecodeFailed(mlirBytecodeDialectReaderReadVarInt(reader, &kind)))
    return mlirBytecodeFailure();
  mlirBytecodeEmitDebug("builtin attr kind %d", (int)kind);

  switch (kind) {
  case kBuiltinAttrIntegerAttrCode:
    return parseIntegerAttr(reader, attrHandle);
  case kBuiltinAttrTypeAttrCode:
    return parseTypeAttr(reader, attrHandle);
  case kBuiltinAttrDictionaryAttrCode:
    return parseDictionaryAttr(reader, attrHandle);
  case kBuiltinAttrStringAttrCode:
    return parseStringAttr(reader, attrHandle);
  case kBuiltinAttrStringAttrWithTypeCode:
    return parseStringAttrWithType(reader, attrHandle);
  case kBuiltinAttrUnknownLocCode:
    return mlirBytecodeCreateBuiltinUnknownLoc(reader, attrHandle);
  case kBuiltinAttrFileLineColLocCode:
    return parseFileLineColLoc(reader, attrHandle);
  case kBuiltinAttrUnitAttrCode:
    return mlirBytecodeCreateBuiltinUnitAttr(reader->context, attrHandle);
  default:
    mlirBytecodeEmitDebug("missing parsing for builtin attr %" PRIu64 "\n",
                          kind);
    return mlirBytecodeUnhandled();
  }
  return mlirBytecodeSuccess();
}

static MlirBytecodeStatus parseFloat32Type(MlirBytecodeDialectReader* reader,
                                           MlirBytecodeTypeHandle typeHandle) {
  return mlirBytecodeCreateBuiltinFloat32Type(reader->context, typeHandle);
}

static MlirBytecodeStatus parseIndexType(MlirBytecodeDialectReader* reader,
                                           MlirBytecodeTypeHandle typeHandle) {
  return mlirBytecodeCreateBuiltinIndexType(reader->context, typeHandle);
}

static MlirBytecodeStatus parseIntegerType(MlirBytecodeDialectReader* reader,
                                           MlirBytecodeTypeHandle typeHandle) {
  uint64_t _widthAndSignedness, signedness, width;
  if (mlirBytecodeSucceeded(mlirBytecodeDialectReaderReadVarInt(reader,
                                                    &_widthAndSignedness)) &&
      mlirBytecodeSucceeded(
          ((signedness = _widthAndSignedness & 0x3), mlirBytecodeSuccess())) &&
      mlirBytecodeSucceeded(
          ((width = _widthAndSignedness >> 2), mlirBytecodeSuccess()))) {
    return mlirBytecodeCreateBuiltinIntegerType(
        reader->context, typeHandle, (MlirBuiltinSignednessSemantics)signedness,
        width);
  }
  return mlirBytecodeEmitError(reader->context, "invalid IntegerType");
}

static MlirBytecodeStatus parseFunctionType(MlirBytecodeDialectReader* reader,
                                           MlirBytecodeTypeHandle typeHandle
) {
  // FIXME
  return mlirBytecodeCreateBuiltinFunctionType(reader->context, typeHandle);
}

static MlirBytecodeStatus
parseRankedTensorType(MlirBytecodeDialectReader* reader,
                                           MlirBytecodeTypeHandle typeHandle) {
  // FIXME
  return mlirBytecodeCreateBuiltinRankedTensorType(reader->context, typeHandle);
}

MlirBytecodeStatus
mlirBytecodeParseBuiltinType(MlirBytecodeDialectReader *reader,
                             MlirBytecodeTypeHandle typeHandle) {
  uint64_t kind;
  if (mlirBytecodeFailed(mlirBytecodeDialectReaderReadVarInt(reader, &kind)))
    return mlirBytecodeFailure();
  mlirBytecodeEmitDebug("builtin type kind %d", (int)kind);

  switch (kind) {
  case kBuiltinTypeIntegerTypeCode:
    return parseIntegerType(reader, typeHandle);
  case kBuiltinTypeRankedTensorTypeCode:
    return parseRankedTensorType(reader, typeHandle);
  case kBuiltinTypeIndexTypeCode:
    return parseIndexType(reader, typeHandle);
  case kBuiltinTypeFloat32TypeCode:
    return parseFloat32Type(reader, typeHandle);
  case kBuiltinTypeFunctionTypeCode:
    return parseFunctionType(reader, typeHandle);
  default:
    mlirBytecodeEmitDebug("missing parsing for builtin type %" PRIu64 "\n",
                          kind);
    return mlirBytecodeUnhandled();
  }

  return mlirBytecodeSuccess();
}
