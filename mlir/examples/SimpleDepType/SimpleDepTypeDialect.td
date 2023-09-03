//===-- SimpleDepTypeDialect.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SIMPLE_DEP_TYPE_DIALECT
#define SIMPLE_DEP_TYPE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SimpleDepType_Dialect : Dialect {
  let name = "simple_deptype";
  let cppNamespace = "::mlir::example::simple_deptype";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

// Attribute for capturing the symbol references.
// This should rather be an attribute interface than a concrete attribute.
def SimpleDepTypeTensorEncodingAttr : AttrDef<SimpleDepType_Dialect, "TensorEncoding",
         [ DeclareAttrInterfaceMethods<VerifiableTensorEncoding> ] > {
  let mnemonic = "symbolic_dims";
  let parameters = (ins ArrayRefParameter<"FlatSymbolRefAttr", "">:$symbolic);
  let assemblyFormat = "$symbolic";
  let genVerifyDecl = 1;
}

def SimpleDepType_AssociateOp : Op<SimpleDepType_Dialect, "associate", [Symbol]>,
    Arguments<(ins AnyType:$value, SymbolNameAttr:$sym_name)> {
  let assemblyFormat = "$value `with` $sym_name attr-dict-with-keyword `:` type($value)";
}

def SimpleDepType_AssociateTensorOp : Op<SimpleDepType_Dialect, "associate_tensor", [DeclareOpInterfaceMethods<InferTypeOpInterface>]>,
    Arguments<(ins AnyTensor:$value, Variadic<Index>:$dims, ArrayAttr:$names)>, Results<(outs AnyType:$res)> {
  let assemblyFormat = "$value `with` $dims `named` $names attr-dict `:` type($value)";
}

def SimpleDepType_DeassociateOp : Op<SimpleDepType_Dialect, "deassociate", [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]>,
    Arguments<(ins AnyType:$value)>, Results<(outs AnyType:$res)> {
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def SimpleDepType_WitnessSinkOp : Op<SimpleDepType_Dialect, "witness_sink", []>,
    Arguments<(ins Variadic<AnyType>:$values, SymbolNameAttr:$sym_name)> {
  let assemblyFormat = "$values `as` $sym_name attr-dict `:` type($values)";
}

#endif  //  SIMPLE_DEP_TYPE_DIALECT
