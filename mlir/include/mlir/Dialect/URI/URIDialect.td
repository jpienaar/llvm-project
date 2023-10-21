//===- URIDialect.td - URI dialect defs ---------------------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef URI_DIALECT
#define URI_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

def URI_Dialect : Dialect {
  let name = "uri";
  let cppNamespace = "::mlir::uri";

  let description = [{
    URI dialect is intended to hold attributes for referring to external file
    attributes.
  }];

  let useDefaultAttributePrinterParser = 1;
}

class URI_Attr<string name, list<Trait> traits = []>
    : AttrDef<URI_Dialect, name, traits>;

def URI_ExternalElementsRef : URI_Attr<"ExternalElementsRef", [ElementsAttrInterface, TypedAttrInterface]> {
  let summary = "A reference to an external resource";
  let description = [{
    Syntax:

    ```
    external-elements-ref ::= `<` handle (`::` reference)* `>` `:` type
    ```

    where the handle is to a string resource of the form

    ```
    dialect ({metadata})? `:` uri
    ```

    in the URI dialect.
  }];
  let parameters = (ins
    ResourceHandleParameter<"::mlir::uri::URIResourceHandle">:$handle,
    ArrayRefParameter<"::mlir::FlatSymbolRefAttr", "">:$nestedReferences,
    AttributeSelfTypeParameter<"", "ShapedType">:$type
  );
  // TODO: name? This is shaped too, which many may not be.
  let mnemonic = "elem";
  let assemblyFormat = "`<` $handle $nestedReferences `>`";
}

#endif // URI_DIALECT
