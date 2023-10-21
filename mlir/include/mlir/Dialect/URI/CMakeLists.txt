set(LLVM_TARGET_DEFINITIONS URIDialect.td)
mlir_tablegen(URIDialect.h.inc -gen-dialect-decls)
mlir_tablegen(URIDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRURIDialectIncGen)

set(LLVM_TARGET_DEFINITIONS URIDialect.td)
mlir_tablegen(URIDialectAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(URIDialectAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRURIDialectAttrDefsIncGen)

add_mlir_doc(URIAttributes URIAttributes Dialects/ -gen-attrdef-doc)
