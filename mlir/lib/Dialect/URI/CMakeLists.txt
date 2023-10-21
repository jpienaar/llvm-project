add_mlir_dialect_library(MLIRURIDialect
  URIDialect.cpp

  DEPENDS
  MLIRURIDialectIncGen
  MLIRURIDialectAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
