set(LLVM_TARGET_DEFINITIONS SimpleDepTypeDialect.td)
mlir_tablegen(SimpleDepTypeAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(SimpleDepTypeAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(SimpleDepTypeAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(SimpleDepTypeAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(SimpleDepType.h.inc -gen-op-decls)
mlir_tablegen(SimpleDepType.cpp.inc -gen-op-defs)
mlir_tablegen(SimpleDepTypeDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SimpleDepTypeDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRSimpleDepTypeDialectGen)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_example(mlir-simple-dep-type-opt
  ti-opt.cpp
  DEPENDS
    MLIRSimpleDepTypeDialectGen
  )
target_link_libraries(mlir-simple-dep-type-opt
  PRIVATE
    MLIRArithmetic
    MLIRIR
    MLIROptLib
    MLIRParser
    MLIRShape
    MLIRStandard
    MLIRTensor
    MLIRTransforms
  )
mlir_check_all_link_libraries(mlir-simple-dep-type-opt)
