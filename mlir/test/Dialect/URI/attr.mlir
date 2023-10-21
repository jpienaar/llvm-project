
module attributes { test.blob_ref = #uri.elem<library@foo> : tensor<*xi1>} {

func.func @foo() -> (tensor<10xf32>) {
  %0 = arith.constant #uri.elem<wright@f> : tensor<10xf32>
//  %1 = arith.constant luri<"luri.raw", byte_offset =  0, byte_size = 80> : tensor<10xf32>
  %1 = arith.constant #uri.elem<weight@g> : tensor<10xf32>
  %c = arith.constant dense<1.0e5> : tensor<10xf32>
  %mul = arith.mulf %0, %c : tensor<10xf32>
  // CHECK: arith.constant{{.*}}785.0708
  %add = arith.addf %1, %mul: tensor<10xf32>
  return %add : tensor<10xf32>
}

}

{-#
  dialect_resources: {
    uri: {
      library: "builtin:test-interpreter-external-source.mlir",
      weight: "builtin:luri.raw"
    }
  }
#-}
