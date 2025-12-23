// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Test bytecode roundtrip for scalable vector types.
// This catches issues with boolean encoding in type serialization.

// CHECK-LABEL: @scalable_vector_roundtrip
module @scalable_vector_roundtrip {
  // CHECK: func.func @scalable_vectors
  // CHECK-SAME: %[[ARG0:.*]]: vector<[4]xf32>
  // CHECK-SAME: %[[ARG1:.*]]: vector<[2]xi64>
  // CHECK-SAME: %[[ARG2:.*]]: vector<4x[2]xf64>
  func.func @scalable_vectors(
    %arg0: vector<[4]xf32>,
    %arg1: vector<[2]xi64>,
    %arg2: vector<4x[2]xf64>
  ) -> (vector<[4]xf32>, vector<[2]xi64>, vector<4x[2]xf64>) {
    // CHECK: return %[[ARG0]], %[[ARG1]], %[[ARG2]]
    return %arg0, %arg1, %arg2 : vector<[4]xf32>, vector<[2]xi64>, vector<4x[2]xf64>
  }

  // CHECK: func.func @mixed_scalable
  // CHECK-SAME: vector<[4]x[2]xf32>
  func.func @mixed_scalable(%arg: vector<[4]x[2]xf32>) -> vector<[4]x[2]xf32> {
    return %arg : vector<[4]x[2]xf32>
  }
}
