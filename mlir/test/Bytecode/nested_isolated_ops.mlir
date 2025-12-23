// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Test bytecode roundtrip for nested isolated-from-above operations.
// This catches issues with:
// - Region stack finalization  
// - Splice condition detection
// - ValueScope management across nested scopes

// CHECK-LABEL: module @outer_module
module @outer_module {
  // CHECK: module @inner_module_1
  module @inner_module_1 {
    // CHECK: func.func @nested_func_1
    func.func @nested_func_1(%arg0: i32) -> i32 {
      %c1 = arith.constant 1 : i32
      %result = arith.addi %arg0, %c1 : i32
      return %result : i32
    }
  }

  // CHECK: module @inner_module_2
  module @inner_module_2 {
    // CHECK: func.func @nested_func_2
    func.func @nested_func_2(%arg0: f32) -> f32 {
      %c = arith.constant 2.0 : f32
      %result = arith.mulf %arg0, %c : f32
      return %result : f32
    }
  }

  // CHECK: func.func @top_level_func
  func.func @top_level_func() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
}
