// RUN: mlir-opt -allow-unregistered-dialect -emit-bytecode %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// Test bytecode roundtrip for operations with multiple regions.
// This catches issues with:
// - Section header reading (once per op vs once per region)
// - Region isolation handling

// CHECK-LABEL: @multi_region_roundtrip
module @multi_region_roundtrip {
  // CHECK: func.func @if_else
  func.func @if_else(%cond: i1) -> i32 {
    // CHECK: scf.if
    %result = scf.if %cond -> i32 {
      // CHECK: arith.constant 1
      %c1 = arith.constant 1 : i32
      scf.yield %c1 : i32
    } else {
      // CHECK: arith.constant 2
      %c2 = arith.constant 2 : i32
      scf.yield %c2 : i32
    }
    return %result : i32
  }

  // CHECK: func.func @while_loop
  func.func @while_loop(%init: i32) -> i32 {
    %c10 = arith.constant 10 : i32
    // CHECK: scf.while
    %result = scf.while (%arg = %init) : (i32) -> i32 {
      %cond = arith.cmpi slt, %arg, %c10 : i32
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg: i32):
      %c1 = arith.constant 1 : i32
      %next = arith.addi %arg, %c1 : i32
      scf.yield %next : i32
    }
    return %result : i32
  }
}
