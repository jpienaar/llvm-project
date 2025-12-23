// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Test bytecode roundtrip for empty regions.
// This catches issues with:
// - Empty region handling (returning success vs failure)
// - ValueScope push/pop for empty regions

// CHECK-LABEL: @empty_region_roundtrip
module @empty_region_roundtrip {
  // CHECK: llvm.mlir.global external @empty_global(0 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @empty_global(0 : i32) {addr_space = 0 : i32} : i32

  // CHECK: llvm.mlir.global internal @another_empty("hello") {addr_space = 0 : i32}
  llvm.mlir.global internal @another_empty("hello") {addr_space = 0 : i32} : !llvm.array<5 x i8>

  // CHECK: func.func @uses_globals
  func.func @uses_globals() -> i32 {
    %addr = llvm.mlir.addressof @empty_global : !llvm.ptr
    %val = llvm.load %addr : !llvm.ptr -> i32
    return %val : i32
  }
}
