// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Test bytecode roundtrip for boolean attributes.
// This catches issues with byte vs varint encoding for booleans.

// CHECK-LABEL: @boolean_attrs_roundtrip
module @boolean_attrs_roundtrip {
  // CHECK: func.func @bool_attrs()
  // CHECK-SAME: attributes {test.flag_false = false, test.flag_true = true}
  func.func @bool_attrs() attributes {
    test.flag_true = true,
    test.flag_false = false
  } {
    return
  }

  // CHECK: func.func @unit_attr()
  // CHECK-SAME: attributes {test.unit}
  func.func @unit_attr() attributes {test.unit} {
    return
  }
}
