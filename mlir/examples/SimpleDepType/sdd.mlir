// RUN: -allow-unregistered-dialect

// Test/story showing simple dependent type example through multiple
// different passes.
//
// Different stories to explore
//   * --sccp --sdt-fold-associate
//   * --inline --sdt-canonicalize --sdt-fold-associate=merge-equal
//   * --sdt-to-shape --shape-to-sdt
//   * --sdt-to-shape=use-shape-fns --shape-to-sdt=use-shape=fns
//   * --sdt-fold-associate --sdt-fold-associate=merge-equal
//   * --sdt-shape-report (this one isn't really dependent type specific, but shows germ of the idea)
// All the above requires -allow-unregistered-dialect.

// Some type aliases to make more concise to write.

!tensor_x = type tensor<?x i17, #simple_deptype.symbolic_dims<@x>
!tensor_xz = type tensor<?x?x i17, #simple_deptype.symbolic_dims<@x, @z>
!tensor_x10 = type tensor<?x10x i17, #simple_deptype<"symbolic_dims @x">>
!tensor_y10 = type tensor<?x10x i17, #simple_deptype<"symbolic_dims @y">>

module attributes {shape.lib = [@shape_lib]} {

func private @f(%arg0: index) {
  %0 = "test.foo1"(%arg0) : (index) -> !tensor_xz
  simple_deptype.associate %arg0 with @x : index
  %c_0 = arith.constant 0 : index

  // One could also "obscure" that dims are equal.
  //  %1 = arith.addi %arg0, %c_0 : index
  simple_deptype.associate %arg0 with @y : index

  // At this point the regular type system would not
  // allow
  //   "arith.addi"(%0, %0) : (!tensor_x10, !tensor_y10) -> !tensor_x10)
  // as the two types are not statically equal.

  %foo = "test.foo2"(%arg0) : (index) -> !tensor_y10
  // %foo_1 = type_intro(%foo) : tensor<?xf32> -> tensor<@dim x f32>
  //
  // %dim0_size = (size.get_shape(%foo))[0]
  // associate %dim0_size with @dim
  %foo2 = call @h(%foo_1, %arg0) : (!tensor_y10, index) -> (!tensor_y10)

  %addi_0 = arith.addi %0, %0 : !tensor_xz
  %addi_foo = arith.addi %foo, %foo : !tensor_y10

  %c = arith.constant 10 : index
  simple_deptype.associate %c with @z : index
  return
}

func @add(%arg0: !tensor_x, !tensor_x) -> !tensor_x {
   call @sub(...)
}

func @add_x(%arg0: !tensor_x, !tensor_x) -> !tensor_x {
  
}

func @sub_with_same(%arg0: tensor<?xi17>, !tensor<?xi17>, %n : index) -> tensor<?xi17> {
  %0 = simple_deptype.associate_tensor %arg0 with %n named [@sub_int] : tensor<?x10xi17>
  %1 = simple_deptype.associate_tensor %arg1 with %n named [@sub_int] : tensor<?x10xi17>

  return %arg0 : !tensor_y10

}

func private @sub_with_same(%arg0: tensor<?xi17>, !tensor<?xi17>) -> tensor<?xi17> {
  assert(shape.size_eq(shape.shape_of %arg0 [0], %n));
  assert(shape.shape_eq(shape.shape_of %arg0, shape.shape_of %arg1));

  return %arg0 : !tensor_y10

}

func @g() {
  %c = arith.constant 10 : index
  call @f(%c) : (index) -> ()
  return
}

func private @h(%arg0 : !tensor_y10, %arg1 : index) -> !tensor_y10 {
  // The dependent symbols is bound inside each function.  Now this would be
  // "weird" if we could have that !tensor_y10 is not really a fixed alias
  // across all functions - it could mean something different here than in
  // another. And would be unfortunate for optimizations such as inlining -
  // post inlining a valid function would now fail verification as there are 2
  // associations for the same symbol with unequal bound value.
  //
  // There are two options here 1) ensure that symbol names are unique per
  // module, 2) treat all symbols as global. Option 1 makes return values more
  // cumbersome (one needs to remove the mapping before returning and
  // reassociate in caller), so following option 2 and relying on verification.
  //
  // There are other options of course :) Including requiring only that symbol
  // is consistent wrt callgraph, and so the value of the symbol may change
  // across the program. But being conservative here.
  //
  // TODO: Add some convention for function local symbols (probably just some
  // syntactic sugar).

  simple_deptype.associate %arg1 with @y : index

  return %arg0 : !tensor_y10
}

// If we want to invoke a function from multiple different call sites (say we
// only care about the dimensions but not a fixed symbol - or that we want to
// specialize for specific symbol later/progressively specialize) we can't have
// it in function argument type nor return type.
//
// But we could associate & deassociate.
func @r(%arg0 : tensor<?x10xi17>, %arg1 : index) -> tensor<?x10xi17> {
  %0 = simple_deptype.associate_tensor %arg0 with %arg1 named [@x] : tensor<?x10xi17>
  %1 = simple_deptype.deassociate %0 : tensor<?x10xi17, #simple_deptype<"symbolic_dims @x">>
  return %1 : tensor<?x10xi17>
}

shape.function_library @shape_lib {
  builtin.func @shape_common(%d : !shape.size) -> !shape.shape {
    %0 = shape.const_size 10
    %1 = shape.from_extents %0, %d : !shape.size, !shape.size
    return %1 : !shape.shape
  }
} mapping {
  test.foo1 = @shape_common,
  test.foo2 = @shape_common
}
}
