# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *


def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  return f


# CHECK-LABEL: TEST: testLogicalResultExplicitChecks
# CHECK: Success == LogicalResult.success: True
# CHECK: Success == LogicalResult.failure: False
# CHECK: Failure == LogicalResult.success: False
# CHECK: Failure == LogicalResult.failure: True
@run
def testLogicalResultExplicitChecks():
  success = LogicalResult.success
  failure = LogicalResult.failure

  print("Success == LogicalResult.success:", success == LogicalResult.success)
  print("Success == LogicalResult.failure:", success == LogicalResult.failure)
  print("Failure == LogicalResult.success:", failure == LogicalResult.success)
  print("Failure == LogicalResult.failure:", failure == LogicalResult.failure)
