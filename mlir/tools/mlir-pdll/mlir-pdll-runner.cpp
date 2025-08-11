//===- mlir-pdll-runner.cpp - PDLL Interpreter Runner -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A tool to run PDLL patterns against an MLIR file.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::pdll;

// A helper function to check if an operation is a native PDL op. This is
// used to skip patterns that contain native constraints or rewrites, as they
// are not yet supported by the PDLL runner.
static WalkResult checkForNativePDLOps(Operation *op) {
  if (isa<pdl::ApplyNativeConstraintOp, pdl::ApplyNativeRewriteOp>(op))
    return WalkResult::interrupt();
  return WalkResult::advance();
}

/// Populates the given pattern set with patterns from the given PDL module.
static void populatePatternsFromPDL(ModuleOp pdlModule,
                                    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  for (pdl::PatternOp patternOp :
       llvm::to_vector(pdlModule.getOps<pdl::PatternOp>())) {
    OwningOpRef<ModuleOp> patternModule =
        ModuleOp::create(UnknownLoc::get(context));
    // For now, skip patterns that contain native constraints or rewrites.
    if (patternOp->walk(checkForNativePDLOps).wasInterrupted()) {
      mlir::emitError(patternOp->getLoc())
          << "Skipping, PDLL pattern contains native constraints or rewrites";
      continue;
    }
    patternOp->remove();
    patternModule->push_back(patternOp);
    patterns.add<PDLPatternModule>(std::move(patternModule));
  }
}

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputPdllFilename(
      llvm::cl::Positional, llvm::cl::desc("<input-pdll-file>"),
      llvm::cl::init("-"), llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> inputMlirFilename(
      "mlir-file", llvm::cl::desc("Input MLIR file to transform"),
      llvm::cl::value_desc("filename"), llvm::cl::Required);

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "PDLL-based pattern driver");

  // Set up the input pdll file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> pdllFile =
      openInputFile(inputPdllFilename, &errorMessage);
  if (!pdllFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Parse the PDLL file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(pdllFile), SMLoc());

  MLIRContext context;
  context.loadDialect<pdl::PDLDialect, pdl_interp::PDLInterpDialect,
                      test::TestDialect>();
  DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  ods::Context odsContext;
  ast::Context astContext(odsContext);
  FailureOr<ast::Module *> astModule =
      parsePDLLAST(astContext, sourceMgr, /*enableDocumentation=*/false);
  if (failed(astModule)) {
    llvm::errs() << "Failed to parse PDLL file\n";
    return 1;
  }

  // Generate PDL MLIR from the AST.
  OwningOpRef<ModuleOp> pdlModule =
      codegenPDLLToMLIR(&context, astContext, sourceMgr, **astModule);
  if (!pdlModule) {
    llvm::errs() << "Failed to generate PDL MLIR\n";
    return 1;
  }

  // Set up the input mlir file.
  std::unique_ptr<llvm::MemoryBuffer> mlirFile =
      openInputFile(inputMlirFilename, &errorMessage);
  if (!mlirFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Parse the input MLIR file.
  llvm::SourceMgr mlirSourceMgr;
  mlirSourceMgr.AddNewSourceBuffer(std::move(mlirFile), SMLoc());
  OwningOpRef<ModuleOp> mlirModule =
      parseSourceFile<ModuleOp>(mlirSourceMgr, &context);
  if (!mlirModule) {
    llvm::errs() << "Failed to parse MLIR file\n";
    return 1;
  }

  // Apply the patterns.
  RewritePatternSet patterns(&context);
  populatePatternsFromPDL(*pdlModule, patterns);
  if (failed(applyPatternsGreedily(*mlirModule, std::move(patterns)))) {
    llvm::errs() << "Failed to apply patterns\n";
    return 1;
  }

  // Write the output.
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  mlirModule->print(outputFile->os());
  outputFile->keep();

  return 0;
}
