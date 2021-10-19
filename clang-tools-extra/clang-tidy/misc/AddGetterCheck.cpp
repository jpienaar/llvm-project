//===--- AddGetterCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddGetterCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void AddGetterCheck::registerMatchers(MatchFinder *finder) {
#define WITH_GETTER(X) X, X "Attr", X "AttrName", X "Mutable",
  finder->addMatcher(cxxMemberCallExpr(callee(cxxMethodDecl(hasAnyName(
#include "/tmp/match.inc"
    "SomeLongNameThatShouldntMatchToJustAvoidTheNeedForMeToMakeSmarterMacroConcat"
                                           ))))
                         .bind("call"),
                     this);
}

void AddGetterCheck::check(const MatchFinder::MatchResult &result) {
  const auto *call = result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *callee = cast<MemberExpr>(call->getCallee());
  std::string name = callee->getMemberNameInfo().getName().getAsString();
  SourceRange range = callee->getSourceRange();
  llvm::StringRef src = Lexer::getSourceText(
      CharSourceRange::getCharRange(range), *result.SourceManager,
      result.Context->getLangOpts());
  // Heuristically assume set if void return.
  std::string prefix = call->getType()->isVoidType() ? "set" : "get";
  std::string dst =
      (src + prefix + llvm::convertToCamelFromSnakeCase(name, true)).str();
  diag(range.getBegin(), "call '%0' is missing prefix")
      << name << FixItHint::CreateReplacement(callee->getSourceRange(), dst);
}

} // namespace misc
} // namespace tidy
} // namespace clang
