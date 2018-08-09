#include "clang/AST/AST.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"

#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include <string>
#include <sstream>
#include <iostream>
#include <queue>
#include <forward_list>

#define PRAGMA_SIZE 8
#define DEFAULT_THREAD_LIMIT 128
#define DEFAULT_NUM_TEAMS 112
#define MAX_BLOCKS 448

using namespace llvm;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

struct Variable {
    std::string name;
    SourceLocation loc;
    OpenMPMapClauseKind mapKind;
    int size = 0;
    bool alreadyMapped = false;
};

struct Iterator {
    std::string name;
    SourceLocation loc;
    int maxSize = 0;
};
