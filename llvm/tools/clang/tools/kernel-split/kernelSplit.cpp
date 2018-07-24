#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"

#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
    private:
        Rewriter &rewriter;

    public:
        MyASTVisitor(Rewriter &R) : rewriter(R) {}
       
        bool VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *d)
        {
            
            SourceLocation firstClause = d->clauses()[0]->getLocStart();
            for(int i = 0; i < (int) d->getNumClauses(); ++i)
            {
                OMPClause *c = d->clauses()[i];
                if(c->getClauseKind() != OMPC_map)
                    rewriter.RemoveText(SourceRange(c->getLocStart(),c->getLocEnd()));
            }
            rewriter.ReplaceText(SourceRange(d->getLocStart(),firstClause.getLocWithOffset(-1)), "omp target data");
            return true;
        }

        bool VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *d)
        {
            rewriter.InsertText(d->getLocStart().getLocWithOffset(4), "target teams ");
            rewriter.InsertText(d->getLocEnd(), " thread_limit(128)");
            return true;
        }
};

class MyASTConsumer : public ASTConsumer {
    private:
        MyASTVisitor visitor;

    public:
        MyASTConsumer(Rewriter &R) : visitor(R) {}
        
        bool HandleTopLevelDecl(DeclGroupRef DR) override {
            for(DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b !=e; ++b) {
                visitor.TraverseDecl(*b);
            }
            return true;
        }
};

class MyFrontendAction : public ASTFrontendAction {
    private:
        Rewriter rewriter;

    public:
        MyFrontendAction() {}
        
        void EndSourceFileAction() override {
            SourceManager &SM = rewriter.getSourceMgr();
            llvm::errs() << "** EndSourceFileAction for: " << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";
            rewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
        }
        
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
            llvm::errs() << "** Creating AST consumer for: " << file << "\n";
            rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
            return llvm::make_unique<MyASTConsumer>(rewriter);
        }
};    
		
//Tool help menu options
static llvm::cl::OptionCategory toolCategory("my-tool options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nMore help text...\n");

//Parse options and apply tool to defined files
int main(int argc, const char **argv) {
    CommonOptionsParser op(argc, argv, toolCategory);
    ClangTool Tool(op.getCompilations(),op.getSourcePathList());
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
