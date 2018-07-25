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
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <iterator>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
    private:
        Rewriter &rewriter;
        ASTContext &context;
        SourceLocation targetRange;
        SourceLocation distributeRange;

    public:
        MyASTVisitor(Rewriter &R, ASTContext &C) : rewriter(R) , context(C) {}
        
        bool VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *d)
        {
            bool clauses = false;
            for(int i = 0; i < (int) d->getNumClauses(); ++i)
            {
                OMPClause *c = d->clauses()[i];
                if(!c->isImplicit())
                {
                    clauses = true;
                    if(c->getClauseKind() != OMPC_map)
                        rewriter.RemoveText(SourceRange(c->getLocStart(),c->getLocEnd()));
                }
            }

            if(clauses)
                rewriter.ReplaceText(SourceRange(d->getLocStart(),d->clauses()[0]->getLocStart().getLocWithOffset(-1)), "omp target data");
            else
                rewriter.ReplaceText(SourceRange(d->getLocStart(),d->getLocEnd()),"omp target data");


            const Stmt *parent = context.getParents(*d)[0].get<Stmt>();
            bool next = false;
            for(ConstStmtIterator iter = parent->child_begin(); iter!=parent->child_end(); iter++)
            {
                if(next)
                {
                    targetRange = iter->getLocStart().getLocWithOffset(-1);
                    break;
                }     
                if(iter->getLocStart() == d->getLocStart())
                    next = true;    
            }

            return true;
        }

        bool VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *d)
        {
            if(rewriter.getSourceMgr().isBeforeInTranslationUnit(d->getLocStart(),targetRange))
            {
                rewriter.InsertText(d->getLocStart().getLocWithOffset(4), "target teams ");
                rewriter.InsertText(d->getLocEnd(), " thread_limit(128)");

                const Stmt *parent = context.getParents(*d)[0].get<Stmt>();
                bool next = false;
                for(ConstStmtIterator iter = parent->child_begin(); iter!=parent->child_end(); iter++)
                {
                    std::cout << iter->getStmtClassName() << std::endl;
                    if(next)
                    {
                        distributeRange = iter->getLocStart().getLocWithOffset(-1);
                        break;
                    }     
                    if(iter->getLocStart() == d->getLocStart())
                        next = true;    
                }
            }
            return true;
        }
};

class MyASTConsumer : public ASTConsumer {
    private:
        MyASTVisitor visitor;

    public:
        MyASTConsumer(Rewriter &R, ASTContext &C) : visitor(R, C) {}
        
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
            return llvm::make_unique<MyASTConsumer>(rewriter, CI.getASTContext());
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
