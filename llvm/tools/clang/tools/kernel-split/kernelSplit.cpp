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
#include <iterator>

using namespace llvm;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
    private:
        Rewriter &rewriter;
        ASTContext &context;
        std::stringstream targetClauses;
        std::stringstream mapClauses;
        bool inTarget = false;
        bool firstParallel = false;
        bool findEnd = false;
        SourceLocation end;
        int numParallel;
        int curParallel;

    public:
        MyASTVisitor(Rewriter &R, ASTContext &C) : rewriter(R) , context(C) {}

        void removeRightBrkt(SourceLocation loc) {
            int lefts = 0;
            SourceLocation tokLoc;
            Token currTok;
            while(true) {
                tokLoc = Lexer::GetBeginningOfToken(loc,rewriter.getSourceMgr(), rewriter.getLangOpts());
                bool noToken = Lexer::getRawToken(tokLoc, currTok, rewriter.getSourceMgr(), rewriter.getLangOpts()); 
                loc = loc.getLocWithOffset(1);
                if(!noToken) {
                    if (currTok.getKind() == tok::l_brace)
                        lefts++;
                    else if (currTok.getKind() == tok::r_brace) {
                        if(lefts != 0)
                            lefts--;
                        else
                            break;
                    }
                }
            }
            rewriter.RemoveText(SourceRange(tokLoc,loc));
            return;
        }
        
        void addRightBrkt(Stmt *s) {
            end = s->getLocEnd();
            findEnd = true;
            RecursiveASTVisitor<MyASTVisitor>::TraverseStmt(s);
            findEnd = false;
            rewriter.InsertText(end.getLocWithOffset(1), "}");
            return;
        }
 
        bool VisitStmt(Stmt *s) {
            if(findEnd and rewriter.getSourceMgr().isBeforeInTranslationUnit(end,s->getLocEnd()))
                end = s->getLocEnd();
            return true;
        }
        
        bool TraverseOMPTargetTeamsDirective(OMPTargetTeamsDirective *d) {
            RecursiveASTVisitor<MyASTVisitor>::TraverseOMPTargetTeamsDirective(d);
            inTarget = false;
            firstParallel = false;
            return true;
        }

        bool VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *d) {
            numParallel = 0;
            curParallel = 0;
            targetClauses.str("");
            mapClauses.str("");
            for(int i = 0; i < (int) d->getNumClauses(); ++i) {
                OMPClause *c = d->clauses()[i];
                if(!c->isImplicit()) {
                        if(c->getClauseKind() != OMPC_map)
                            targetClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),c->getLocEnd()));
                        else
                            mapClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),c->getLocEnd()));
                }
            }
            SourceLocation leftBrkt = Lexer::findLocationAfterToken(d->getLocEnd(),tok::l_brace,rewriter.getSourceMgr(),rewriter.getLangOpts(),true);
            if(leftBrkt.isValid()) {
                rewriter.RemoveText(SourceRange(d->getLocStart().getLocWithOffset(-8),leftBrkt));
                removeRightBrkt(leftBrkt);
            }
            else
                rewriter.RemoveText(SourceRange(d->getLocStart().getLocWithOffset(-8),leftBrkt));
            inTarget = true;
            return true;
        }

        bool VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *d) {
            if(inTarget)
            {
               if(!firstParallel) { 
                   firstParallel = true;
                   rewriter.InsertText(d->getLocStart().getLocWithOffset(-8), "#pragma omp target data " + mapClauses.str() + "{\n", true, true);
                   const Stmt *parent = context.getParents(*d)[0].get<Stmt>();
                   for(ConstStmtIterator iter = parent->child_begin(); iter!=parent->child_end();iter++) {
                       if(isa<OMPDistributeParallelForDirective>(*iter))
                           numParallel++;
                   }
               }
               curParallel++;
               if(numParallel == curParallel)
                   addRightBrkt(d);
                rewriter.InsertText(d->getLocStart().getLocWithOffset(4), "target teams ");
                rewriter.InsertText(d->getLocEnd()," " + targetClauses.str());
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
