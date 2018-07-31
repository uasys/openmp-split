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

#define PRAGMA_SIZE 8
#define DEFAULT_THREAD_LIMIT 128
#define DEFAULT_NUM_TEAMS 112
#define MAX_BLOCKS 448

using namespace llvm;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
    private:
        Rewriter &rewriter;
        ASTContext &context;

        //Variables used for the basic tree traversal and application of the splitting
        std::stringstream targetClauses;
        std::stringstream mapClauses;
        bool inTarget = false;
        bool inTargetData = false;
        bool hasParallel = false;
        bool firstParallel = false;
        bool findEnd = false;
        SourceLocation endLoc;
        int numParallel;
        int curParallel;
        bool findParallelDepth = false;
        int stmtDepth;
        int parallelDepth;

        //Variables used for the application of custom grid geometry
        bool customGeo = true;
        bool searchFor = false;
        int totalParallelism;
        int loopsToCheck;
        int threadLimit = DEFAULT_THREAD_LIMIT;
        
    public:
        MyASTVisitor(Rewriter &R, ASTContext &C) : rewriter(R) , context(C) {}

        //Removes the right bracket corresponding to a left bracket found right before loc
        void removeRightBrkt(SourceLocation loc) {
            int lefts = 0;
            SourceLocation tokLoc;
            Token currTok;

            //Iterates through the original source code tokens until a corresponding right parantheses is found
            while(lefts > -1) {
                tokLoc = Lexer::GetBeginningOfToken(loc,rewriter.getSourceMgr(), rewriter.getLangOpts());
                bool noToken = Lexer::getRawToken(tokLoc, currTok, rewriter.getSourceMgr(), rewriter.getLangOpts()); 
                loc = loc.getLocWithOffset(1);
                if(!noToken) {
                    if (currTok.getKind() == tok::l_brace)
                        lefts++;
                    else if (currTok.getKind() == tok::r_brace)
                        lefts--;
                }
            }

            //Removes that right parantheses from the source code
            rewriter.RemoveText(SourceRange(tokLoc,loc));

            return;
        }
        
        //Adds a right bracket at the end of the given subtree by finding the furthers end location within and adding a right bracket there
        void addRightBrkt(Stmt *s) {
            endLoc = s->getLocEnd();
            findEnd = true;
            RecursiveASTVisitor<MyASTVisitor>::TraverseStmt(s);
            findEnd = false;
            rewriter.InsertText(endLoc.getLocWithOffset(1), "}");
            return;
        }

        //Finds the location of the right parantheses of a given statement starting at loc
        SourceLocation findRightParenth(SourceLocation loc) {
            int lefts = 0;
            SourceLocation tokLoc;
            Token currTok;
            bool within = false;

            //Iterates through the original source code tokens until a corresponding right parantheses is found
            while(lefts > 0 or !within) {
                tokLoc = Lexer::GetBeginningOfToken(loc,rewriter.getSourceMgr(), rewriter.getLangOpts());
                bool noToken = Lexer::getRawToken(tokLoc, currTok, rewriter.getSourceMgr(), rewriter.getLangOpts());
                loc = loc.getLocWithOffset(1);
                if(!noToken) {
                    if (currTok.getKind() == tok::l_paren) {
                        lefts++;
                        within = true;
                    }
                    else if (currTok.getKind() == tok::r_paren)
                        lefts--;
                }
            }
            return loc;
        }

        //Function called by the visitor upon finding a stmt
        bool VisitStmt(Stmt *s) {

            //Used when calculating the end point of a subtree, compares the current statements end location to the furthest one calculated
            if(findEnd and rewriter.getSourceMgr().isBeforeInTranslationUnit(endLoc,s->getLocEnd()))
                endLoc = s->getLocEnd();

            return true;
        }
        
        //Function called by the visitor before traversing down an OpenMP "target teams" directive
        bool TraverseOMPTargetTeamsDirective(OMPTargetTeamsDirective *d) {

            //Performs a traversal of the subtree to calculate the highest depth parallel directive
            stmtDepth = 0;
            parallelDepth = -1;
            findParallelDepth = true;
            RecursiveASTVisitor<MyASTVisitor>::TraverseOMPTargetTeamsDirective(d);
            findParallelDepth = false;

            //Performs the actual full traversal of the subtree with the previously calculated information
            RecursiveASTVisitor<MyASTVisitor>::TraverseOMPTargetTeamsDirective(d);
            threadLimit = DEFAULT_THREAD_LIMIT;
            inTarget = false;
            firstParallel = false;
            hasParallel = false;

            return true;
        }
        
        //Function called by the visitor before traversing down an OpenMP "target data" directive 
        bool TraverseOMPTargetDataDirective(OMPTargetDataDirective *d) {
            inTargetData = true;
            RecursiveASTVisitor<MyASTVisitor>::TraverseOMPTargetDataDirective(d);
            inTargetData = false;
            return true;
        }

        //Function called by the visitor before traversing down a compound stmt (for, while, if, do, switch statments generally)
        bool TraverseCompoundStmt(CompoundStmt *s) {
            ++stmtDepth;
            RecursiveASTVisitor<MyASTVisitor>::TraverseCompoundStmt(s);
            --stmtDepth;
            return true;
        }

        //Function called by the visitor upon finding an OpenMP "target teams" directive
        bool VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *d) {

            //Prevents changes when just searching for parallel depth
            if(findParallelDepth)
                return true;
            
            //Prevents changes to a target region that is only serial but is in a target data region
            if(!hasParallel and inTargetData)
                return true;            

            //Resets variables containing target region information
            inTarget = true;
            customGeo = true;
            numParallel = 0;
            curParallel = 0;
            targetClauses.str("");
            mapClauses.str("");

            //Finds all clauses for the given directive and saves then for later use
            for (int i = 0; i < (int) d->getNumClauses(); ++i) {
                OMPClause *c = d->clauses()[i];
                if (!c->isImplicit()) {
                        SourceLocation end = findRightParenth(c->getLocStart());
                        if (c->getClauseKind() == OMPC_thread_limit) {
                            const OMPThreadLimitClause *co = d->getSingleClause<OMPThreadLimitClause>();
                            Expr *threadL = co->getThreadLimit();
                            APSInt num;
                            if (threadL->EvaluateAsInt(num,context))
                                threadLimit = num.getExtValue();
                            else
                                customGeo = false;
                            targetClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),end)) << " ";
                        }
                        else if (c->getClauseKind() == OMPC_num_teams) {
                            customGeo = false;
                            targetClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),end)) << " ";
                        }
                        else if (c->getClauseKind() == OMPC_map)
                            mapClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),end)) << " ";
                        else 
                            targetClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),end)) << " ";
                }
            }

            //Removes the original target teams statement and any related brackets in the original source code 
            SourceLocation leftBrkt = Lexer::findLocationAfterToken(d->getLocEnd(),tok::l_brace,rewriter.getSourceMgr(),rewriter.getLangOpts(),true);
            if(leftBrkt.isValid()) {
                rewriter.RemoveText(SourceRange(d->getLocStart().getLocWithOffset(-PRAGMA_SIZE),leftBrkt));
                removeRightBrkt(leftBrkt);
            }
            else
                rewriter.RemoveText(SourceRange(d->getLocStart().getLocWithOffset(-PRAGMA_SIZE),d->getLocEnd()));

            return true;
        }

        //Function called by the visitor upon finding an OpenMP "distribute parallel for" directive
        bool VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *d) {
            
            hasParallel = true;
 
            //Compares the given directives depth to the highest one
            if (findParallelDepth and (stmtDepth < parallelDepth or parallelDepth == -1))
                parallelDepth = stmtDepth;

            //Handles the transformation of the directive that it at the highest depth
            else if (inTarget and !findEnd and !searchFor and stmtDepth == parallelDepth) {
                curParallel++;
               
                //Calculates the number of parallel regions at the given depth and handles all serial regions
                if (!firstParallel) {
                    firstParallel = true;
                    
                    //Adds an encompassing target data region starting at the first parallel region
                    if (!inTargetData)
                        rewriter.InsertText(d->getLocStart().getLocWithOffset(-8), "#pragma omp target data "+mapClauses.str()+"\n{\n", true, true);
                    
                    //Finds all other stmts at the same depth as the directive and counts the number of parallel regions and their positions
                    const Stmt *parent = context.getParents(*d)[0].get<Stmt>();
                    int i = 0, fp = -1, lp = -1;
                    for (ConstStmtIterator iter = parent->child_begin(); iter!=parent->child_end(); ++iter, ++i) {
                        if (isa<OMPDistributeParallelForDirective>(*iter)) {
                            numParallel++;
                            if (fp == -1)
                                fp = i;
                            lp = i;
                        }
                    }

                    //Using the previous information, blocks of all serial inbetween parallel ones into their own seperate target regions 
                    bool prevPrl = false;
                    bool prevSrl = false;
                    i = 0;
                    for(ConstStmtIterator iter = parent->child_begin(); iter!=parent->child_end(); ++iter, ++i) {  
                        if(isa<OMPDistributeParallelForDirective>(*iter)) {
                            if(prevSrl and fp != i)
                                rewriter.InsertText(iter->getLocStart().getLocWithOffset(-8), "}\n", true, true);
                            prevPrl = true;
                            prevSrl = false;
                        }
                        else {
                            if(prevPrl and lp > i)
                                rewriter.InsertText(iter->getLocStart(), "#pragma omp target teams "+targetClauses.str()+"\n{\n", true, true);
                            prevPrl = false;
                            prevSrl = true;
                        }
                    }                  
                }

                //Adds a closing right bracket after the last parallel region
                if(numParallel == curParallel and !inTargetData)
                    addRightBrkt(d);

                //Converts the directive into its own target region
                rewriter.InsertText(d->getLocStart().getLocWithOffset(4), "target teams ");
                std::stringstream clauses;
                if(inTargetData)
                    clauses << mapClauses.str();
                clauses << targetClauses.str();
                if(customGeo)
                    clauses << "num_teams(" << calculateCustomGeo(d) << ")";
                rewriter.InsertText(d->getLocEnd()," "+clauses.str());
            }
            return true;
        }

        int calculateCustomGeo(OMPDistributeParallelForDirective *d) {

            //Checks for collapse
            loopsToCheck = 1;
            const OMPCollapseClause *c = d->getSingleClause<OMPCollapseClause>();
            if(c) {
                Expr *collapseExpr = c->getNumForLoops();
                APSInt num;
                if (collapseExpr->EvaluateAsInt(num,context))
                    loopsToCheck = num.getExtValue();
            }

            //Collects the total parallelism available for the GPU
            totalParallelism  = 0;
            searchFor = true;
            std::cout << "Begin Search at: " << d->getStmtClassName() << std::endl;
            RecursiveASTVisitor<MyASTVisitor>::TraverseStmt(d);
            searchFor = false;

            //Calculates the custom grid geometry based on the total parallelism
            int numTeams;
            if (totalParallelism == 0)
                numTeams = DEFAULT_NUM_TEAMS; 
            else if (totalParallelism/threadLimit > MAX_BLOCKS)
                numTeams = MAX_BLOCKS;
            else
                numTeams = totalParallelism/threadLimit; 

            return numTeams;
        }

        bool VisitForStmt(ForStmt *s) {
            if(searchFor  and loopsToCheck > 0) {
                Stmt *init = s->getInit();
                Expr *cond = s->getCond();
                Expr *incr = s->getInc();
 
                if(init and cond and incr) {
                    std::cout << "FOUND FOR STMT" << std::endl;
                    int start, end, increment;

                    init->dump();
                    if (isa<DeclStmt>(init)) {
                        std::cout << "FOUND DECL" << std::endl;
                        DeclStmt *stmt = reinterpret_cast<DeclStmt*>(init);
                        
                    }
                    else if (isa<BinaryOperator>(init)) {
                        std::cout << "FOUND BINARY OP" << std::endl;
                        BinaryOperator *stmt = reinterpret_cast<BinaryOperator*>(init);
                        APSInt num;
                        stmt->getRHS()->EvaluateAsInt(num,context);
                        start = num.getExtValue();
                        std::cout << "VALUE = " << start << std::endl;
                    }
                    else { 
                        std::cout << "FOUND OTHER" << std::endl; 
                    }
                    cond->dump();
                    if (isa<BinaryOperator>(cond)) {
                        std::cout << "FOUND BINARY OP" << std::endl;
                        BinaryOperator *stmt = reinterpret_cast<BinaryOperator*>(cond);
                        APSInt num;
                        stmt->getRHS()->EvaluateAsInt(num,context);
                        end = num.getExtValue();
                        std::cout << "VALUE = " << end << std::endl;
                    }
                    incr->dump();
                    if (isa<UnaryOperator>(incr)) {
                        std::cout << "FOUND UNARY OP" << std::endl;
                        UnaryOperator *stmt = reinterpret_cast<UnaryOperator*>(cond);
                        APSInt num;
                        stmt->getRHS()->EvaluateAsInt(num,context);
                        increment = num.getExtValue();
                        std::cout << "VALUE = " << increment << std::endl;
                    }

                    --loopsToCheck;
                }
                else {
                    loopsToCheck = 0;
                    totalParallelism = 0;
                }
            }
            return true;
        }
};

//Consumes the created AST by the compiler, with each found function initiates a traversal from which the recursive visitor works
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

//The front end of the tool, which creates the AST consumer for the given file and produces the rewritten code upon tool completion
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
		
//Defines the tools help menu options
static llvm::cl::OptionCategory toolCategory("my-tool options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nMore help text...\n");

//Parses any given options and then applies the tool to the defined files through the frontend action
int main(int argc, const char **argv) {
    CommonOptionsParser op(argc, argv, toolCategory);
    ClangTool Tool(op.getCompilations(),op.getSourcePathList());
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
