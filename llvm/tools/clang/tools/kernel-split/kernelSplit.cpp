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
        bool findParallelDepth = false;
        int stmtDepth;
        int parallelDepth;
        SourceLocation strtPrl;
        SourceLocation lastPrl;

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
            hasParallel = false;
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
 
            //Compares the given directives depth to the highest one and calculates the last and first parallel directives
            if (findParallelDepth) {
                if(parallelDepth == -1) {
                    strtPrl = d->getLocStart();
                    lastPrl = strtPrl;
                }
                else if(rewriter.getSourceMgr().isBeforeInTranslationUnit(lastPrl,d->getLocStart()))
                    lastPrl = d->getLocStart();
                if(stmtDepth < parallelDepth or parallelDepth == -1)
                    parallelDepth = stmtDepth;
            }

            //Handles the transformation of the directive that it at the highest depth
            else if (inTarget and !findEnd and !searchFor and stmtDepth == parallelDepth) {
               
                //Calculates the number of parallel regions at the given depth and handles all serial regions
                if (!firstParallel) {
                    firstParallel = true;
                   
                    bool addTargetData = true;
                    if (inTargetData or strtPrl == lastPrl)
                        addTargetData = false;
 
                    //Adds and all encompassing target data region around parallel code and blocks all serial regions inbetween into seperate target regions 
                    const Stmt *parent = context.getParents(*d)[0].get<Stmt>();
                    const Stmt *prev = NULL;
                    bool prevPrl = false;
                    bool prevSrl = false;
                    bool startSet = false;
                    bool endSet = false;
                    for(ConstStmtIterator iter = parent->child_begin(); iter!=parent->child_end(); ++iter) {

                        if (!startSet and addTargetData) {
                            if(rewriter.getSourceMgr().isBeforeInTranslationUnit(strtPrl,iter->getLocStart())) {
                                if(isa<OMPExecutableDirective>(*prev))
                                    rewriter.InsertText(prev->getLocStart().getLocWithOffset(-PRAGMA_SIZE), "#pragma omp target data "+mapClauses.str()+"\n{\n", true, true);
                                else {
                                    rewriter.InsertText(prev->getLocStart(), "#pragma omp target data "+mapClauses.str()+"\n{\n", true, true);
                                    rewriter.InsertText(prev->getLocStart(), "#pragma omp target teams "+targetClauses.str()+"\n{\n", true, true);
                                }
                                startSet = true;
                            }
                        }
                        else if (!endSet and addTargetData) {
                            if(rewriter.getSourceMgr().isBeforeInTranslationUnit(lastPrl,iter->getLocStart())) {
                                rewriter.InsertText(iter->getLocStart(), "}\n", true, true);
                                endSet = true;
                            }
                        }

                        if (isa<OMPDistributeParallelForDirective>(*iter)) {
                            if(prevSrl and startSet)
                                rewriter.InsertText(iter->getLocStart().getLocWithOffset(-PRAGMA_SIZE), "}\n", true, true);
                            prevPrl = true;
                            prevSrl = false;
                        }
                        else {
                            if(prevPrl and !endSet)
                                rewriter.InsertText(iter->getLocStart(), "#pragma omp target teams "+targetClauses.str()+"\n{\n", true, true);
                            prevPrl = false;
                            prevSrl = true;
                        }

                        prev = *iter;
                    }

                    if (!startSet and addTargetData) {
                        if(isa<OMPExecutableDirective>(*prev))
                            rewriter.InsertText(prev->getLocStart().getLocWithOffset(-PRAGMA_SIZE), "#pragma omp target data "+mapClauses.str()+"\n{\n", true, true);
                        else {
                            rewriter.InsertText(prev->getLocStart(), "#pragma omp target data "+mapClauses.str()+"\n{\n", true, true);
                            rewriter.InsertText(prev->getLocStart(), "#pragma omp target teams "+targetClauses.str()+"\n{\n", true, true);
                        }
                    }

                    if (!endSet and addTargetData)
                        addRightBrkt(const_cast<Stmt*>(prev));
                }

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
            totalParallelism  = 1;
            searchFor = true;
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

        //Function called up finding a for loop in the recursive traversal
        bool VisitForStmt(ForStmt *s) {
            //Used when searching for parallelism by calculating the iterations of the given for loop
            if(searchFor and loopsToCheck > 0) {
                int iterations = getForStmtIterations(s);
                if(iterations == 0) {
                    totalParallelism = 0;
                    loopsToCheck = 0;
                }
                else {
                    totalParallelism *= iterations;
                    loopsToCheck--;
                }
            }
            return true;
        }

        //Searches the given ForStmt for the start, end and increment
        int getForStmtIterations(ForStmt *s) {            
            int start, end, increment;
            bool st = false, en = false, in = false;

            Stmt *init = s->getInit();
            Expr *cond = s->getCond();
            Expr *incr = s->getInc();

            //Tries to calculate the the start
            if (init) {
                if (isa<DeclStmt>(init))
                    st = readDeclStmt(reinterpret_cast<DeclStmt*>(init),&start);
                else if (isa<BinaryOperator>(init))
                    st = readBinaryOperator(reinterpret_cast<BinaryOperator*>(init),&start); 
            }

            //Tries to calculate the end
            if (cond) {
                if (isa<BinaryOperator>(cond))
                    en = readBinaryOperator(reinterpret_cast<BinaryOperator*>(cond),&end);
            }
            
            //Tries to calculate the increment
            if (incr) {
                if (isa<UnaryOperator>(incr))
                    in = readUnaryOperator(reinterpret_cast<UnaryOperator*>(incr),&increment);
                else if (isa<BinaryOperator>(incr))
                    in = readBinaryOperator(reinterpret_cast<BinaryOperator*>(incr),&increment);
            }
            
            //If all found returns number of iterations, otherwise returns 0 which indicates failure
            if (st and en and in)
                return (end-start)/increment;
            return 0;
        }

        //Searchs the binary operator for the right side value 
        bool readBinaryOperator(BinaryOperator *op, int *result) {
            APSInt num;
            if (!isa<ImplicitCastExpr>(op->getLHS())) {
                if (op->getLHS()->EvaluateAsInt(num,context)) {
                    *result = num.getExtValue();
                    return true;
                }
            }
            else if (!isa<ImplicitCastExpr>(op->getRHS())) {
                if(op->getRHS()->EvaluateAsInt(num,context)) {
                    *result = num.getExtValue();
                    return true;
                }
            }
            return false;
        }

        //Searches the unary operator to calculate if it is an increment/decrement
        bool readUnaryOperator(UnaryOperator *op, int *result) {
            if(op->isIncrementOp()) {
                *result = 1;
                return true;
            }
            else if (op->isDecrementOp()) {
                *result = -1;
                return true;
            }
            return false;
        }
 
        //Searches the declaration statement to find what the values initialization is
        bool readDeclStmt(DeclStmt *s, int *result) {
            if(s->isSingleDecl()) {
                Decl *d = s->getSingleDecl();
                if(isa<VarDecl>(d)) {
                    VarDecl *var = reinterpret_cast<VarDecl*>(d);
                    if(var->hasInit()) {
                        Expr *exp = var->getInit();
                        APSInt num;
                        if(exp->EvaluateAsInt(num,context)) {
                            *result = num.getExtValue();
                            return true;
                        }
                    }
                }
            }
            return false;
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
