#include "kernelSplit.h"

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
    private:
        Rewriter &rewriter;
        ASTContext &context;

        //Variables used for the basic tree traversal and application of the splitting
        std::stringstream targetClauses;
        std::stringstream mapClauses;
        bool inTarget = false;
        bool inTargetData = false;
        int stmtDepth;

        //Variables used for the application of custom grid geometry
        bool customGeo = true;
        bool searchFor = false;
        int totalParallelism;
        int loopsToCheck;
        int threadLimit = DEFAULT_THREAD_LIMIT;

        //Variables used for calculating all the implicit and explicit shared mappings required for the given code
        std::forward_list<Variable> vars;
        std::forward_list<Iterator> loopIterators;
        bool findMappings = false;
        SourceLocation startOfTarget;
        
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

        //Function used to retrieve the top level compound stmt, corresponding to the region right within the initial target teams brackets
        Stmt* getTopCompoundStmt(Stmt *s) {
            std::queue<Stmt*> q;
            q.push(s);
            Stmt *curr = NULL;
            while(!q.empty()) {
                curr = q.front();
                q.pop();
                if(isa<CapturedStmt>(*curr))
                    return reinterpret_cast<CapturedStmt*>(curr)->getCapturedStmt();
                for(StmtIterator iter = curr->child_begin(); iter!=curr->child_end(); ++iter)
                    q.push(*iter);
            }
            return NULL;
        }

        //Function called by the visitor before traversing down a compound stmt (for, while, if, do, switch statments generally)
        bool TraverseCompoundStmt(CompoundStmt *s) {
            ++stmtDepth;
            RecursiveASTVisitor<MyASTVisitor>::TraverseCompoundStmt(s);
            --stmtDepth;
            return true;
        }

        //Function called by the visitor before traversing down an OpenMP "target data" directive 
        bool TraverseOMPTargetDataDirective(OMPTargetDataDirective *d) {
            inTargetData = true;
            RecursiveASTVisitor<MyASTVisitor>::TraverseOMPTargetDataDirective(d);
            inTargetData = false;
            return true;
        }

        //Function called by the visitor before traversing down an OpenMP "target teams" directive
        bool TraverseOMPTargetTeamsDirective(OMPTargetTeamsDirective *d) {
 
            //Performs the actual full traversal of the subtree with the previously calculated information
            stmtDepth = 0;
            customGeo = true;
            inTarget = true;
            targetClauses.str("");
            mapClauses.str("");
            vars.clear();
            loopIterators.clear();
            RecursiveASTVisitor<MyASTVisitor>::TraverseOMPTargetTeamsDirective(d);
            threadLimit = DEFAULT_THREAD_LIMIT;
            inTarget = false;

            return true;
        }
        
        //Function called by the visitor upon finding an OpenMP "target teams" directive
        bool VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *d) {

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
                            readMapping(reinterpret_cast<OMPMapClause*>(c));
                        else 
                            targetClauses << rewriter.getRewrittenText(SourceRange(c->getLocStart(),end)) << " ";
                }
            }

            //Creates the new map clauses for the target teams area based on the data gathered
            findMappings = true;
            startOfTarget = d->getLocStart();
            for (StmtIterator iter = d->child_begin(); iter!=d->child_end(); ++iter)
                TraverseStmt(*iter);
            findMappings = false;
            createNewMapClauses();

            //Finds the compound statement holding all the code within brackets
            CompoundStmt *cmpS = reinterpret_cast<CompoundStmt*>(getTopCompoundStmt(d));
            if(!cmpS)
                 return true;
            
            //Removes the original target teams region 
            SourceLocation leftBrkt = Lexer::findLocationAfterToken(d->getLocEnd(),tok::l_brace,rewriter.getSourceMgr(),rewriter.getLangOpts(),true);
            if(leftBrkt.isValid()) {
                rewriter.RemoveText(SourceRange(d->getLocStart().getLocWithOffset(-PRAGMA_SIZE),leftBrkt));
                removeRightBrkt(leftBrkt);
            }
            else
                rewriter.RemoveText(SourceRange(d->getLocStart().getLocWithOffset(-PRAGMA_SIZE),d->getLocEnd()));

            //Adds encompassing target data directive if not within one already
            if (!inTargetData) {
                rewriter.InsertText(cmpS->getLBracLoc(), "\n#pragma omp target data "+mapClauses.str()+"\n{\n", true, true);
                rewriter.InsertText(cmpS->getRBracLoc(), "}\n", true, true);
            }

            //Blocks off the code within, placing serial regions in large target blocks and parallel regions in individual target blocks
            bool prevSrl = false;
            for (StmtIterator iter = cmpS->child_begin(); iter!=cmpS->child_end(); ++iter)
            {
                 if (isa<OMPDistributeParallelForDirective>(*iter)) {
                     if(prevSrl)
                         rewriter.InsertText(iter->getLocStart().getLocWithOffset(-PRAGMA_SIZE), "}\n", true, true);
                     prevSrl = false;
                 }
                 else if (!prevSrl) {
                     if (isa<OMPExecutableDirective>(*iter))
                         rewriter.InsertText(iter->getLocStart().getLocWithOffset(-PRAGMA_SIZE), "#\npragma omp target teams "+targetClauses.str()+"\n{\n", true, true);
                     else
                         rewriter.InsertText(iter->getLocStart(), "\n#pragma omp target teams "+targetClauses.str()+"\n{\n", true, true);
                     prevSrl = true;
                 } 
            }
            
           //Adds final serial region bracket if needed 
           if (prevSrl)
                rewriter.InsertText(cmpS->getRBracLoc(), "}\n", true, true);

            return true;
        }

        //Function called by the visitor upon finding an OpenMP "distribute parallel for" directive
        bool VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *d) {
 
            //Transforms the distribute parallel statmement by adding a target teams if possible for the given directive
            if (inTarget and !searchFor and !findMappings and stmtDepth == 1) {
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

        //Function that calculates improved custom grid geometry for a given distribute parallel for statement
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
            else if (totalParallelism/threadLimit == 0)
                numTeams = 1;
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
            if (!isa<ImplicitCastExpr>(op->getRHS())) {
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

        //Used by the find mapping function to keep track of and record loop iterators that are available for a given array indexing 
        bool TraverseForStmt(ForStmt *s) {
            bool added = false;
            if (findMappings) {
                Expr *exp = s->getCond();
                Expr *name;
                APSInt maxSize;
                bool isEquals = false;
                if (isa<BinaryOperator>(*exp)) {
                    BinaryOperator *op = reinterpret_cast<BinaryOperator*>(exp);
                    if (op->getRHS()->EvaluateAsInt(maxSize,context))
                        name = op->getLHS();
                    else if (op->getLHS()->EvaluateAsInt(maxSize,context))
                        name = op->getRHS();
                    if (op->getOpcode() == BO_LE or op->getOpcode() == BO_GE or op->getOpcode() == BO_EQ)
                        isEquals = true;
                }

                if (name) {
                    if (isa<ImplicitCastExpr>(*name))
                        name = reinterpret_cast<ImplicitCastExpr*>(name)->getSubExpr();
                    if (isa<DeclRefExpr>(*name)) {
                        added = true;
                        Iterator newI;
                        VarDecl *vDecl = reinterpret_cast<VarDecl*>(reinterpret_cast<DeclRefExpr*>(name)->getDecl());
                        newI.name = vDecl->getNameAsString();
                        newI.maxSize = maxSize.getExtValue(); 
                        if (!isEquals)
                            newI.maxSize--;
                        loopIterators.push_front(newI);
                    }
                }
            }
            RecursiveASTVisitor<MyASTVisitor>::TraverseForStmt(s);
            if (findMappings and added)
                loopIterators.pop_front();
            return true;
        }

        //Returns the value held for a specific expression that is either a literal, operation or a variable
        int getVarValue(Expr *e) {
             if (isa<BinaryOperator>(*e))
                 return calculateOperation(reinterpret_cast<BinaryOperator*>(e));
             else if (isa<ImplicitCastExpr>(*e)) {
                Expr *exp = reinterpret_cast<ImplicitCastExpr*>(e)->getSubExpr();
                if (isa<DeclRefExpr>(*exp)) {
                    VarDecl *vDecl = reinterpret_cast<VarDecl*>(reinterpret_cast<DeclRefExpr*>(exp)->getDecl());
                    auto iter = loopIterators.begin();
                    while (iter != loopIterators.end()) {
                        if (iter->name == vDecl->getNameAsString())
                            return iter->maxSize;
                        ++iter;
                    }
                }
             }
             else if (isa<IntegerLiteral>(*e)) {
                  APSInt value;
                  e->EvaluateAsInt(value, context);
                  return value.getExtValue();
             }
             return 0;
        }

        //Calculates the given binary operation if constants are present recursively
        int calculateOperation(BinaryOperator* bop) {
            int leftVal = getVarValue(bop->getLHS());
            int righVal = getVarValue(bop->getRHS()); 
           
            switch (bop->getOpcode()) {
                case (BO_Mul): return leftVal * righVal;
                case (BO_Div): return leftVal / righVal;
                case (BO_Rem): return leftVal % righVal;
                case (BO_Add): return leftVal + righVal;
                case (BO_Sub): return leftVal - righVal; 
                default: break;
            }
            return 0; 
        }

        //Used to find all implicit mappings of arrays that must be performed
        bool VisitArraySubscriptExpr(ArraySubscriptExpr *e) {
            if (findMappings) {

                Expr *exp = e->getBase();
                if (isa<ImplicitCastExpr>(*exp))
                    exp = reinterpret_cast<ImplicitCastExpr*>(exp)->getSubExpr();
                if (!isa<DeclRefExpr>(*exp))
                    return true;
                VarDecl *vDecl = reinterpret_cast<VarDecl*>(reinterpret_cast<DeclRefExpr*>(exp)->getDecl());
           
                if (rewriter.getSourceMgr().isBeforeInTranslationUnit(startOfTarget,vDecl->getLocStart()))
                    return true;

                APSInt index;
                int size = -1;
                if (e->getIdx()->EvaluateAsInt(index,context))
                    size = index.getExtValue();
                else if (isa<BinaryOperator>(*(e->getIdx())))
                    size = calculateOperation(reinterpret_cast<BinaryOperator*>(e->getIdx()))+1;
                addMapping(vDecl->getNameAsString(), OMPC_MAP_tofrom, size, true);
            }
            return true;
        }

        //Used to find all implicit mappings of variables that must be performed
        bool VisitDeclRefExpr(DeclRefExpr *dex) {
            if (findMappings) {
                if (!isa<VarDecl>(*(dex->getDecl())))
                    return true;
                VarDecl *vDecl = reinterpret_cast<VarDecl*>(dex->getDecl());
                if (rewriter.getSourceMgr().isBeforeInTranslationUnit(startOfTarget,vDecl->getLocStart()))
                    return true;
                addMapping(vDecl->getNameAsString(), OMPC_MAP_tofrom, 0, false); 
            }
            return true;
        }

        //Calculates the variables mapped in the given clause
        void readMapping(OMPMapClause *c) {
            OpenMPMapClauseKind mapKind = c->getMapType();

            for (StmtIterator iter = c->children().begin(); iter!=c->children().end(); ++iter) {
                if (isa<OMPArraySectionExpr>(*iter)) {
                    OMPArraySectionExpr *ompArray = reinterpret_cast<OMPArraySectionExpr*>(*iter);
                    Expr *exp = ompArray->getBase();
                    if (isa<ImplicitCastExpr>(*exp))
                        exp = reinterpret_cast<ImplicitCastExpr*>(exp)->getSubExpr();
                    if (!isa<DeclRefExpr>(*exp))
                        continue;
                    VarDecl *vDecl = reinterpret_cast<VarDecl*>(reinterpret_cast<DeclRefExpr*>(exp)->getDecl());

                    APSInt length;
                    int size = -1;
                    if (ompArray->getLength()->EvaluateAsInt(length,context))
                        size = length.getExtValue();
                 
                    addMapping(vDecl->getNameAsString(), mapKind, size, true, true);
                }
            }
            return;
        }

        //Adds the given mem object to the list of those to be mapped
        bool addMapping(std::string name, OpenMPMapClauseKind mapKind, int size, bool isArray, bool inExistingMap=false) {
            if (checkMappings(name, mapKind, size, isArray) and size != -1) {
                Variable newVar;
                newVar.name = name, newVar.mapKind = mapKind, newVar.size = size, newVar.array = isArray, newVar.alreadyMapped = inExistingMap;
                vars.push_front(newVar);
                return true;
            }
            return false;
        }

        //Checks the given mem object to see if it is already recorded for mapping and updates map type and size if needed
        bool checkMappings(std::string name, OpenMPMapClauseKind mapKind, int size, bool isArray) {
            auto iter = vars.begin();
            while (iter != vars.end()) {
                if (iter->name == name) {
                    if (iter->size < size)
                        iter->size = size;
                    if (!iter->alreadyMapped) {
                        if (iter->mapKind < mapKind and !(iter->mapKind == OMPC_MAP_from and mapKind == OMPC_MAP_to))
                            iter->mapKind = mapKind;
                        else if (iter->mapKind == OMPC_MAP_from and mapKind == OMPC_MAP_to)
                            iter->mapKind = OMPC_MAP_tofrom;
                    }
                    if(isArray)
                        iter->array = isArray;
                    return false;
                }
                ++iter; 
            }
            return true;
        }

        //Creates the resulting map clauses for the target region based on the analysis of the target region
        void createNewMapClauses() {
            int alloc = 0, to = 0, from = 0, tofrom = 0;
            std::stringstream a, t, f, tf, temp;
            a << "map(alloc :";
            t << "map(to :";
            f << "map(from :";
            tf << "map(tofrom :";
            
            auto iter = vars.begin();
            while (iter != vars.end()) {
                temp.str("");
                if (iter->array) {
                    temp << "[:"; 
                    if (iter->size != -1)
                        temp << iter->size;
                    temp << "]";
                }
                switch(iter->mapKind) {
                    case OMPC_MAP_alloc:  
                        if (alloc > 0)  a  << ",";  
                        a << " " << iter->name << temp.str();
                        alloc++;  
                        break;
                    case OMPC_MAP_to:     
                        if (to > 0) t  << ",";
                        t << " " << iter->name << temp.str();
                        to++;     
                        break;
                    case OMPC_MAP_from:
                        if (from > 0) f  << ",";  
                        f << " " << iter->name << temp.str();
                        from++;     
                        break;
                    case OMPC_MAP_tofrom:
                        if (tofrom > 0) tf << ",";  
                        tf << " " << iter->name << temp.str();
                        tofrom++;     
                        break;
                    default:
                        if (tofrom > 0) tf << ",";  
                        tf << " " << iter->name << temp.str();
                        tofrom++;     
                        break;
                }
                ++iter;
            }
            if (alloc)
                mapClauses << a.str() << ") ";
            if (to)
                mapClauses << t.str() << ") ";
            if (from)
                mapClauses << f.str() << ") ";
            if (tofrom)
                mapClauses << tf.str() << ") ";
            return;
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
