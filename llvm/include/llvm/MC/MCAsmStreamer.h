//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class Twine;
class MCAsmBackend;
class MCAsmInfo;
class MCCodeEmitter;
class MCExpr;
class MCInst;
class raw_ostream;

class MCAsmStreamer : public MCStreamer {
  std::unique_ptr<formatted_raw_ostream> OSOwner;
  formatted_raw_ostream &OS;
  const MCAsmInfo *MAI;
  std::unique_ptr<MCInstPrinter> InstPrinter;
  std::unique_ptr<MCCodeEmitter> Emitter;
  std::unique_ptr<MCAsmBackend> AsmBackend;

  SmallString<128> ExplicitCommentToEmit;
  SmallString<128> CommentToEmit;
  raw_svector_ostream CommentStream;

  unsigned IsVerboseAsm : 1;
  unsigned ShowInst : 1;
  unsigned UseDwarfDirectory : 1;

  void EmitRegisterName(int64_t Register);
  void EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override;
  void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) override;

public:
  MCAsmStreamer(MCContext &Context, std::unique_ptr<formatted_raw_ostream> os,
                bool isVerboseAsm, bool useDwarfDirectory,
                MCInstPrinter *printer, MCCodeEmitter *emitter,
                MCAsmBackend *asmbackend, bool showInst)
      : MCStreamer(Context), OSOwner(std::move(os)), OS(*OSOwner),
        MAI(Context.getAsmInfo()), InstPrinter(printer), Emitter(emitter),
        AsmBackend(asmbackend), CommentStream(CommentToEmit),
        IsVerboseAsm(isVerboseAsm), ShowInst(showInst),
        UseDwarfDirectory(useDwarfDirectory) {
    assert(InstPrinter);
    if (IsVerboseAsm)
        InstPrinter->setCommentStream(CommentStream);
  }

  void EmitEOL();

  void EmitSyntaxDirective() override;

  void EmitCommentsAndEOL();

  /// Prints the specified quoted \p Data string to the stream \p OS.
  static void PrintQuotedString(StringRef Data, raw_ostream &OS);

  /// isVerboseAsm - Return true if this streamer supports verbose assembly at
  /// all.
  bool isVerboseAsm() const override { return IsVerboseAsm; }

  /// hasRawTextSupport - We support EmitRawText.
  bool hasRawTextSupport() const override { return true; }

  /// AddComment - Add a comment that can be emitted to the generated .s
  /// file if applicable as a QoI issue to make the output of the compiler
  /// more readable.  This only affects the MCAsmStreamer, and only when
  /// verbose assembly output is enabled.
  void AddComment(const Twine &T, bool EOL = true) override;

  /// AddEncodingComment - Add a comment showing the encoding of an instruction.
  void AddEncodingComment(const MCInst &Inst, const MCSubtargetInfo &);

  /// GetCommentOS - Return a raw_ostream that comments can be written to.
  /// Unlike AddComment, you are required to terminate comments with \n if you
  /// use this method.
  raw_ostream &GetCommentOS() override {
    if (!IsVerboseAsm)
      return nulls();  // Discard comments unless in verbose asm mode.
    return CommentStream;
  }

  void emitRawComment(const Twine &T, bool TabPrefix = true) override;

  void addExplicitComment(const Twine &T) override;
  void emitExplicitComments() override;

  /// AddBlankLine - Emit a blank line to a .s file to pretty it up.
  void AddBlankLine() override {
    EmitEOL();
  }

  /// @name MCStreamer Interface
  /// @{

  void ChangeSection(MCSection *Section, const MCExpr *Subsection) override;

  void EmitLOHDirective(MCLOHType Kind, const MCLOHArgs &Args) override;
  void EmitLabel(MCSymbol *Symbol) override;

  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitLinkerOptions(ArrayRef<std::string> Options) override;
  void EmitDataRegion(MCDataRegionType Kind) override;
  void EmitVersionMin(MCVersionMinType Kind, unsigned Major, unsigned Minor,
                      unsigned Update) override;
  void EmitThumbFunc(MCSymbol *Func) override;

  void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;

  void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void BeginCOFFSymbolDef(const MCSymbol *Symbol) override;
  void EmitCOFFSymbolStorageClass(int StorageClass) override;
  void EmitCOFFSymbolType(int Type) override;
  void EndCOFFSymbolDef() override;
  void EmitCOFFSafeSEH(MCSymbol const *Symbol) override;
  void EmitCOFFSectionIndex(MCSymbol const *Symbol) override;
  void EmitCOFFSecRel32(MCSymbol const *Symbol, uint64_t Offset) override;
  void emitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;

  /// EmitLocalCommonSymbol - Emit a local common (.lcomm) symbol.
  ///
  /// @param Symbol - The common symbol to emit.
  /// @param Size - The size of the common symbol.
  /// @param ByteAlignment - The alignment of the common symbol in bytes.
  void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;

  void EmitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0) override;

  void EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                      unsigned ByteAlignment = 0) override;

  void EmitBinaryData(StringRef Data) override;

  void EmitBytes(StringRef Data) override;

  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;
  void EmitIntValue(uint64_t Value, unsigned Size) override;

  void EmitULEB128Value(const MCExpr *Value) override;

  void EmitSLEB128Value(const MCExpr *Value) override;

  void EmitDTPRel32Value(const MCExpr *Value) override;
  void EmitDTPRel64Value(const MCExpr *Value) override;
  void EmitTPRel32Value(const MCExpr *Value) override;
  void EmitTPRel64Value(const MCExpr *Value) override;

  void EmitGPRel64Value(const MCExpr *Value) override;

  void EmitGPRel32Value(const MCExpr *Value) override;


  void emitFill(uint64_t NumBytes, uint8_t FillValue) override;

  void emitFill(const MCExpr &NumBytes, uint64_t FillValue,
                SMLoc Loc = SMLoc()) override;

  void emitFill(uint64_t NumValues, int64_t Size, int64_t Expr) override;

  void emitFill(const MCExpr &NumValues, int64_t Size, int64_t Expr,
                SMLoc Loc = SMLoc()) override;

  void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                            unsigned ValueSize = 1,
                            unsigned MaxBytesToEmit = 0) override;

  void EmitCodeAlignment(unsigned ByteAlignment,
                         unsigned MaxBytesToEmit = 0) override;

  void emitValueToOffset(const MCExpr *Offset,
                         unsigned char Value,
                         SMLoc Loc) override;

  void EmitFileDirective(StringRef Filename) override;
  unsigned EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                  StringRef Filename,
                                  unsigned CUID = 0) override;
  void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                             unsigned Column, unsigned Flags,
                             unsigned Isa, unsigned Discriminator,
                             StringRef FileName) override;
  MCSymbol *getDwarfLineTableSymbol(unsigned CUID) override;

  bool EmitCVFileDirective(unsigned FileNo, StringRef Filename) override;
  bool EmitCVFuncIdDirective(unsigned FuncId) override;
  bool EmitCVInlineSiteIdDirective(unsigned FunctionId, unsigned IAFunc,
                                   unsigned IAFile, unsigned IALine,
                                   unsigned IACol, SMLoc Loc) override;
  void EmitCVLocDirective(unsigned FunctionId, unsigned FileNo, unsigned Line,
                          unsigned Column, bool PrologueEnd, bool IsStmt,
                          StringRef FileName, SMLoc Loc) override;
  void EmitCVLinetableDirective(unsigned FunctionId, const MCSymbol *FnStart,
                                const MCSymbol *FnEnd) override;
  void EmitCVInlineLinetableDirective(unsigned PrimaryFunctionId,
                                      unsigned SourceFileId,
                                      unsigned SourceLineNum,
                                      const MCSymbol *FnStartSym,
                                      const MCSymbol *FnEndSym) override;
  void EmitCVDefRangeDirective(
      ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> Ranges,
      StringRef FixedSizePortion) override;
  void EmitCVStringTableDirective() override;
  void EmitCVFileChecksumsDirective() override;

  void EmitIdent(StringRef IdentString) override;
  void EmitCFISections(bool EH, bool Debug) override;
  void EmitCFIDefCfa(int64_t Register, int64_t Offset) override;
  void EmitCFIDefCfaOffset(int64_t Offset) override;
  void EmitCFIDefCfaRegister(int64_t Register) override;
  void EmitCFIOffset(int64_t Register, int64_t Offset) override;
  void EmitCFIPersonality(const MCSymbol *Sym, unsigned Encoding) override;
  void EmitCFILsda(const MCSymbol *Sym, unsigned Encoding) override;
  void EmitCFIRememberState() override;
  void EmitCFIRestoreState() override;
  void EmitCFISameValue(int64_t Register) override;
  void EmitCFIRelOffset(int64_t Register, int64_t Offset) override;
  void EmitCFIAdjustCfaOffset(int64_t Adjustment) override;
  void EmitCFIEscape(StringRef Values) override;
  void EmitCFIGnuArgsSize(int64_t Size) override;
  void EmitCFISignalFrame() override;
  void EmitCFIUndefined(int64_t Register) override;
  void EmitCFIRegister(int64_t Register1, int64_t Register2) override;
  void EmitCFIWindowSave() override;

  void EmitWinCFIStartProc(const MCSymbol *Symbol) override;
  void EmitWinCFIEndProc() override;
  void EmitWinCFIStartChained() override;
  void EmitWinCFIEndChained() override;
  void EmitWinCFIPushReg(unsigned Register) override;
  void EmitWinCFISetFrame(unsigned Register, unsigned Offset) override;
  void EmitWinCFIAllocStack(unsigned Size) override;
  void EmitWinCFISaveReg(unsigned Register, unsigned Offset) override;
  void EmitWinCFISaveXMM(unsigned Register, unsigned Offset) override;
  void EmitWinCFIPushFrame(bool Code) override;
  void EmitWinCFIEndProlog() override;

  void EmitWinEHHandler(const MCSymbol *Sym, bool Unwind, bool Except) override;
  void EmitWinEHHandlerData() override;

  void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;

  void EmitBundleAlignMode(unsigned AlignPow2) override;
  void EmitBundleLock(bool AlignToEnd) override;
  void EmitBundleUnlock() override;

  bool EmitRelocDirective(const MCExpr &Offset, StringRef Name,
                          const MCExpr *Expr, SMLoc Loc) override;

  /// EmitRawText - If this file is backed by an assembly streamer, this dumps
  /// the specified string in the output .s file.  This capability is
  /// indicated by the hasRawTextSupport() predicate.
  void EmitRawTextImpl(StringRef String) override;

  void FinishImpl() override;
};

} // end namespace llvm
