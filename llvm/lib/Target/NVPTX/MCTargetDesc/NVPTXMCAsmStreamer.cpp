//===- lib/MC/NVPTXMCAsmStreamer.cpp - Assembly Output ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NVPTXMCAsmStreamer.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace {
class NVPTXMCAsmStreamer final : public MCAsmStreamer {
private:
  bool NeedCloseBrace = false;
  bool UseDwarfDirectory = false;
  SmallVector<std::string, 4> DwarfFiles;
  llvm::SmallPtrSet<const MCSymbol *, 8> EmittedSymbols;

public:
  NVPTXMCAsmStreamer(MCContext &Context,
                     std::unique_ptr<formatted_raw_ostream> OS,
                     bool IsVerboseAsm, bool UseDwarfDirectory,
                     MCInstPrinter *Printer, MCCodeEmitter *Emitter,
                     MCAsmBackend *AsmBackend, bool ShowInst)
      : MCAsmStreamer(Context, std::move(OS), IsVerboseAsm, UseDwarfDirectory,
                      Printer, Emitter, AsmBackend, ShowInst),
        UseDwarfDirectory(UseDwarfDirectory) {}
  void ChangeSection(MCSection *Section, const MCExpr *Subsection) override;
  unsigned EmitDwarfFileDirective(unsigned FileNo, StringRef Directory,
                                  StringRef Filename,
                                  unsigned CUID = 0) override;
  void EmitLabel(MCSymbol *Symbol) override;
  void EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                             unsigned Column, unsigned Flags,
                             unsigned Isa, unsigned Discriminator,
                             StringRef FileName) override;
  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;
  void EmitBytes(StringRef Data) override;

};
} // namespace

void NVPTXMCAsmStreamer::ChangeSection(MCSection *Section,
                                       const MCExpr *Subsection) {
  assert(!Subsection && "SubSection is not null!");
  const MCObjectFileInfo *FI = getContext().getObjectFileInfo();
  SmallString<128> Str;
  raw_svector_ostream OS1(Str);
  if (NeedCloseBrace) {
    OS1 << "}\n";
    NeedCloseBrace = false;
  }
  if (Section != FI->getTextSection() && Section != FI->getDataSection()) {
    for (std::string &S : DwarfFiles) {
      MCStreamer::EmitRawText(S.data());
    }
    DwarfFiles.clear();
    NeedCloseBrace = true;
    if (Section == FI->getDwarfLocSection())
      OS1 << ".section .debug_loc {\n";
    else if (Section == FI->getDwarfAbbrevSection())
      OS1 << ".section .debug_abbrev {\n";
    else if (Section == FI->getDwarfInfoSection())
      OS1 << ".section .debug_info {\n";
    else if (Section == FI->getDwarfLineSection())
      NeedCloseBrace = false;
    else
      llvm_unreachable("Unknown dwarf section!");
  }
  MCStreamer::EmitRawText(OS1.str());
  MCStreamer::ChangeSection(Section, Subsection);
}

unsigned NVPTXMCAsmStreamer::EmitDwarfFileDirective(unsigned FileNo,
                                                    StringRef Directory,
                                                    StringRef Filename,
                                                    unsigned CUID) {
  assert(CUID == 0);

  MCDwarfLineTable &Table = getContext().getMCDwarfLineTable(CUID);
  unsigned NumFiles = Table.getMCDwarfFiles().size();
  FileNo = Table.getFile(Directory, Filename, FileNo);
  if (FileNo == 0)
    return 0;
  if (NumFiles == Table.getMCDwarfFiles().size())
    return FileNo;

  SmallString<128> FullPathName;

  SmallString<128> Str;
  raw_svector_ostream OS1(Str);
  if (!UseDwarfDirectory && !Directory.empty()) {
    if (sys::path::is_absolute(Filename))
      Directory = "";
    else {
      FullPathName = Directory;
      sys::path::append(FullPathName, Filename);
      Directory = "";
      Filename = FullPathName;
    }
  }

  OS1 << "\t.file\t" << FileNo << ' ';
  if (!Directory.empty()) {
    PrintQuotedString(Directory, OS1);
    OS1 << ' ';
  }
  PrintQuotedString(Filename, OS1);
  OS1 << "\n";
  DwarfFiles.emplace_back(OS1.str());

  return FileNo;
}

void NVPTXMCAsmStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  const MCObjectFileInfo *FI = getContext().getObjectFileInfo();
  if (getCurrentSectionOnly() == FI->getTextSection() ||
      getCurrentSectionOnly() == FI->getDataSection()) {
    EmittedSymbols.insert(Symbol);
    MCAsmStreamer::EmitLabel(Symbol);
    return;
  }
  MCStreamer::EmitLabel(Symbol);
}

void NVPTXMCAsmStreamer::EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                               unsigned Column, unsigned Flags,
                                               unsigned Isa,
                                               unsigned Discriminator,
                                               StringRef FileName) {
  SmallString<128> Str;
  raw_svector_ostream OS1(Str);
  OS1 << "\t.loc\t" << FileNo << " " << Line << " " << Column << '\n';
  EmitRawText(OS1.str());
  MCStreamer::EmitDwarfLocDirective(FileNo, Line, Column, Flags, Isa,
                                    Discriminator, FileName);
}

void NVPTXMCAsmStreamer::EmitValueImpl(const MCExpr *Value, unsigned Size,
                                       SMLoc Loc) {
  assert(Size <= 8 && "Invalid size");
  assert(getCurrentSectionOnly() &&
         "Cannot emit contents before setting section!");
  const MCAsmInfo *MAI = getContext().getAsmInfo();
  const char *Directive = nullptr;
  switch (Size) {
  default:
    break;
  case 1:
    Directive = MAI->getData8bitsDirective();
    break;
  case 2:
    Directive = MAI->getData16bitsDirective();
    break;
  case 4:
    Directive = MAI->getData32bitsDirective();
    break;
  case 8:
    Directive = MAI->getData64bitsDirective();
    break;
  }

  if (!Directive) {
    MCAsmStreamer::EmitValueImpl(Value, Size, Loc);
    return;
  }

  SmallString<128> Str;
  raw_svector_ostream OS1(Str);
  OS1 << Directive;
  switch (Value->getKind()) {
  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*Value);
    const MCSymbol *Sym = &SRE.getSymbol();
    if (EmittedSymbols.count(Sym) > 0 || Sym->isUsed()) {
      Value->print(OS1, MAI);
      break;
    }
    MCDwarfLineTable &Table = getContext().getMCDwarfLineTable(0);
    if (!Sym->isInSection(/*SetUsed=*/false) && Sym == Table.getLabel()) {
      OS1 << ".debug_line";
      break;
    }
    const MCObjectFileInfo *FI = getContext().getObjectFileInfo();
    if (Sym == FI->getDwarfLocSection()->getBeginSymbol())
      OS1 << ".debug_loc";
    else if (Sym == FI->getDwarfAbbrevSection()->getBeginSymbol())
      OS1 << ".debug_abbrev";
    else if (Sym == FI->getDwarfInfoSection()->getBeginSymbol())
      OS1 << ".debug_info";
    else if (Sym == FI->getDwarfRangesSection()->getBeginSymbol())
      OS1 << ".debug_ranges";
    else if (Sym == FI->getDwarfPubNamesSection()->getBeginSymbol())
      OS1 << ".debug_pubnames";
    else if (Sym == FI->getDwarfLineSection()->getBeginSymbol())
      ;
    else if (EmittedSymbols.count(Sym) > 0 ||
             !Sym->isInSection(/*SetUsed=*/false))
      Value->print(OS1, MAI);
    else
      llvm_unreachable("Unknown section.");
    break;
  }
  case MCExpr::Constant:
    Value->print(OS1, MAI);
    break;
  case MCExpr::Target:
  case MCExpr::Unary:
  case MCExpr::Binary:
    llvm_unreachable("Unexpected expression.");
  }
  EmitRawText(OS1.str());
  EmitEOL();
}

void NVPTXMCAsmStreamer::EmitBytes(StringRef Data) {
  assert(getCurrentSectionOnly() &&
         "Cannot emit contents before setting section!");
  if (Data.empty()) return;

  AddComment(Data);
  const MCAsmInfo *MAI = getContext().getAsmInfo();
  SmallString<128> Str;
  raw_svector_ostream OS1(Str);
  const char *Directive = MAI->getData8bitsDirective();
  for (const unsigned char C : Data.bytes())
    OS1 << Directive << (unsigned)C << "\n";
  EmitRawText(OS1.str());
}

MCStreamer *llvm::createNVPTXAsmStreamer(
    MCContext &Context, std::unique_ptr<formatted_raw_ostream> OS,
    bool isVerboseAsm, bool useDwarfDirectory, MCInstPrinter *IP,
    MCCodeEmitter *CE, MCAsmBackend *MAB, bool ShowInst) {
  return new NVPTXMCAsmStreamer(Context, std::move(OS), isVerboseAsm,
                                useDwarfDirectory, IP, CE, MAB, ShowInst);
}
