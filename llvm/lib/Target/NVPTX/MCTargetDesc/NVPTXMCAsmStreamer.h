//===- lib/MC/MCAsmStreamer.cpp - Text Assembly Output ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
MCStreamer *createNVPTXAsmStreamer(MCContext &Context,
                                   std::unique_ptr<formatted_raw_ostream> OS,
                                   bool isVerboseAsm, bool useDwarfDirectory,
                                   MCInstPrinter *IP, MCCodeEmitter *CE,
                                   MCAsmBackend *MAB, bool ShowInst);
} // namespace llvm
