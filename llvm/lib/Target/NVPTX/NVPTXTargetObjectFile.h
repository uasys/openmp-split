//===-- NVPTXTargetObjectFile.h - NVPTX Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXTARGETOBJECTFILE_H

#include "NVPTXSection.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
class GlobalVariable;
class Module;

class NVPTXTargetObjectFile : public TargetLoweringObjectFile {

public:
  NVPTXTargetObjectFile() {
  }

  virtual ~NVPTXTargetObjectFile();

  void Initialize(MCContext &ctx, const TargetMachine &TM) override {
    TargetLoweringObjectFile::Initialize(ctx, TM);
  }

  MCSection *getSectionForConstant(const DataLayout &DL, SectionKind Kind,
                                   const Constant *C,
                                   unsigned &Align) const override {
    return ReadOnlySection;
  }

  MCSection *getExplicitSectionGlobal(const GlobalObject *GO, SectionKind Kind,
                                      const TargetMachine &TM) const override {
    return DataSection;
  }

  MCSection *SelectSectionForGlobal(const GlobalObject *GO, SectionKind Kind,
                                    const TargetMachine &TM) const override;
};

} // end namespace llvm

#endif
