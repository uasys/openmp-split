//===-- NVPTXAssignValidGlobalNames.cpp - Assign valid names to globals ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Clean up the names of global variables in the module to not contain symbols
// that are invalid in PTX.
//
// Currently NVPTX, like other backends, relies on generic symbol name
// sanitizing done by MC. However, the ptxas assembler is more stringent and
// disallows some additional characters in symbol names. This pass makes sure
// such names do not reach MC at all.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

namespace {
/// \brief NVPTXAssignValidGlobalNames
class NVPTXAssignValidGlobalNames : public ModulePass {
public:
  static char ID;
  NVPTXAssignValidGlobalNames() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  /// \brief Clean up the name to remove symbols invalid in PTX.
  std::string cleanUpName(StringRef Name);
  /// \brief Set a clean name, ensuring collisions are avoided.
  void generateCleanName(Value &V);
};
}

char NVPTXAssignValidGlobalNames::ID = 0;

namespace llvm {
void initializeNVPTXAssignValidGlobalNamesPass(PassRegistry &);
}

INITIALIZE_PASS(NVPTXAssignValidGlobalNames, "nvptx-assign-valid-global-names",
                "Assign valid PTX names to globals", false, false)

bool NVPTXAssignValidGlobalNames::runOnModule(Module &M) {
  for (GlobalVariable &GV : M.globals()) {
    // We are only allowed to rename local symbols.
    if (GV.hasLocalLinkage()) {
      generateCleanName(GV);
    }
  }

  // Clean function symbols.
  for (auto &FN : M.functions()) {
    if (FN.hasLocalLinkage()) {
      generateCleanName(FN);
    }
  }

  return true;
}

void NVPTXAssignValidGlobalNames::generateCleanName(Value &V) {
  while (1) {
    std::string ValidName = cleanUpName(V.getName());
    // setName doesn't do extra work if the name does not change.
    // Collisions are avoided by adding a suffix (which may yet be unclean in
    // PTX).
    V.setName(ValidName);
    // If there are no collisions return, otherwise clean up the new name.
    if (V.getName().equals(ValidName))
      return;
  }
}

std::string NVPTXAssignValidGlobalNames::cleanUpName(StringRef Name) {
  std::string ValidName;
  raw_string_ostream ValidNameStream(ValidName);
  for (unsigned I = 0, E = Name.size(); I != E; ++I) {
    char C = Name[I];
    if (C == '.' || C == '@') {
      ValidNameStream << "_$_";
    } else {
      ValidNameStream << C;
    }
  }

  return ValidNameStream.str();
}

ModulePass *llvm::createNVPTXAssignValidGlobalNamesPass() {
  return new NVPTXAssignValidGlobalNames();
}
