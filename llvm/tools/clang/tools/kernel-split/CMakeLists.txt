set(LLVM_LINK_COMPONENTS support)

add_clang_executable(kernel-split
  kernelSplit.cpp
  )
target_link_libraries(kernel-split
  clangTooling
  clangBasic
  clangASTMatchers
  )
