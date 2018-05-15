; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64

; Verify that the NVPTX target removes invalid symbol names prior to emitting
; PTX.
; A symbol is invalid in PTX if it contains a '.' or a '@'.

; PTX32-NOT: .str
; PTX64-NOT: .str

; PTX32-DAG: _$_str_$_1
; PTX32-DAG: _$_str

; PTX64-DAG: _$_str_$_1
; PTX64-DAG: _$_str

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"


@.str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1
@_$_str = private unnamed_addr constant [13 x i8] c"%d %f %c %d\0A\00", align 1


; Function Attrs: nounwind
define void @foo(i32 %a, float %b, i8 signext %c, i32 %e) {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0))
  ret void
}

; PTX32-NOT: omp_outlined.1
; PTX64-NOT: omp_outlined.1
; PTX32-DAG: omp_outlined_$_1_$_2_$_3
; PTX64-DAG: omp_outlined_$_1_$_2_$_3
define internal void @omp_outlined.1() {
  ret void
}

declare void @omp_outlined_$_1()
declare void @omp_outlined_$_1_$_2()

declare i32 @printf(i8*, ...)
