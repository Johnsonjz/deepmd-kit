/*
 * Copyright 2018-2023 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .section .note.GNU-stack,"",@progbits

  .data

  .globl _libcudart_so_tramp_table
  .hidden _libcudart_so_tramp_table
  .align 8
_libcudart_so_tramp_table:
  .zero 3488

  .text

  .globl _libcudart_so_tramp_resolve
  .hidden _libcudart_so_tramp_resolve

  .globl _libcudart_so_save_regs_and_resolve
  .hidden _libcudart_so_save_regs_and_resolve
  .type _libcudart_so_save_regs_and_resolve, %function
_libcudart_so_save_regs_and_resolve:
  .cfi_startproc

#define PUSH_REG(reg) pushq %reg ; .cfi_adjust_cfa_offset 8; .cfi_rel_offset reg, 0
#define POP_REG(reg) popq %reg ; .cfi_adjust_cfa_offset -8; .cfi_restore reg

#define DEC_STACK(d) subq $d, %rsp; .cfi_adjust_cfa_offset d
#define INC_STACK(d) addq $d, %rsp; .cfi_adjust_cfa_offset -d

#define PUSH_XMM_REG(reg) DEC_STACK(16); movdqa %reg, (%rsp); .cfi_rel_offset reg, 0
#define POP_XMM_REG(reg) movdqa (%rsp), %reg; .cfi_restore reg; INC_STACK(16)

  // Slow path which calls dlsym, taken only on first call.
  // All registers are stored to handle arbitrary calling conventions
  // (except x87 FPU registers which do not have to be preserved).
  // For Dwarf directives, read https://www.imperialviolet.org/2017/01/18/cfi.html.

  .cfi_def_cfa_offset 8  // Return address

  // FIXME: AVX (YMM, ZMM) registers are NOT saved to simplify code.

  PUSH_REG(rdi)  // 16
  mov 0x10(%rsp), %rdi
  PUSH_REG(rax)
  PUSH_REG(rbx)  // 16
  PUSH_REG(rcx)
  PUSH_REG(rdx)  // 16
  PUSH_REG(rbp)
  PUSH_REG(rsi)  // 16
  PUSH_REG(r8)
  PUSH_REG(r9)  // 16
  PUSH_REG(r10)
  PUSH_REG(r11)  // 16
  PUSH_REG(r12)
  PUSH_REG(r13)  // 16
  PUSH_REG(r14)
  PUSH_REG(r15)  // 16
  PUSH_XMM_REG(xmm0)
  PUSH_XMM_REG(xmm1)
  PUSH_XMM_REG(xmm2)
  PUSH_XMM_REG(xmm3)
  PUSH_XMM_REG(xmm4)
  PUSH_XMM_REG(xmm5)
  PUSH_XMM_REG(xmm6)
  PUSH_XMM_REG(xmm7)

  // Stack is just 8-byte aligned but callee will re-align to 16
  call _libcudart_so_tramp_resolve

  POP_XMM_REG(xmm7)
  POP_XMM_REG(xmm6)
  POP_XMM_REG(xmm5)
  POP_XMM_REG(xmm4)
  POP_XMM_REG(xmm3)
  POP_XMM_REG(xmm2)
  POP_XMM_REG(xmm1)
  POP_XMM_REG(xmm0)  // 16
  POP_REG(r15)
  POP_REG(r14)  // 16
  POP_REG(r13)
  POP_REG(r12)  // 16
  POP_REG(r11)
  POP_REG(r10)  // 16
  POP_REG(r9)
  POP_REG(r8)  // 16
  POP_REG(rsi)
  POP_REG(rbp)  // 16
  POP_REG(rdx)
  POP_REG(rcx)  // 16
  POP_REG(rbx)
  POP_REG(rax)  // 16
  POP_REG(rdi)

  ret

  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaGetKernel
  .p2align 4
  .type __cudaGetKernel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaGetKernel
#endif
__cudaGetKernel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+0(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+0(%rip)
2:
  pushq $0
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaInitModule
  .p2align 4
  .type __cudaInitModule, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaInitModule
#endif
__cudaInitModule:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+8(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+8(%rip)
2:
  pushq $1
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaLaunchKernel
  .p2align 4
  .type __cudaLaunchKernel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaLaunchKernel
#endif
__cudaLaunchKernel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+16(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+16(%rip)
2:
  pushq $2
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaLaunchKernel_ptsz
  .p2align 4
  .type __cudaLaunchKernel_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaLaunchKernel_ptsz
#endif
__cudaLaunchKernel_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+24(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+24(%rip)
2:
  pushq $3
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaPopCallConfiguration
  .p2align 4
  .type __cudaPopCallConfiguration, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaPopCallConfiguration
#endif
__cudaPopCallConfiguration:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+32(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+32(%rip)
2:
  pushq $4
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaPushCallConfiguration
  .p2align 4
  .type __cudaPushCallConfiguration, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaPushCallConfiguration
#endif
__cudaPushCallConfiguration:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+40(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+40(%rip)
2:
  pushq $5
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterFatBinary
  .p2align 4
  .type __cudaRegisterFatBinary, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterFatBinary
#endif
__cudaRegisterFatBinary:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+48(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+48(%rip)
2:
  pushq $6
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterFatBinaryEnd
  .p2align 4
  .type __cudaRegisterFatBinaryEnd, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterFatBinaryEnd
#endif
__cudaRegisterFatBinaryEnd:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+56(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+56(%rip)
2:
  pushq $7
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterFunction
  .p2align 4
  .type __cudaRegisterFunction, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterFunction
#endif
__cudaRegisterFunction:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+64(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+64(%rip)
2:
  pushq $8
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterHostVar
  .p2align 4
  .type __cudaRegisterHostVar, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterHostVar
#endif
__cudaRegisterHostVar:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+72(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+72(%rip)
2:
  pushq $9
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterManagedVar
  .p2align 4
  .type __cudaRegisterManagedVar, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterManagedVar
#endif
__cudaRegisterManagedVar:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+80(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+80(%rip)
2:
  pushq $10
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterUnifiedTable
  .p2align 4
  .type __cudaRegisterUnifiedTable, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterUnifiedTable
#endif
__cudaRegisterUnifiedTable:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+88(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+88(%rip)
2:
  pushq $11
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaRegisterVar
  .p2align 4
  .type __cudaRegisterVar, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaRegisterVar
#endif
__cudaRegisterVar:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+96(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+96(%rip)
2:
  pushq $12
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl __cudaUnregisterFatBinary
  .p2align 4
  .type __cudaUnregisterFatBinary, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden __cudaUnregisterFatBinary
#endif
__cudaUnregisterFatBinary:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+104(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+104(%rip)
2:
  pushq $13
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaArrayGetInfo
  .p2align 4
  .type cudaArrayGetInfo, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaArrayGetInfo
#endif
cudaArrayGetInfo:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+112(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+112(%rip)
2:
  pushq $14
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaArrayGetMemoryRequirements
  .p2align 4
  .type cudaArrayGetMemoryRequirements, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaArrayGetMemoryRequirements
#endif
cudaArrayGetMemoryRequirements:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+120(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+120(%rip)
2:
  pushq $15
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaArrayGetPlane
  .p2align 4
  .type cudaArrayGetPlane, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaArrayGetPlane
#endif
cudaArrayGetPlane:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+128(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+128(%rip)
2:
  pushq $16
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaArrayGetSparseProperties
  .p2align 4
  .type cudaArrayGetSparseProperties, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaArrayGetSparseProperties
#endif
cudaArrayGetSparseProperties:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+136(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+136(%rip)
2:
  pushq $17
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaChooseDevice
  .p2align 4
  .type cudaChooseDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaChooseDevice
#endif
cudaChooseDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+144(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+144(%rip)
2:
  pushq $18
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaCreateChannelDesc
  .p2align 4
  .type cudaCreateChannelDesc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaCreateChannelDesc
#endif
cudaCreateChannelDesc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+152(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+152(%rip)
2:
  pushq $19
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaCreateSurfaceObject
  .p2align 4
  .type cudaCreateSurfaceObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaCreateSurfaceObject
#endif
cudaCreateSurfaceObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+160(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+160(%rip)
2:
  pushq $20
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaCreateTextureObject
  .p2align 4
  .type cudaCreateTextureObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaCreateTextureObject
#endif
cudaCreateTextureObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+168(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+168(%rip)
2:
  pushq $21
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaCtxResetPersistingL2Cache
  .p2align 4
  .type cudaCtxResetPersistingL2Cache, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaCtxResetPersistingL2Cache
#endif
cudaCtxResetPersistingL2Cache:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+176(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+176(%rip)
2:
  pushq $22
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDestroyExternalMemory
  .p2align 4
  .type cudaDestroyExternalMemory, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDestroyExternalMemory
#endif
cudaDestroyExternalMemory:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+184(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+184(%rip)
2:
  pushq $23
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDestroyExternalSemaphore
  .p2align 4
  .type cudaDestroyExternalSemaphore, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDestroyExternalSemaphore
#endif
cudaDestroyExternalSemaphore:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+192(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+192(%rip)
2:
  pushq $24
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDestroySurfaceObject
  .p2align 4
  .type cudaDestroySurfaceObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDestroySurfaceObject
#endif
cudaDestroySurfaceObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+200(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+200(%rip)
2:
  pushq $25
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDestroyTextureObject
  .p2align 4
  .type cudaDestroyTextureObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDestroyTextureObject
#endif
cudaDestroyTextureObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+208(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+208(%rip)
2:
  pushq $26
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceCanAccessPeer
  .p2align 4
  .type cudaDeviceCanAccessPeer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceCanAccessPeer
#endif
cudaDeviceCanAccessPeer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+216(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+216(%rip)
2:
  pushq $27
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceDisablePeerAccess
  .p2align 4
  .type cudaDeviceDisablePeerAccess, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceDisablePeerAccess
#endif
cudaDeviceDisablePeerAccess:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+224(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+224(%rip)
2:
  pushq $28
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceEnablePeerAccess
  .p2align 4
  .type cudaDeviceEnablePeerAccess, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceEnablePeerAccess
#endif
cudaDeviceEnablePeerAccess:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+232(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+232(%rip)
2:
  pushq $29
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceFlushGPUDirectRDMAWrites
  .p2align 4
  .type cudaDeviceFlushGPUDirectRDMAWrites, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceFlushGPUDirectRDMAWrites
#endif
cudaDeviceFlushGPUDirectRDMAWrites:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+240(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+240(%rip)
2:
  pushq $30
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetAttribute
  .p2align 4
  .type cudaDeviceGetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetAttribute
#endif
cudaDeviceGetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+248(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+248(%rip)
2:
  pushq $31
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetByPCIBusId
  .p2align 4
  .type cudaDeviceGetByPCIBusId, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetByPCIBusId
#endif
cudaDeviceGetByPCIBusId:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+256(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+256(%rip)
2:
  pushq $32
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetCacheConfig
  .p2align 4
  .type cudaDeviceGetCacheConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetCacheConfig
#endif
cudaDeviceGetCacheConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+264(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+264(%rip)
2:
  pushq $33
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetDefaultMemPool
  .p2align 4
  .type cudaDeviceGetDefaultMemPool, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetDefaultMemPool
#endif
cudaDeviceGetDefaultMemPool:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+272(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+272(%rip)
2:
  pushq $34
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetGraphMemAttribute
  .p2align 4
  .type cudaDeviceGetGraphMemAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetGraphMemAttribute
#endif
cudaDeviceGetGraphMemAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+280(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+280(%rip)
2:
  pushq $35
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetLimit
  .p2align 4
  .type cudaDeviceGetLimit, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetLimit
#endif
cudaDeviceGetLimit:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+288(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+288(%rip)
2:
  pushq $36
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetMemPool
  .p2align 4
  .type cudaDeviceGetMemPool, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetMemPool
#endif
cudaDeviceGetMemPool:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+296(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+296(%rip)
2:
  pushq $37
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetNvSciSyncAttributes
  .p2align 4
  .type cudaDeviceGetNvSciSyncAttributes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetNvSciSyncAttributes
#endif
cudaDeviceGetNvSciSyncAttributes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+304(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+304(%rip)
2:
  pushq $38
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetP2PAttribute
  .p2align 4
  .type cudaDeviceGetP2PAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetP2PAttribute
#endif
cudaDeviceGetP2PAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+312(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+312(%rip)
2:
  pushq $39
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetPCIBusId
  .p2align 4
  .type cudaDeviceGetPCIBusId, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetPCIBusId
#endif
cudaDeviceGetPCIBusId:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+320(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+320(%rip)
2:
  pushq $40
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetSharedMemConfig
  .p2align 4
  .type cudaDeviceGetSharedMemConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetSharedMemConfig
#endif
cudaDeviceGetSharedMemConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+328(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+328(%rip)
2:
  pushq $41
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetStreamPriorityRange
  .p2align 4
  .type cudaDeviceGetStreamPriorityRange, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetStreamPriorityRange
#endif
cudaDeviceGetStreamPriorityRange:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+336(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+336(%rip)
2:
  pushq $42
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGetTexture1DLinearMaxWidth
  .p2align 4
  .type cudaDeviceGetTexture1DLinearMaxWidth, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGetTexture1DLinearMaxWidth
#endif
cudaDeviceGetTexture1DLinearMaxWidth:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+344(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+344(%rip)
2:
  pushq $43
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceGraphMemTrim
  .p2align 4
  .type cudaDeviceGraphMemTrim, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceGraphMemTrim
#endif
cudaDeviceGraphMemTrim:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+352(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+352(%rip)
2:
  pushq $44
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceRegisterAsyncNotification
  .p2align 4
  .type cudaDeviceRegisterAsyncNotification, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceRegisterAsyncNotification
#endif
cudaDeviceRegisterAsyncNotification:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+360(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+360(%rip)
2:
  pushq $45
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceReset
  .p2align 4
  .type cudaDeviceReset, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceReset
#endif
cudaDeviceReset:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+368(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+368(%rip)
2:
  pushq $46
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceSetCacheConfig
  .p2align 4
  .type cudaDeviceSetCacheConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceSetCacheConfig
#endif
cudaDeviceSetCacheConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+376(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+376(%rip)
2:
  pushq $47
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceSetGraphMemAttribute
  .p2align 4
  .type cudaDeviceSetGraphMemAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceSetGraphMemAttribute
#endif
cudaDeviceSetGraphMemAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+384(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+384(%rip)
2:
  pushq $48
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceSetLimit
  .p2align 4
  .type cudaDeviceSetLimit, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceSetLimit
#endif
cudaDeviceSetLimit:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+392(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+392(%rip)
2:
  pushq $49
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceSetMemPool
  .p2align 4
  .type cudaDeviceSetMemPool, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceSetMemPool
#endif
cudaDeviceSetMemPool:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+400(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+400(%rip)
2:
  pushq $50
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceSetSharedMemConfig
  .p2align 4
  .type cudaDeviceSetSharedMemConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceSetSharedMemConfig
#endif
cudaDeviceSetSharedMemConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+408(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+408(%rip)
2:
  pushq $51
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceSynchronize
  .p2align 4
  .type cudaDeviceSynchronize, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceSynchronize
#endif
cudaDeviceSynchronize:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+416(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+416(%rip)
2:
  pushq $52
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDeviceUnregisterAsyncNotification
  .p2align 4
  .type cudaDeviceUnregisterAsyncNotification, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDeviceUnregisterAsyncNotification
#endif
cudaDeviceUnregisterAsyncNotification:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+424(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+424(%rip)
2:
  pushq $53
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaDriverGetVersion
  .p2align 4
  .type cudaDriverGetVersion, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaDriverGetVersion
#endif
cudaDriverGetVersion:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+432(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+432(%rip)
2:
  pushq $54
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamConsumerAcquireFrame
  .p2align 4
  .type cudaEGLStreamConsumerAcquireFrame, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamConsumerAcquireFrame
#endif
cudaEGLStreamConsumerAcquireFrame:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+440(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+440(%rip)
2:
  pushq $55
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamConsumerConnect
  .p2align 4
  .type cudaEGLStreamConsumerConnect, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamConsumerConnect
#endif
cudaEGLStreamConsumerConnect:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+448(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+448(%rip)
2:
  pushq $56
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamConsumerConnectWithFlags
  .p2align 4
  .type cudaEGLStreamConsumerConnectWithFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamConsumerConnectWithFlags
#endif
cudaEGLStreamConsumerConnectWithFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+456(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+456(%rip)
2:
  pushq $57
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamConsumerDisconnect
  .p2align 4
  .type cudaEGLStreamConsumerDisconnect, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamConsumerDisconnect
#endif
cudaEGLStreamConsumerDisconnect:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+464(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+464(%rip)
2:
  pushq $58
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamConsumerReleaseFrame
  .p2align 4
  .type cudaEGLStreamConsumerReleaseFrame, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamConsumerReleaseFrame
#endif
cudaEGLStreamConsumerReleaseFrame:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+472(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+472(%rip)
2:
  pushq $59
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamProducerConnect
  .p2align 4
  .type cudaEGLStreamProducerConnect, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamProducerConnect
#endif
cudaEGLStreamProducerConnect:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+480(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+480(%rip)
2:
  pushq $60
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamProducerDisconnect
  .p2align 4
  .type cudaEGLStreamProducerDisconnect, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamProducerDisconnect
#endif
cudaEGLStreamProducerDisconnect:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+488(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+488(%rip)
2:
  pushq $61
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamProducerPresentFrame
  .p2align 4
  .type cudaEGLStreamProducerPresentFrame, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamProducerPresentFrame
#endif
cudaEGLStreamProducerPresentFrame:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+496(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+496(%rip)
2:
  pushq $62
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEGLStreamProducerReturnFrame
  .p2align 4
  .type cudaEGLStreamProducerReturnFrame, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEGLStreamProducerReturnFrame
#endif
cudaEGLStreamProducerReturnFrame:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+504(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+504(%rip)
2:
  pushq $63
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventCreate
  .p2align 4
  .type cudaEventCreate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventCreate
#endif
cudaEventCreate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+512(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+512(%rip)
2:
  pushq $64
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventCreateFromEGLSync
  .p2align 4
  .type cudaEventCreateFromEGLSync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventCreateFromEGLSync
#endif
cudaEventCreateFromEGLSync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+520(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+520(%rip)
2:
  pushq $65
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventCreateWithFlags
  .p2align 4
  .type cudaEventCreateWithFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventCreateWithFlags
#endif
cudaEventCreateWithFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+528(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+528(%rip)
2:
  pushq $66
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventDestroy
  .p2align 4
  .type cudaEventDestroy, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventDestroy
#endif
cudaEventDestroy:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+536(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+536(%rip)
2:
  pushq $67
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventElapsedTime
  .p2align 4
  .type cudaEventElapsedTime, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventElapsedTime
#endif
cudaEventElapsedTime:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+544(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+544(%rip)
2:
  pushq $68
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventElapsedTime_v2
  .p2align 4
  .type cudaEventElapsedTime_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventElapsedTime_v2
#endif
cudaEventElapsedTime_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+552(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+552(%rip)
2:
  pushq $69
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventQuery
  .p2align 4
  .type cudaEventQuery, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventQuery
#endif
cudaEventQuery:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+560(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+560(%rip)
2:
  pushq $70
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventRecord
  .p2align 4
  .type cudaEventRecord, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventRecord
#endif
cudaEventRecord:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+568(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+568(%rip)
2:
  pushq $71
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventRecordWithFlags
  .p2align 4
  .type cudaEventRecordWithFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventRecordWithFlags
#endif
cudaEventRecordWithFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+576(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+576(%rip)
2:
  pushq $72
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventRecordWithFlags_ptsz
  .p2align 4
  .type cudaEventRecordWithFlags_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventRecordWithFlags_ptsz
#endif
cudaEventRecordWithFlags_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+584(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+584(%rip)
2:
  pushq $73
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventRecord_ptsz
  .p2align 4
  .type cudaEventRecord_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventRecord_ptsz
#endif
cudaEventRecord_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+592(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+592(%rip)
2:
  pushq $74
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaEventSynchronize
  .p2align 4
  .type cudaEventSynchronize, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaEventSynchronize
#endif
cudaEventSynchronize:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+600(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+600(%rip)
2:
  pushq $75
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaExternalMemoryGetMappedBuffer
  .p2align 4
  .type cudaExternalMemoryGetMappedBuffer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaExternalMemoryGetMappedBuffer
#endif
cudaExternalMemoryGetMappedBuffer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+608(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+608(%rip)
2:
  pushq $76
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaExternalMemoryGetMappedMipmappedArray
  .p2align 4
  .type cudaExternalMemoryGetMappedMipmappedArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaExternalMemoryGetMappedMipmappedArray
#endif
cudaExternalMemoryGetMappedMipmappedArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+616(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+616(%rip)
2:
  pushq $77
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFree
  .p2align 4
  .type cudaFree, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFree
#endif
cudaFree:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+624(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+624(%rip)
2:
  pushq $78
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFreeArray
  .p2align 4
  .type cudaFreeArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFreeArray
#endif
cudaFreeArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+632(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+632(%rip)
2:
  pushq $79
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFreeAsync
  .p2align 4
  .type cudaFreeAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFreeAsync
#endif
cudaFreeAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+640(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+640(%rip)
2:
  pushq $80
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFreeAsync_ptsz
  .p2align 4
  .type cudaFreeAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFreeAsync_ptsz
#endif
cudaFreeAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+648(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+648(%rip)
2:
  pushq $81
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFreeHost
  .p2align 4
  .type cudaFreeHost, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFreeHost
#endif
cudaFreeHost:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+656(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+656(%rip)
2:
  pushq $82
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFreeMipmappedArray
  .p2align 4
  .type cudaFreeMipmappedArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFreeMipmappedArray
#endif
cudaFreeMipmappedArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+664(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+664(%rip)
2:
  pushq $83
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFuncGetAttributes
  .p2align 4
  .type cudaFuncGetAttributes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFuncGetAttributes
#endif
cudaFuncGetAttributes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+672(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+672(%rip)
2:
  pushq $84
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFuncGetName
  .p2align 4
  .type cudaFuncGetName, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFuncGetName
#endif
cudaFuncGetName:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+680(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+680(%rip)
2:
  pushq $85
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFuncGetParamInfo
  .p2align 4
  .type cudaFuncGetParamInfo, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFuncGetParamInfo
#endif
cudaFuncGetParamInfo:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+688(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+688(%rip)
2:
  pushq $86
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFuncSetAttribute
  .p2align 4
  .type cudaFuncSetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFuncSetAttribute
#endif
cudaFuncSetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+696(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+696(%rip)
2:
  pushq $87
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFuncSetCacheConfig
  .p2align 4
  .type cudaFuncSetCacheConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFuncSetCacheConfig
#endif
cudaFuncSetCacheConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+704(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+704(%rip)
2:
  pushq $88
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaFuncSetSharedMemConfig
  .p2align 4
  .type cudaFuncSetSharedMemConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaFuncSetSharedMemConfig
#endif
cudaFuncSetSharedMemConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+712(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+712(%rip)
2:
  pushq $89
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLGetDevices
  .p2align 4
  .type cudaGLGetDevices, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLGetDevices
#endif
cudaGLGetDevices:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+720(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+720(%rip)
2:
  pushq $90
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLMapBufferObject
  .p2align 4
  .type cudaGLMapBufferObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLMapBufferObject
#endif
cudaGLMapBufferObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+728(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+728(%rip)
2:
  pushq $91
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLMapBufferObjectAsync
  .p2align 4
  .type cudaGLMapBufferObjectAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLMapBufferObjectAsync
#endif
cudaGLMapBufferObjectAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+736(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+736(%rip)
2:
  pushq $92
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLRegisterBufferObject
  .p2align 4
  .type cudaGLRegisterBufferObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLRegisterBufferObject
#endif
cudaGLRegisterBufferObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+744(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+744(%rip)
2:
  pushq $93
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLSetBufferObjectMapFlags
  .p2align 4
  .type cudaGLSetBufferObjectMapFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLSetBufferObjectMapFlags
#endif
cudaGLSetBufferObjectMapFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+752(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+752(%rip)
2:
  pushq $94
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLSetGLDevice
  .p2align 4
  .type cudaGLSetGLDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLSetGLDevice
#endif
cudaGLSetGLDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+760(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+760(%rip)
2:
  pushq $95
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLUnmapBufferObject
  .p2align 4
  .type cudaGLUnmapBufferObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLUnmapBufferObject
#endif
cudaGLUnmapBufferObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+768(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+768(%rip)
2:
  pushq $96
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLUnmapBufferObjectAsync
  .p2align 4
  .type cudaGLUnmapBufferObjectAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLUnmapBufferObjectAsync
#endif
cudaGLUnmapBufferObjectAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+776(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+776(%rip)
2:
  pushq $97
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGLUnregisterBufferObject
  .p2align 4
  .type cudaGLUnregisterBufferObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGLUnregisterBufferObject
#endif
cudaGLUnregisterBufferObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+784(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+784(%rip)
2:
  pushq $98
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetChannelDesc
  .p2align 4
  .type cudaGetChannelDesc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetChannelDesc
#endif
cudaGetChannelDesc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+792(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+792(%rip)
2:
  pushq $99
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDevice
  .p2align 4
  .type cudaGetDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDevice
#endif
cudaGetDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+800(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+800(%rip)
2:
  pushq $100
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDeviceCount
  .p2align 4
  .type cudaGetDeviceCount, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDeviceCount
#endif
cudaGetDeviceCount:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+808(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+808(%rip)
2:
  pushq $101
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDeviceFlags
  .p2align 4
  .type cudaGetDeviceFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDeviceFlags
#endif
cudaGetDeviceFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+816(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+816(%rip)
2:
  pushq $102
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDeviceProperties
  .p2align 4
  .type cudaGetDeviceProperties, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDeviceProperties
#endif
cudaGetDeviceProperties:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+824(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+824(%rip)
2:
  pushq $103
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDeviceProperties_v2
  .p2align 4
  .type cudaGetDeviceProperties_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDeviceProperties_v2
#endif
cudaGetDeviceProperties_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+832(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+832(%rip)
2:
  pushq $104
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDriverEntryPoint
  .p2align 4
  .type cudaGetDriverEntryPoint, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDriverEntryPoint
#endif
cudaGetDriverEntryPoint:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+840(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+840(%rip)
2:
  pushq $105
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDriverEntryPointByVersion
  .p2align 4
  .type cudaGetDriverEntryPointByVersion, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDriverEntryPointByVersion
#endif
cudaGetDriverEntryPointByVersion:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+848(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+848(%rip)
2:
  pushq $106
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDriverEntryPointByVersion_ptsz
  .p2align 4
  .type cudaGetDriverEntryPointByVersion_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDriverEntryPointByVersion_ptsz
#endif
cudaGetDriverEntryPointByVersion_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+856(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+856(%rip)
2:
  pushq $107
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetDriverEntryPoint_ptsz
  .p2align 4
  .type cudaGetDriverEntryPoint_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetDriverEntryPoint_ptsz
#endif
cudaGetDriverEntryPoint_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+864(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+864(%rip)
2:
  pushq $108
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetErrorName
  .p2align 4
  .type cudaGetErrorName, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetErrorName
#endif
cudaGetErrorName:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+872(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+872(%rip)
2:
  pushq $109
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetErrorString
  .p2align 4
  .type cudaGetErrorString, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetErrorString
#endif
cudaGetErrorString:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+880(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+880(%rip)
2:
  pushq $110
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetExportTable
  .p2align 4
  .type cudaGetExportTable, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetExportTable
#endif
cudaGetExportTable:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+888(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+888(%rip)
2:
  pushq $111
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetFuncBySymbol
  .p2align 4
  .type cudaGetFuncBySymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetFuncBySymbol
#endif
cudaGetFuncBySymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+896(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+896(%rip)
2:
  pushq $112
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetKernel
  .p2align 4
  .type cudaGetKernel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetKernel
#endif
cudaGetKernel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+904(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+904(%rip)
2:
  pushq $113
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetLastError
  .p2align 4
  .type cudaGetLastError, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetLastError
#endif
cudaGetLastError:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+912(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+912(%rip)
2:
  pushq $114
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetMipmappedArrayLevel
  .p2align 4
  .type cudaGetMipmappedArrayLevel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetMipmappedArrayLevel
#endif
cudaGetMipmappedArrayLevel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+920(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+920(%rip)
2:
  pushq $115
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetSurfaceObjectResourceDesc
  .p2align 4
  .type cudaGetSurfaceObjectResourceDesc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetSurfaceObjectResourceDesc
#endif
cudaGetSurfaceObjectResourceDesc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+928(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+928(%rip)
2:
  pushq $116
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetSymbolAddress
  .p2align 4
  .type cudaGetSymbolAddress, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetSymbolAddress
#endif
cudaGetSymbolAddress:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+936(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+936(%rip)
2:
  pushq $117
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetSymbolSize
  .p2align 4
  .type cudaGetSymbolSize, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetSymbolSize
#endif
cudaGetSymbolSize:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+944(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+944(%rip)
2:
  pushq $118
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetTextureObjectResourceDesc
  .p2align 4
  .type cudaGetTextureObjectResourceDesc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetTextureObjectResourceDesc
#endif
cudaGetTextureObjectResourceDesc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+952(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+952(%rip)
2:
  pushq $119
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetTextureObjectResourceViewDesc
  .p2align 4
  .type cudaGetTextureObjectResourceViewDesc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetTextureObjectResourceViewDesc
#endif
cudaGetTextureObjectResourceViewDesc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+960(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+960(%rip)
2:
  pushq $120
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGetTextureObjectTextureDesc
  .p2align 4
  .type cudaGetTextureObjectTextureDesc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGetTextureObjectTextureDesc
#endif
cudaGetTextureObjectTextureDesc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+968(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+968(%rip)
2:
  pushq $121
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddChildGraphNode
  .p2align 4
  .type cudaGraphAddChildGraphNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddChildGraphNode
#endif
cudaGraphAddChildGraphNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+976(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+976(%rip)
2:
  pushq $122
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddDependencies
  .p2align 4
  .type cudaGraphAddDependencies, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddDependencies
#endif
cudaGraphAddDependencies:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+984(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+984(%rip)
2:
  pushq $123
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddDependencies_v2
  .p2align 4
  .type cudaGraphAddDependencies_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddDependencies_v2
#endif
cudaGraphAddDependencies_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+992(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+992(%rip)
2:
  pushq $124
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddEmptyNode
  .p2align 4
  .type cudaGraphAddEmptyNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddEmptyNode
#endif
cudaGraphAddEmptyNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1000(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1000(%rip)
2:
  pushq $125
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddEventRecordNode
  .p2align 4
  .type cudaGraphAddEventRecordNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddEventRecordNode
#endif
cudaGraphAddEventRecordNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1008(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1008(%rip)
2:
  pushq $126
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddEventWaitNode
  .p2align 4
  .type cudaGraphAddEventWaitNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddEventWaitNode
#endif
cudaGraphAddEventWaitNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1016(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1016(%rip)
2:
  pushq $127
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddExternalSemaphoresSignalNode
  .p2align 4
  .type cudaGraphAddExternalSemaphoresSignalNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddExternalSemaphoresSignalNode
#endif
cudaGraphAddExternalSemaphoresSignalNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1024(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1024(%rip)
2:
  pushq $128
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddExternalSemaphoresWaitNode
  .p2align 4
  .type cudaGraphAddExternalSemaphoresWaitNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddExternalSemaphoresWaitNode
#endif
cudaGraphAddExternalSemaphoresWaitNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1032(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1032(%rip)
2:
  pushq $129
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddHostNode
  .p2align 4
  .type cudaGraphAddHostNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddHostNode
#endif
cudaGraphAddHostNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1040(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1040(%rip)
2:
  pushq $130
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddKernelNode
  .p2align 4
  .type cudaGraphAddKernelNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddKernelNode
#endif
cudaGraphAddKernelNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1048(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1048(%rip)
2:
  pushq $131
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemAllocNode
  .p2align 4
  .type cudaGraphAddMemAllocNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemAllocNode
#endif
cudaGraphAddMemAllocNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1056(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1056(%rip)
2:
  pushq $132
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemFreeNode
  .p2align 4
  .type cudaGraphAddMemFreeNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemFreeNode
#endif
cudaGraphAddMemFreeNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1064(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1064(%rip)
2:
  pushq $133
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemcpyNode
  .p2align 4
  .type cudaGraphAddMemcpyNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemcpyNode
#endif
cudaGraphAddMemcpyNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1072(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1072(%rip)
2:
  pushq $134
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemcpyNode1D
  .p2align 4
  .type cudaGraphAddMemcpyNode1D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemcpyNode1D
#endif
cudaGraphAddMemcpyNode1D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1080(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1080(%rip)
2:
  pushq $135
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemcpyNodeFromSymbol
  .p2align 4
  .type cudaGraphAddMemcpyNodeFromSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemcpyNodeFromSymbol
#endif
cudaGraphAddMemcpyNodeFromSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1088(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1088(%rip)
2:
  pushq $136
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemcpyNodeToSymbol
  .p2align 4
  .type cudaGraphAddMemcpyNodeToSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemcpyNodeToSymbol
#endif
cudaGraphAddMemcpyNodeToSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1096(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1096(%rip)
2:
  pushq $137
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddMemsetNode
  .p2align 4
  .type cudaGraphAddMemsetNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddMemsetNode
#endif
cudaGraphAddMemsetNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1104(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1104(%rip)
2:
  pushq $138
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddNode
  .p2align 4
  .type cudaGraphAddNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddNode
#endif
cudaGraphAddNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1112(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1112(%rip)
2:
  pushq $139
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphAddNode_v2
  .p2align 4
  .type cudaGraphAddNode_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphAddNode_v2
#endif
cudaGraphAddNode_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1120(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1120(%rip)
2:
  pushq $140
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphChildGraphNodeGetGraph
  .p2align 4
  .type cudaGraphChildGraphNodeGetGraph, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphChildGraphNodeGetGraph
#endif
cudaGraphChildGraphNodeGetGraph:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1128(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1128(%rip)
2:
  pushq $141
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphClone
  .p2align 4
  .type cudaGraphClone, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphClone
#endif
cudaGraphClone:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1136(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1136(%rip)
2:
  pushq $142
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphConditionalHandleCreate
  .p2align 4
  .type cudaGraphConditionalHandleCreate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphConditionalHandleCreate
#endif
cudaGraphConditionalHandleCreate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1144(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1144(%rip)
2:
  pushq $143
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphCreate
  .p2align 4
  .type cudaGraphCreate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphCreate
#endif
cudaGraphCreate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1152(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1152(%rip)
2:
  pushq $144
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphDebugDotPrint
  .p2align 4
  .type cudaGraphDebugDotPrint, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphDebugDotPrint
#endif
cudaGraphDebugDotPrint:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1160(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1160(%rip)
2:
  pushq $145
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphDestroy
  .p2align 4
  .type cudaGraphDestroy, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphDestroy
#endif
cudaGraphDestroy:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1168(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1168(%rip)
2:
  pushq $146
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphDestroyNode
  .p2align 4
  .type cudaGraphDestroyNode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphDestroyNode
#endif
cudaGraphDestroyNode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1176(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1176(%rip)
2:
  pushq $147
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphEventRecordNodeGetEvent
  .p2align 4
  .type cudaGraphEventRecordNodeGetEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphEventRecordNodeGetEvent
#endif
cudaGraphEventRecordNodeGetEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1184(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1184(%rip)
2:
  pushq $148
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphEventRecordNodeSetEvent
  .p2align 4
  .type cudaGraphEventRecordNodeSetEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphEventRecordNodeSetEvent
#endif
cudaGraphEventRecordNodeSetEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1192(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1192(%rip)
2:
  pushq $149
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphEventWaitNodeGetEvent
  .p2align 4
  .type cudaGraphEventWaitNodeGetEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphEventWaitNodeGetEvent
#endif
cudaGraphEventWaitNodeGetEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1200(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1200(%rip)
2:
  pushq $150
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphEventWaitNodeSetEvent
  .p2align 4
  .type cudaGraphEventWaitNodeSetEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphEventWaitNodeSetEvent
#endif
cudaGraphEventWaitNodeSetEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1208(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1208(%rip)
2:
  pushq $151
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecChildGraphNodeSetParams
  .p2align 4
  .type cudaGraphExecChildGraphNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecChildGraphNodeSetParams
#endif
cudaGraphExecChildGraphNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1216(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1216(%rip)
2:
  pushq $152
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecDestroy
  .p2align 4
  .type cudaGraphExecDestroy, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecDestroy
#endif
cudaGraphExecDestroy:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1224(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1224(%rip)
2:
  pushq $153
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecEventRecordNodeSetEvent
  .p2align 4
  .type cudaGraphExecEventRecordNodeSetEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecEventRecordNodeSetEvent
#endif
cudaGraphExecEventRecordNodeSetEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1232(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1232(%rip)
2:
  pushq $154
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecEventWaitNodeSetEvent
  .p2align 4
  .type cudaGraphExecEventWaitNodeSetEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecEventWaitNodeSetEvent
#endif
cudaGraphExecEventWaitNodeSetEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1240(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1240(%rip)
2:
  pushq $155
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecExternalSemaphoresSignalNodeSetParams
  .p2align 4
  .type cudaGraphExecExternalSemaphoresSignalNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecExternalSemaphoresSignalNodeSetParams
#endif
cudaGraphExecExternalSemaphoresSignalNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1248(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1248(%rip)
2:
  pushq $156
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecExternalSemaphoresWaitNodeSetParams
  .p2align 4
  .type cudaGraphExecExternalSemaphoresWaitNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecExternalSemaphoresWaitNodeSetParams
#endif
cudaGraphExecExternalSemaphoresWaitNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1256(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1256(%rip)
2:
  pushq $157
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecGetFlags
  .p2align 4
  .type cudaGraphExecGetFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecGetFlags
#endif
cudaGraphExecGetFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1264(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1264(%rip)
2:
  pushq $158
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecHostNodeSetParams
  .p2align 4
  .type cudaGraphExecHostNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecHostNodeSetParams
#endif
cudaGraphExecHostNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1272(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1272(%rip)
2:
  pushq $159
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecKernelNodeSetParams
  .p2align 4
  .type cudaGraphExecKernelNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecKernelNodeSetParams
#endif
cudaGraphExecKernelNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1280(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1280(%rip)
2:
  pushq $160
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecMemcpyNodeSetParams
  .p2align 4
  .type cudaGraphExecMemcpyNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecMemcpyNodeSetParams
#endif
cudaGraphExecMemcpyNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1288(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1288(%rip)
2:
  pushq $161
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecMemcpyNodeSetParams1D
  .p2align 4
  .type cudaGraphExecMemcpyNodeSetParams1D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecMemcpyNodeSetParams1D
#endif
cudaGraphExecMemcpyNodeSetParams1D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1296(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1296(%rip)
2:
  pushq $162
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecMemcpyNodeSetParamsFromSymbol
  .p2align 4
  .type cudaGraphExecMemcpyNodeSetParamsFromSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecMemcpyNodeSetParamsFromSymbol
#endif
cudaGraphExecMemcpyNodeSetParamsFromSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1304(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1304(%rip)
2:
  pushq $163
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecMemcpyNodeSetParamsToSymbol
  .p2align 4
  .type cudaGraphExecMemcpyNodeSetParamsToSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecMemcpyNodeSetParamsToSymbol
#endif
cudaGraphExecMemcpyNodeSetParamsToSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1312(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1312(%rip)
2:
  pushq $164
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecMemsetNodeSetParams
  .p2align 4
  .type cudaGraphExecMemsetNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecMemsetNodeSetParams
#endif
cudaGraphExecMemsetNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1320(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1320(%rip)
2:
  pushq $165
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecNodeSetParams
  .p2align 4
  .type cudaGraphExecNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecNodeSetParams
#endif
cudaGraphExecNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1328(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1328(%rip)
2:
  pushq $166
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExecUpdate
  .p2align 4
  .type cudaGraphExecUpdate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExecUpdate
#endif
cudaGraphExecUpdate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1336(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1336(%rip)
2:
  pushq $167
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExternalSemaphoresSignalNodeGetParams
  .p2align 4
  .type cudaGraphExternalSemaphoresSignalNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExternalSemaphoresSignalNodeGetParams
#endif
cudaGraphExternalSemaphoresSignalNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1344(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1344(%rip)
2:
  pushq $168
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExternalSemaphoresSignalNodeSetParams
  .p2align 4
  .type cudaGraphExternalSemaphoresSignalNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExternalSemaphoresSignalNodeSetParams
#endif
cudaGraphExternalSemaphoresSignalNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1352(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1352(%rip)
2:
  pushq $169
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExternalSemaphoresWaitNodeGetParams
  .p2align 4
  .type cudaGraphExternalSemaphoresWaitNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExternalSemaphoresWaitNodeGetParams
#endif
cudaGraphExternalSemaphoresWaitNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1360(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1360(%rip)
2:
  pushq $170
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphExternalSemaphoresWaitNodeSetParams
  .p2align 4
  .type cudaGraphExternalSemaphoresWaitNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphExternalSemaphoresWaitNodeSetParams
#endif
cudaGraphExternalSemaphoresWaitNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1368(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1368(%rip)
2:
  pushq $171
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphGetEdges
  .p2align 4
  .type cudaGraphGetEdges, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphGetEdges
#endif
cudaGraphGetEdges:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1376(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1376(%rip)
2:
  pushq $172
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphGetEdges_v2
  .p2align 4
  .type cudaGraphGetEdges_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphGetEdges_v2
#endif
cudaGraphGetEdges_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1384(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1384(%rip)
2:
  pushq $173
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphGetNodes
  .p2align 4
  .type cudaGraphGetNodes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphGetNodes
#endif
cudaGraphGetNodes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1392(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1392(%rip)
2:
  pushq $174
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphGetRootNodes
  .p2align 4
  .type cudaGraphGetRootNodes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphGetRootNodes
#endif
cudaGraphGetRootNodes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1400(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1400(%rip)
2:
  pushq $175
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphHostNodeGetParams
  .p2align 4
  .type cudaGraphHostNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphHostNodeGetParams
#endif
cudaGraphHostNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1408(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1408(%rip)
2:
  pushq $176
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphHostNodeSetParams
  .p2align 4
  .type cudaGraphHostNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphHostNodeSetParams
#endif
cudaGraphHostNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1416(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1416(%rip)
2:
  pushq $177
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphInstantiate
  .p2align 4
  .type cudaGraphInstantiate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphInstantiate
#endif
cudaGraphInstantiate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1424(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1424(%rip)
2:
  pushq $178
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphInstantiateWithFlags
  .p2align 4
  .type cudaGraphInstantiateWithFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphInstantiateWithFlags
#endif
cudaGraphInstantiateWithFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1432(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1432(%rip)
2:
  pushq $179
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphInstantiateWithParams
  .p2align 4
  .type cudaGraphInstantiateWithParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphInstantiateWithParams
#endif
cudaGraphInstantiateWithParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1440(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1440(%rip)
2:
  pushq $180
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphInstantiateWithParams_ptsz
  .p2align 4
  .type cudaGraphInstantiateWithParams_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphInstantiateWithParams_ptsz
#endif
cudaGraphInstantiateWithParams_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1448(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1448(%rip)
2:
  pushq $181
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphKernelNodeCopyAttributes
  .p2align 4
  .type cudaGraphKernelNodeCopyAttributes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphKernelNodeCopyAttributes
#endif
cudaGraphKernelNodeCopyAttributes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1456(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1456(%rip)
2:
  pushq $182
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphKernelNodeGetAttribute
  .p2align 4
  .type cudaGraphKernelNodeGetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphKernelNodeGetAttribute
#endif
cudaGraphKernelNodeGetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1464(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1464(%rip)
2:
  pushq $183
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphKernelNodeGetParams
  .p2align 4
  .type cudaGraphKernelNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphKernelNodeGetParams
#endif
cudaGraphKernelNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1472(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1472(%rip)
2:
  pushq $184
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphKernelNodeSetAttribute
  .p2align 4
  .type cudaGraphKernelNodeSetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphKernelNodeSetAttribute
#endif
cudaGraphKernelNodeSetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1480(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1480(%rip)
2:
  pushq $185
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphKernelNodeSetParams
  .p2align 4
  .type cudaGraphKernelNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphKernelNodeSetParams
#endif
cudaGraphKernelNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1488(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1488(%rip)
2:
  pushq $186
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphLaunch
  .p2align 4
  .type cudaGraphLaunch, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphLaunch
#endif
cudaGraphLaunch:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1496(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1496(%rip)
2:
  pushq $187
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphLaunch_ptsz
  .p2align 4
  .type cudaGraphLaunch_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphLaunch_ptsz
#endif
cudaGraphLaunch_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1504(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1504(%rip)
2:
  pushq $188
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemAllocNodeGetParams
  .p2align 4
  .type cudaGraphMemAllocNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemAllocNodeGetParams
#endif
cudaGraphMemAllocNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1512(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1512(%rip)
2:
  pushq $189
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemFreeNodeGetParams
  .p2align 4
  .type cudaGraphMemFreeNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemFreeNodeGetParams
#endif
cudaGraphMemFreeNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1520(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1520(%rip)
2:
  pushq $190
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemcpyNodeGetParams
  .p2align 4
  .type cudaGraphMemcpyNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemcpyNodeGetParams
#endif
cudaGraphMemcpyNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1528(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1528(%rip)
2:
  pushq $191
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemcpyNodeSetParams
  .p2align 4
  .type cudaGraphMemcpyNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemcpyNodeSetParams
#endif
cudaGraphMemcpyNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1536(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1536(%rip)
2:
  pushq $192
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemcpyNodeSetParams1D
  .p2align 4
  .type cudaGraphMemcpyNodeSetParams1D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemcpyNodeSetParams1D
#endif
cudaGraphMemcpyNodeSetParams1D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1544(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1544(%rip)
2:
  pushq $193
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemcpyNodeSetParamsFromSymbol
  .p2align 4
  .type cudaGraphMemcpyNodeSetParamsFromSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemcpyNodeSetParamsFromSymbol
#endif
cudaGraphMemcpyNodeSetParamsFromSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1552(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1552(%rip)
2:
  pushq $194
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemcpyNodeSetParamsToSymbol
  .p2align 4
  .type cudaGraphMemcpyNodeSetParamsToSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemcpyNodeSetParamsToSymbol
#endif
cudaGraphMemcpyNodeSetParamsToSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1560(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1560(%rip)
2:
  pushq $195
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemsetNodeGetParams
  .p2align 4
  .type cudaGraphMemsetNodeGetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemsetNodeGetParams
#endif
cudaGraphMemsetNodeGetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1568(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1568(%rip)
2:
  pushq $196
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphMemsetNodeSetParams
  .p2align 4
  .type cudaGraphMemsetNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphMemsetNodeSetParams
#endif
cudaGraphMemsetNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1576(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1576(%rip)
2:
  pushq $197
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeFindInClone
  .p2align 4
  .type cudaGraphNodeFindInClone, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeFindInClone
#endif
cudaGraphNodeFindInClone:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1584(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1584(%rip)
2:
  pushq $198
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeGetDependencies
  .p2align 4
  .type cudaGraphNodeGetDependencies, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeGetDependencies
#endif
cudaGraphNodeGetDependencies:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1592(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1592(%rip)
2:
  pushq $199
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeGetDependencies_v2
  .p2align 4
  .type cudaGraphNodeGetDependencies_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeGetDependencies_v2
#endif
cudaGraphNodeGetDependencies_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1600(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1600(%rip)
2:
  pushq $200
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeGetDependentNodes
  .p2align 4
  .type cudaGraphNodeGetDependentNodes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeGetDependentNodes
#endif
cudaGraphNodeGetDependentNodes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1608(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1608(%rip)
2:
  pushq $201
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeGetDependentNodes_v2
  .p2align 4
  .type cudaGraphNodeGetDependentNodes_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeGetDependentNodes_v2
#endif
cudaGraphNodeGetDependentNodes_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1616(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1616(%rip)
2:
  pushq $202
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeGetEnabled
  .p2align 4
  .type cudaGraphNodeGetEnabled, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeGetEnabled
#endif
cudaGraphNodeGetEnabled:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1624(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1624(%rip)
2:
  pushq $203
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeGetType
  .p2align 4
  .type cudaGraphNodeGetType, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeGetType
#endif
cudaGraphNodeGetType:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1632(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1632(%rip)
2:
  pushq $204
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeSetEnabled
  .p2align 4
  .type cudaGraphNodeSetEnabled, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeSetEnabled
#endif
cudaGraphNodeSetEnabled:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1640(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1640(%rip)
2:
  pushq $205
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphNodeSetParams
  .p2align 4
  .type cudaGraphNodeSetParams, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphNodeSetParams
#endif
cudaGraphNodeSetParams:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1648(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1648(%rip)
2:
  pushq $206
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphReleaseUserObject
  .p2align 4
  .type cudaGraphReleaseUserObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphReleaseUserObject
#endif
cudaGraphReleaseUserObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1656(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1656(%rip)
2:
  pushq $207
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphRemoveDependencies
  .p2align 4
  .type cudaGraphRemoveDependencies, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphRemoveDependencies
#endif
cudaGraphRemoveDependencies:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1664(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1664(%rip)
2:
  pushq $208
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphRemoveDependencies_v2
  .p2align 4
  .type cudaGraphRemoveDependencies_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphRemoveDependencies_v2
#endif
cudaGraphRemoveDependencies_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1672(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1672(%rip)
2:
  pushq $209
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphRetainUserObject
  .p2align 4
  .type cudaGraphRetainUserObject, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphRetainUserObject
#endif
cudaGraphRetainUserObject:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1680(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1680(%rip)
2:
  pushq $210
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphUpload
  .p2align 4
  .type cudaGraphUpload, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphUpload
#endif
cudaGraphUpload:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1688(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1688(%rip)
2:
  pushq $211
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphUpload_ptsz
  .p2align 4
  .type cudaGraphUpload_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphUpload_ptsz
#endif
cudaGraphUpload_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1696(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1696(%rip)
2:
  pushq $212
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsEGLRegisterImage
  .p2align 4
  .type cudaGraphicsEGLRegisterImage, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsEGLRegisterImage
#endif
cudaGraphicsEGLRegisterImage:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1704(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1704(%rip)
2:
  pushq $213
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsGLRegisterBuffer
  .p2align 4
  .type cudaGraphicsGLRegisterBuffer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsGLRegisterBuffer
#endif
cudaGraphicsGLRegisterBuffer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1712(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1712(%rip)
2:
  pushq $214
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsGLRegisterImage
  .p2align 4
  .type cudaGraphicsGLRegisterImage, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsGLRegisterImage
#endif
cudaGraphicsGLRegisterImage:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1720(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1720(%rip)
2:
  pushq $215
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsMapResources
  .p2align 4
  .type cudaGraphicsMapResources, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsMapResources
#endif
cudaGraphicsMapResources:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1728(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1728(%rip)
2:
  pushq $216
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsResourceGetMappedEglFrame
  .p2align 4
  .type cudaGraphicsResourceGetMappedEglFrame, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsResourceGetMappedEglFrame
#endif
cudaGraphicsResourceGetMappedEglFrame:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1736(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1736(%rip)
2:
  pushq $217
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsResourceGetMappedMipmappedArray
  .p2align 4
  .type cudaGraphicsResourceGetMappedMipmappedArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsResourceGetMappedMipmappedArray
#endif
cudaGraphicsResourceGetMappedMipmappedArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1744(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1744(%rip)
2:
  pushq $218
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsResourceGetMappedPointer
  .p2align 4
  .type cudaGraphicsResourceGetMappedPointer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsResourceGetMappedPointer
#endif
cudaGraphicsResourceGetMappedPointer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1752(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1752(%rip)
2:
  pushq $219
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsResourceSetMapFlags
  .p2align 4
  .type cudaGraphicsResourceSetMapFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsResourceSetMapFlags
#endif
cudaGraphicsResourceSetMapFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1760(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1760(%rip)
2:
  pushq $220
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsSubResourceGetMappedArray
  .p2align 4
  .type cudaGraphicsSubResourceGetMappedArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsSubResourceGetMappedArray
#endif
cudaGraphicsSubResourceGetMappedArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1768(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1768(%rip)
2:
  pushq $221
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsUnmapResources
  .p2align 4
  .type cudaGraphicsUnmapResources, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsUnmapResources
#endif
cudaGraphicsUnmapResources:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1776(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1776(%rip)
2:
  pushq $222
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsUnregisterResource
  .p2align 4
  .type cudaGraphicsUnregisterResource, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsUnregisterResource
#endif
cudaGraphicsUnregisterResource:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1784(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1784(%rip)
2:
  pushq $223
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsVDPAURegisterOutputSurface
  .p2align 4
  .type cudaGraphicsVDPAURegisterOutputSurface, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsVDPAURegisterOutputSurface
#endif
cudaGraphicsVDPAURegisterOutputSurface:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1792(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1792(%rip)
2:
  pushq $224
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaGraphicsVDPAURegisterVideoSurface
  .p2align 4
  .type cudaGraphicsVDPAURegisterVideoSurface, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaGraphicsVDPAURegisterVideoSurface
#endif
cudaGraphicsVDPAURegisterVideoSurface:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1800(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1800(%rip)
2:
  pushq $225
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaHostAlloc
  .p2align 4
  .type cudaHostAlloc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaHostAlloc
#endif
cudaHostAlloc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1808(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1808(%rip)
2:
  pushq $226
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaHostGetDevicePointer
  .p2align 4
  .type cudaHostGetDevicePointer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaHostGetDevicePointer
#endif
cudaHostGetDevicePointer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1816(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1816(%rip)
2:
  pushq $227
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaHostGetFlags
  .p2align 4
  .type cudaHostGetFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaHostGetFlags
#endif
cudaHostGetFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1824(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1824(%rip)
2:
  pushq $228
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaHostRegister
  .p2align 4
  .type cudaHostRegister, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaHostRegister
#endif
cudaHostRegister:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1832(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1832(%rip)
2:
  pushq $229
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaHostUnregister
  .p2align 4
  .type cudaHostUnregister, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaHostUnregister
#endif
cudaHostUnregister:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1840(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1840(%rip)
2:
  pushq $230
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaImportExternalMemory
  .p2align 4
  .type cudaImportExternalMemory, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaImportExternalMemory
#endif
cudaImportExternalMemory:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1848(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1848(%rip)
2:
  pushq $231
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaImportExternalSemaphore
  .p2align 4
  .type cudaImportExternalSemaphore, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaImportExternalSemaphore
#endif
cudaImportExternalSemaphore:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1856(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1856(%rip)
2:
  pushq $232
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaInitDevice
  .p2align 4
  .type cudaInitDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaInitDevice
#endif
cudaInitDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1864(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1864(%rip)
2:
  pushq $233
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaIpcCloseMemHandle
  .p2align 4
  .type cudaIpcCloseMemHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaIpcCloseMemHandle
#endif
cudaIpcCloseMemHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1872(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1872(%rip)
2:
  pushq $234
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaIpcGetEventHandle
  .p2align 4
  .type cudaIpcGetEventHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaIpcGetEventHandle
#endif
cudaIpcGetEventHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1880(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1880(%rip)
2:
  pushq $235
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaIpcGetMemHandle
  .p2align 4
  .type cudaIpcGetMemHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaIpcGetMemHandle
#endif
cudaIpcGetMemHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1888(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1888(%rip)
2:
  pushq $236
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaIpcOpenEventHandle
  .p2align 4
  .type cudaIpcOpenEventHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaIpcOpenEventHandle
#endif
cudaIpcOpenEventHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1896(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1896(%rip)
2:
  pushq $237
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaIpcOpenMemHandle
  .p2align 4
  .type cudaIpcOpenMemHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaIpcOpenMemHandle
#endif
cudaIpcOpenMemHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1904(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1904(%rip)
2:
  pushq $238
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaKernelSetAttributeForDevice
  .p2align 4
  .type cudaKernelSetAttributeForDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaKernelSetAttributeForDevice
#endif
cudaKernelSetAttributeForDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1912(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1912(%rip)
2:
  pushq $239
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchCooperativeKernel
  .p2align 4
  .type cudaLaunchCooperativeKernel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchCooperativeKernel
#endif
cudaLaunchCooperativeKernel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1920(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1920(%rip)
2:
  pushq $240
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchCooperativeKernelMultiDevice
  .p2align 4
  .type cudaLaunchCooperativeKernelMultiDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchCooperativeKernelMultiDevice
#endif
cudaLaunchCooperativeKernelMultiDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1928(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1928(%rip)
2:
  pushq $241
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchCooperativeKernel_ptsz
  .p2align 4
  .type cudaLaunchCooperativeKernel_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchCooperativeKernel_ptsz
#endif
cudaLaunchCooperativeKernel_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1936(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1936(%rip)
2:
  pushq $242
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchHostFunc
  .p2align 4
  .type cudaLaunchHostFunc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchHostFunc
#endif
cudaLaunchHostFunc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1944(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1944(%rip)
2:
  pushq $243
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchHostFunc_ptsz
  .p2align 4
  .type cudaLaunchHostFunc_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchHostFunc_ptsz
#endif
cudaLaunchHostFunc_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1952(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1952(%rip)
2:
  pushq $244
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchKernel
  .p2align 4
  .type cudaLaunchKernel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchKernel
#endif
cudaLaunchKernel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1960(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1960(%rip)
2:
  pushq $245
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchKernelExC
  .p2align 4
  .type cudaLaunchKernelExC, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchKernelExC
#endif
cudaLaunchKernelExC:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1968(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1968(%rip)
2:
  pushq $246
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchKernelExC_ptsz
  .p2align 4
  .type cudaLaunchKernelExC_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchKernelExC_ptsz
#endif
cudaLaunchKernelExC_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1976(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1976(%rip)
2:
  pushq $247
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLaunchKernel_ptsz
  .p2align 4
  .type cudaLaunchKernel_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLaunchKernel_ptsz
#endif
cudaLaunchKernel_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1984(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1984(%rip)
2:
  pushq $248
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryEnumerateKernels
  .p2align 4
  .type cudaLibraryEnumerateKernels, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryEnumerateKernels
#endif
cudaLibraryEnumerateKernels:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+1992(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+1992(%rip)
2:
  pushq $249
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryGetGlobal
  .p2align 4
  .type cudaLibraryGetGlobal, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryGetGlobal
#endif
cudaLibraryGetGlobal:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2000(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2000(%rip)
2:
  pushq $250
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryGetKernel
  .p2align 4
  .type cudaLibraryGetKernel, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryGetKernel
#endif
cudaLibraryGetKernel:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2008(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2008(%rip)
2:
  pushq $251
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryGetKernelCount
  .p2align 4
  .type cudaLibraryGetKernelCount, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryGetKernelCount
#endif
cudaLibraryGetKernelCount:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2016(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2016(%rip)
2:
  pushq $252
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryGetManaged
  .p2align 4
  .type cudaLibraryGetManaged, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryGetManaged
#endif
cudaLibraryGetManaged:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2024(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2024(%rip)
2:
  pushq $253
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryGetUnifiedFunction
  .p2align 4
  .type cudaLibraryGetUnifiedFunction, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryGetUnifiedFunction
#endif
cudaLibraryGetUnifiedFunction:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2032(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2032(%rip)
2:
  pushq $254
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryLoadData
  .p2align 4
  .type cudaLibraryLoadData, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryLoadData
#endif
cudaLibraryLoadData:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2040(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2040(%rip)
2:
  pushq $255
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryLoadFromFile
  .p2align 4
  .type cudaLibraryLoadFromFile, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryLoadFromFile
#endif
cudaLibraryLoadFromFile:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2048(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2048(%rip)
2:
  pushq $256
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaLibraryUnload
  .p2align 4
  .type cudaLibraryUnload, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaLibraryUnload
#endif
cudaLibraryUnload:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2056(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2056(%rip)
2:
  pushq $257
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMalloc
  .p2align 4
  .type cudaMalloc, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMalloc
#endif
cudaMalloc:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2064(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2064(%rip)
2:
  pushq $258
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMalloc3D
  .p2align 4
  .type cudaMalloc3D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMalloc3D
#endif
cudaMalloc3D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2072(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2072(%rip)
2:
  pushq $259
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMalloc3DArray
  .p2align 4
  .type cudaMalloc3DArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMalloc3DArray
#endif
cudaMalloc3DArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2080(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2080(%rip)
2:
  pushq $260
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocArray
  .p2align 4
  .type cudaMallocArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocArray
#endif
cudaMallocArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2088(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2088(%rip)
2:
  pushq $261
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocAsync
  .p2align 4
  .type cudaMallocAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocAsync
#endif
cudaMallocAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2096(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2096(%rip)
2:
  pushq $262
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocAsync_ptsz
  .p2align 4
  .type cudaMallocAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocAsync_ptsz
#endif
cudaMallocAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2104(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2104(%rip)
2:
  pushq $263
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocFromPoolAsync
  .p2align 4
  .type cudaMallocFromPoolAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocFromPoolAsync
#endif
cudaMallocFromPoolAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2112(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2112(%rip)
2:
  pushq $264
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocFromPoolAsync_ptsz
  .p2align 4
  .type cudaMallocFromPoolAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocFromPoolAsync_ptsz
#endif
cudaMallocFromPoolAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2120(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2120(%rip)
2:
  pushq $265
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocHost
  .p2align 4
  .type cudaMallocHost, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocHost
#endif
cudaMallocHost:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2128(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2128(%rip)
2:
  pushq $266
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocManaged
  .p2align 4
  .type cudaMallocManaged, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocManaged
#endif
cudaMallocManaged:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2136(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2136(%rip)
2:
  pushq $267
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocMipmappedArray
  .p2align 4
  .type cudaMallocMipmappedArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocMipmappedArray
#endif
cudaMallocMipmappedArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2144(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2144(%rip)
2:
  pushq $268
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMallocPitch
  .p2align 4
  .type cudaMallocPitch, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMallocPitch
#endif
cudaMallocPitch:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2152(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2152(%rip)
2:
  pushq $269
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemAdvise
  .p2align 4
  .type cudaMemAdvise, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemAdvise
#endif
cudaMemAdvise:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2160(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2160(%rip)
2:
  pushq $270
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemAdvise_v2
  .p2align 4
  .type cudaMemAdvise_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemAdvise_v2
#endif
cudaMemAdvise_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2168(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2168(%rip)
2:
  pushq $271
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemGetInfo
  .p2align 4
  .type cudaMemGetInfo, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemGetInfo
#endif
cudaMemGetInfo:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2176(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2176(%rip)
2:
  pushq $272
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolCreate
  .p2align 4
  .type cudaMemPoolCreate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolCreate
#endif
cudaMemPoolCreate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2184(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2184(%rip)
2:
  pushq $273
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolDestroy
  .p2align 4
  .type cudaMemPoolDestroy, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolDestroy
#endif
cudaMemPoolDestroy:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2192(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2192(%rip)
2:
  pushq $274
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolExportPointer
  .p2align 4
  .type cudaMemPoolExportPointer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolExportPointer
#endif
cudaMemPoolExportPointer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2200(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2200(%rip)
2:
  pushq $275
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolExportToShareableHandle
  .p2align 4
  .type cudaMemPoolExportToShareableHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolExportToShareableHandle
#endif
cudaMemPoolExportToShareableHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2208(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2208(%rip)
2:
  pushq $276
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolGetAccess
  .p2align 4
  .type cudaMemPoolGetAccess, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolGetAccess
#endif
cudaMemPoolGetAccess:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2216(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2216(%rip)
2:
  pushq $277
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolGetAttribute
  .p2align 4
  .type cudaMemPoolGetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolGetAttribute
#endif
cudaMemPoolGetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2224(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2224(%rip)
2:
  pushq $278
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolImportFromShareableHandle
  .p2align 4
  .type cudaMemPoolImportFromShareableHandle, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolImportFromShareableHandle
#endif
cudaMemPoolImportFromShareableHandle:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2232(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2232(%rip)
2:
  pushq $279
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolImportPointer
  .p2align 4
  .type cudaMemPoolImportPointer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolImportPointer
#endif
cudaMemPoolImportPointer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2240(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2240(%rip)
2:
  pushq $280
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolSetAccess
  .p2align 4
  .type cudaMemPoolSetAccess, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolSetAccess
#endif
cudaMemPoolSetAccess:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2248(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2248(%rip)
2:
  pushq $281
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolSetAttribute
  .p2align 4
  .type cudaMemPoolSetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolSetAttribute
#endif
cudaMemPoolSetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2256(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2256(%rip)
2:
  pushq $282
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPoolTrimTo
  .p2align 4
  .type cudaMemPoolTrimTo, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPoolTrimTo
#endif
cudaMemPoolTrimTo:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2264(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2264(%rip)
2:
  pushq $283
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPrefetchAsync
  .p2align 4
  .type cudaMemPrefetchAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPrefetchAsync
#endif
cudaMemPrefetchAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2272(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2272(%rip)
2:
  pushq $284
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPrefetchAsync_ptsz
  .p2align 4
  .type cudaMemPrefetchAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPrefetchAsync_ptsz
#endif
cudaMemPrefetchAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2280(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2280(%rip)
2:
  pushq $285
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPrefetchAsync_v2
  .p2align 4
  .type cudaMemPrefetchAsync_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPrefetchAsync_v2
#endif
cudaMemPrefetchAsync_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2288(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2288(%rip)
2:
  pushq $286
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemPrefetchAsync_v2_ptsz
  .p2align 4
  .type cudaMemPrefetchAsync_v2_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemPrefetchAsync_v2_ptsz
#endif
cudaMemPrefetchAsync_v2_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2296(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2296(%rip)
2:
  pushq $287
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemRangeGetAttribute
  .p2align 4
  .type cudaMemRangeGetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemRangeGetAttribute
#endif
cudaMemRangeGetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2304(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2304(%rip)
2:
  pushq $288
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemRangeGetAttributes
  .p2align 4
  .type cudaMemRangeGetAttributes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemRangeGetAttributes
#endif
cudaMemRangeGetAttributes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2312(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2312(%rip)
2:
  pushq $289
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy
  .p2align 4
  .type cudaMemcpy, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy
#endif
cudaMemcpy:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2320(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2320(%rip)
2:
  pushq $290
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2D
  .p2align 4
  .type cudaMemcpy2D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2D
#endif
cudaMemcpy2D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2328(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2328(%rip)
2:
  pushq $291
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DArrayToArray
  .p2align 4
  .type cudaMemcpy2DArrayToArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DArrayToArray
#endif
cudaMemcpy2DArrayToArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2336(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2336(%rip)
2:
  pushq $292
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DArrayToArray_ptds
  .p2align 4
  .type cudaMemcpy2DArrayToArray_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DArrayToArray_ptds
#endif
cudaMemcpy2DArrayToArray_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2344(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2344(%rip)
2:
  pushq $293
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DAsync
  .p2align 4
  .type cudaMemcpy2DAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DAsync
#endif
cudaMemcpy2DAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2352(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2352(%rip)
2:
  pushq $294
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DAsync_ptsz
  .p2align 4
  .type cudaMemcpy2DAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DAsync_ptsz
#endif
cudaMemcpy2DAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2360(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2360(%rip)
2:
  pushq $295
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DFromArray
  .p2align 4
  .type cudaMemcpy2DFromArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DFromArray
#endif
cudaMemcpy2DFromArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2368(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2368(%rip)
2:
  pushq $296
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DFromArrayAsync
  .p2align 4
  .type cudaMemcpy2DFromArrayAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DFromArrayAsync
#endif
cudaMemcpy2DFromArrayAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2376(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2376(%rip)
2:
  pushq $297
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DFromArrayAsync_ptsz
  .p2align 4
  .type cudaMemcpy2DFromArrayAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DFromArrayAsync_ptsz
#endif
cudaMemcpy2DFromArrayAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2384(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2384(%rip)
2:
  pushq $298
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DFromArray_ptds
  .p2align 4
  .type cudaMemcpy2DFromArray_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DFromArray_ptds
#endif
cudaMemcpy2DFromArray_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2392(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2392(%rip)
2:
  pushq $299
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DToArray
  .p2align 4
  .type cudaMemcpy2DToArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DToArray
#endif
cudaMemcpy2DToArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2400(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2400(%rip)
2:
  pushq $300
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DToArrayAsync
  .p2align 4
  .type cudaMemcpy2DToArrayAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DToArrayAsync
#endif
cudaMemcpy2DToArrayAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2408(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2408(%rip)
2:
  pushq $301
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DToArrayAsync_ptsz
  .p2align 4
  .type cudaMemcpy2DToArrayAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DToArrayAsync_ptsz
#endif
cudaMemcpy2DToArrayAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2416(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2416(%rip)
2:
  pushq $302
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2DToArray_ptds
  .p2align 4
  .type cudaMemcpy2DToArray_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2DToArray_ptds
#endif
cudaMemcpy2DToArray_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2424(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2424(%rip)
2:
  pushq $303
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy2D_ptds
  .p2align 4
  .type cudaMemcpy2D_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy2D_ptds
#endif
cudaMemcpy2D_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2432(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2432(%rip)
2:
  pushq $304
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3D
  .p2align 4
  .type cudaMemcpy3D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3D
#endif
cudaMemcpy3D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2440(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2440(%rip)
2:
  pushq $305
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DAsync
  .p2align 4
  .type cudaMemcpy3DAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DAsync
#endif
cudaMemcpy3DAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2448(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2448(%rip)
2:
  pushq $306
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DAsync_ptsz
  .p2align 4
  .type cudaMemcpy3DAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DAsync_ptsz
#endif
cudaMemcpy3DAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2456(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2456(%rip)
2:
  pushq $307
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DBatchAsync
  .p2align 4
  .type cudaMemcpy3DBatchAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DBatchAsync
#endif
cudaMemcpy3DBatchAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2464(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2464(%rip)
2:
  pushq $308
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DBatchAsync_ptsz
  .p2align 4
  .type cudaMemcpy3DBatchAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DBatchAsync_ptsz
#endif
cudaMemcpy3DBatchAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2472(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2472(%rip)
2:
  pushq $309
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DPeer
  .p2align 4
  .type cudaMemcpy3DPeer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DPeer
#endif
cudaMemcpy3DPeer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2480(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2480(%rip)
2:
  pushq $310
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DPeerAsync
  .p2align 4
  .type cudaMemcpy3DPeerAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DPeerAsync
#endif
cudaMemcpy3DPeerAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2488(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2488(%rip)
2:
  pushq $311
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DPeerAsync_ptsz
  .p2align 4
  .type cudaMemcpy3DPeerAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DPeerAsync_ptsz
#endif
cudaMemcpy3DPeerAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2496(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2496(%rip)
2:
  pushq $312
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3DPeer_ptds
  .p2align 4
  .type cudaMemcpy3DPeer_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3DPeer_ptds
#endif
cudaMemcpy3DPeer_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2504(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2504(%rip)
2:
  pushq $313
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy3D_ptds
  .p2align 4
  .type cudaMemcpy3D_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy3D_ptds
#endif
cudaMemcpy3D_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2512(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2512(%rip)
2:
  pushq $314
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyArrayToArray
  .p2align 4
  .type cudaMemcpyArrayToArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyArrayToArray
#endif
cudaMemcpyArrayToArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2520(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2520(%rip)
2:
  pushq $315
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyArrayToArray_ptds
  .p2align 4
  .type cudaMemcpyArrayToArray_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyArrayToArray_ptds
#endif
cudaMemcpyArrayToArray_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2528(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2528(%rip)
2:
  pushq $316
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyAsync
  .p2align 4
  .type cudaMemcpyAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyAsync
#endif
cudaMemcpyAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2536(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2536(%rip)
2:
  pushq $317
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyAsync_ptsz
  .p2align 4
  .type cudaMemcpyAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyAsync_ptsz
#endif
cudaMemcpyAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2544(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2544(%rip)
2:
  pushq $318
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyBatchAsync
  .p2align 4
  .type cudaMemcpyBatchAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyBatchAsync
#endif
cudaMemcpyBatchAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2552(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2552(%rip)
2:
  pushq $319
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyBatchAsync_ptsz
  .p2align 4
  .type cudaMemcpyBatchAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyBatchAsync_ptsz
#endif
cudaMemcpyBatchAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2560(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2560(%rip)
2:
  pushq $320
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromArray
  .p2align 4
  .type cudaMemcpyFromArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromArray
#endif
cudaMemcpyFromArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2568(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2568(%rip)
2:
  pushq $321
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromArrayAsync
  .p2align 4
  .type cudaMemcpyFromArrayAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromArrayAsync
#endif
cudaMemcpyFromArrayAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2576(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2576(%rip)
2:
  pushq $322
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromArrayAsync_ptsz
  .p2align 4
  .type cudaMemcpyFromArrayAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromArrayAsync_ptsz
#endif
cudaMemcpyFromArrayAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2584(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2584(%rip)
2:
  pushq $323
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromArray_ptds
  .p2align 4
  .type cudaMemcpyFromArray_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromArray_ptds
#endif
cudaMemcpyFromArray_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2592(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2592(%rip)
2:
  pushq $324
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromSymbol
  .p2align 4
  .type cudaMemcpyFromSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromSymbol
#endif
cudaMemcpyFromSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2600(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2600(%rip)
2:
  pushq $325
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromSymbolAsync
  .p2align 4
  .type cudaMemcpyFromSymbolAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromSymbolAsync
#endif
cudaMemcpyFromSymbolAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2608(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2608(%rip)
2:
  pushq $326
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromSymbolAsync_ptsz
  .p2align 4
  .type cudaMemcpyFromSymbolAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromSymbolAsync_ptsz
#endif
cudaMemcpyFromSymbolAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2616(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2616(%rip)
2:
  pushq $327
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyFromSymbol_ptds
  .p2align 4
  .type cudaMemcpyFromSymbol_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyFromSymbol_ptds
#endif
cudaMemcpyFromSymbol_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2624(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2624(%rip)
2:
  pushq $328
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyPeer
  .p2align 4
  .type cudaMemcpyPeer, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyPeer
#endif
cudaMemcpyPeer:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2632(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2632(%rip)
2:
  pushq $329
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyPeerAsync
  .p2align 4
  .type cudaMemcpyPeerAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyPeerAsync
#endif
cudaMemcpyPeerAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2640(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2640(%rip)
2:
  pushq $330
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToArray
  .p2align 4
  .type cudaMemcpyToArray, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToArray
#endif
cudaMemcpyToArray:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2648(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2648(%rip)
2:
  pushq $331
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToArrayAsync
  .p2align 4
  .type cudaMemcpyToArrayAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToArrayAsync
#endif
cudaMemcpyToArrayAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2656(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2656(%rip)
2:
  pushq $332
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToArrayAsync_ptsz
  .p2align 4
  .type cudaMemcpyToArrayAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToArrayAsync_ptsz
#endif
cudaMemcpyToArrayAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2664(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2664(%rip)
2:
  pushq $333
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToArray_ptds
  .p2align 4
  .type cudaMemcpyToArray_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToArray_ptds
#endif
cudaMemcpyToArray_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2672(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2672(%rip)
2:
  pushq $334
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToSymbol
  .p2align 4
  .type cudaMemcpyToSymbol, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToSymbol
#endif
cudaMemcpyToSymbol:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2680(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2680(%rip)
2:
  pushq $335
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToSymbolAsync
  .p2align 4
  .type cudaMemcpyToSymbolAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToSymbolAsync
#endif
cudaMemcpyToSymbolAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2688(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2688(%rip)
2:
  pushq $336
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToSymbolAsync_ptsz
  .p2align 4
  .type cudaMemcpyToSymbolAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToSymbolAsync_ptsz
#endif
cudaMemcpyToSymbolAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2696(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2696(%rip)
2:
  pushq $337
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpyToSymbol_ptds
  .p2align 4
  .type cudaMemcpyToSymbol_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpyToSymbol_ptds
#endif
cudaMemcpyToSymbol_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2704(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2704(%rip)
2:
  pushq $338
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemcpy_ptds
  .p2align 4
  .type cudaMemcpy_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemcpy_ptds
#endif
cudaMemcpy_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2712(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2712(%rip)
2:
  pushq $339
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset
  .p2align 4
  .type cudaMemset, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset
#endif
cudaMemset:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2720(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2720(%rip)
2:
  pushq $340
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset2D
  .p2align 4
  .type cudaMemset2D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset2D
#endif
cudaMemset2D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2728(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2728(%rip)
2:
  pushq $341
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset2DAsync
  .p2align 4
  .type cudaMemset2DAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset2DAsync
#endif
cudaMemset2DAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2736(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2736(%rip)
2:
  pushq $342
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset2DAsync_ptsz
  .p2align 4
  .type cudaMemset2DAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset2DAsync_ptsz
#endif
cudaMemset2DAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2744(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2744(%rip)
2:
  pushq $343
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset2D_ptds
  .p2align 4
  .type cudaMemset2D_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset2D_ptds
#endif
cudaMemset2D_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2752(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2752(%rip)
2:
  pushq $344
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset3D
  .p2align 4
  .type cudaMemset3D, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset3D
#endif
cudaMemset3D:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2760(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2760(%rip)
2:
  pushq $345
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset3DAsync
  .p2align 4
  .type cudaMemset3DAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset3DAsync
#endif
cudaMemset3DAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2768(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2768(%rip)
2:
  pushq $346
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset3DAsync_ptsz
  .p2align 4
  .type cudaMemset3DAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset3DAsync_ptsz
#endif
cudaMemset3DAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2776(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2776(%rip)
2:
  pushq $347
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset3D_ptds
  .p2align 4
  .type cudaMemset3D_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset3D_ptds
#endif
cudaMemset3D_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2784(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2784(%rip)
2:
  pushq $348
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemsetAsync
  .p2align 4
  .type cudaMemsetAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemsetAsync
#endif
cudaMemsetAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2792(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2792(%rip)
2:
  pushq $349
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemsetAsync_ptsz
  .p2align 4
  .type cudaMemsetAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemsetAsync_ptsz
#endif
cudaMemsetAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2800(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2800(%rip)
2:
  pushq $350
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMemset_ptds
  .p2align 4
  .type cudaMemset_ptds, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMemset_ptds
#endif
cudaMemset_ptds:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2808(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2808(%rip)
2:
  pushq $351
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMipmappedArrayGetMemoryRequirements
  .p2align 4
  .type cudaMipmappedArrayGetMemoryRequirements, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMipmappedArrayGetMemoryRequirements
#endif
cudaMipmappedArrayGetMemoryRequirements:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2816(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2816(%rip)
2:
  pushq $352
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaMipmappedArrayGetSparseProperties
  .p2align 4
  .type cudaMipmappedArrayGetSparseProperties, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaMipmappedArrayGetSparseProperties
#endif
cudaMipmappedArrayGetSparseProperties:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2824(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2824(%rip)
2:
  pushq $353
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaOccupancyAvailableDynamicSMemPerBlock
  .p2align 4
  .type cudaOccupancyAvailableDynamicSMemPerBlock, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaOccupancyAvailableDynamicSMemPerBlock
#endif
cudaOccupancyAvailableDynamicSMemPerBlock:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2832(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2832(%rip)
2:
  pushq $354
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaOccupancyMaxActiveBlocksPerMultiprocessor
  .p2align 4
  .type cudaOccupancyMaxActiveBlocksPerMultiprocessor, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaOccupancyMaxActiveBlocksPerMultiprocessor
#endif
cudaOccupancyMaxActiveBlocksPerMultiprocessor:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2840(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2840(%rip)
2:
  pushq $355
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  .p2align 4
  .type cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#endif
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2848(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2848(%rip)
2:
  pushq $356
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaOccupancyMaxActiveClusters
  .p2align 4
  .type cudaOccupancyMaxActiveClusters, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaOccupancyMaxActiveClusters
#endif
cudaOccupancyMaxActiveClusters:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2856(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2856(%rip)
2:
  pushq $357
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaOccupancyMaxPotentialClusterSize
  .p2align 4
  .type cudaOccupancyMaxPotentialClusterSize, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaOccupancyMaxPotentialClusterSize
#endif
cudaOccupancyMaxPotentialClusterSize:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2864(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2864(%rip)
2:
  pushq $358
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaPeekAtLastError
  .p2align 4
  .type cudaPeekAtLastError, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaPeekAtLastError
#endif
cudaPeekAtLastError:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2872(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2872(%rip)
2:
  pushq $359
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaPointerGetAttributes
  .p2align 4
  .type cudaPointerGetAttributes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaPointerGetAttributes
#endif
cudaPointerGetAttributes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2880(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2880(%rip)
2:
  pushq $360
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaProfilerStart
  .p2align 4
  .type cudaProfilerStart, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaProfilerStart
#endif
cudaProfilerStart:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2888(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2888(%rip)
2:
  pushq $361
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaProfilerStop
  .p2align 4
  .type cudaProfilerStop, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaProfilerStop
#endif
cudaProfilerStop:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2896(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2896(%rip)
2:
  pushq $362
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaRuntimeGetVersion
  .p2align 4
  .type cudaRuntimeGetVersion, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaRuntimeGetVersion
#endif
cudaRuntimeGetVersion:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2904(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2904(%rip)
2:
  pushq $363
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSetDevice
  .p2align 4
  .type cudaSetDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSetDevice
#endif
cudaSetDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2912(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2912(%rip)
2:
  pushq $364
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSetDeviceFlags
  .p2align 4
  .type cudaSetDeviceFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSetDeviceFlags
#endif
cudaSetDeviceFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2920(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2920(%rip)
2:
  pushq $365
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSetDoubleForDevice
  .p2align 4
  .type cudaSetDoubleForDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSetDoubleForDevice
#endif
cudaSetDoubleForDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2928(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2928(%rip)
2:
  pushq $366
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSetDoubleForHost
  .p2align 4
  .type cudaSetDoubleForHost, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSetDoubleForHost
#endif
cudaSetDoubleForHost:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2936(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2936(%rip)
2:
  pushq $367
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSetValidDevices
  .p2align 4
  .type cudaSetValidDevices, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSetValidDevices
#endif
cudaSetValidDevices:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2944(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2944(%rip)
2:
  pushq $368
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSignalExternalSemaphoresAsync
  .p2align 4
  .type cudaSignalExternalSemaphoresAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSignalExternalSemaphoresAsync
#endif
cudaSignalExternalSemaphoresAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2952(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2952(%rip)
2:
  pushq $369
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSignalExternalSemaphoresAsync_ptsz
  .p2align 4
  .type cudaSignalExternalSemaphoresAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSignalExternalSemaphoresAsync_ptsz
#endif
cudaSignalExternalSemaphoresAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2960(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2960(%rip)
2:
  pushq $370
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSignalExternalSemaphoresAsync_v2
  .p2align 4
  .type cudaSignalExternalSemaphoresAsync_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSignalExternalSemaphoresAsync_v2
#endif
cudaSignalExternalSemaphoresAsync_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2968(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2968(%rip)
2:
  pushq $371
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaSignalExternalSemaphoresAsync_v2_ptsz
  .p2align 4
  .type cudaSignalExternalSemaphoresAsync_v2_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaSignalExternalSemaphoresAsync_v2_ptsz
#endif
cudaSignalExternalSemaphoresAsync_v2_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2976(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2976(%rip)
2:
  pushq $372
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamAddCallback
  .p2align 4
  .type cudaStreamAddCallback, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamAddCallback
#endif
cudaStreamAddCallback:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2984(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2984(%rip)
2:
  pushq $373
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamAddCallback_ptsz
  .p2align 4
  .type cudaStreamAddCallback_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamAddCallback_ptsz
#endif
cudaStreamAddCallback_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+2992(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+2992(%rip)
2:
  pushq $374
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamAttachMemAsync
  .p2align 4
  .type cudaStreamAttachMemAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamAttachMemAsync
#endif
cudaStreamAttachMemAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3000(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3000(%rip)
2:
  pushq $375
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamAttachMemAsync_ptsz
  .p2align 4
  .type cudaStreamAttachMemAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamAttachMemAsync_ptsz
#endif
cudaStreamAttachMemAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3008(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3008(%rip)
2:
  pushq $376
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamBeginCapture
  .p2align 4
  .type cudaStreamBeginCapture, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamBeginCapture
#endif
cudaStreamBeginCapture:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3016(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3016(%rip)
2:
  pushq $377
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamBeginCaptureToGraph
  .p2align 4
  .type cudaStreamBeginCaptureToGraph, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamBeginCaptureToGraph
#endif
cudaStreamBeginCaptureToGraph:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3024(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3024(%rip)
2:
  pushq $378
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamBeginCaptureToGraph_ptsz
  .p2align 4
  .type cudaStreamBeginCaptureToGraph_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamBeginCaptureToGraph_ptsz
#endif
cudaStreamBeginCaptureToGraph_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3032(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3032(%rip)
2:
  pushq $379
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamBeginCapture_ptsz
  .p2align 4
  .type cudaStreamBeginCapture_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamBeginCapture_ptsz
#endif
cudaStreamBeginCapture_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3040(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3040(%rip)
2:
  pushq $380
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamCopyAttributes
  .p2align 4
  .type cudaStreamCopyAttributes, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamCopyAttributes
#endif
cudaStreamCopyAttributes:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3048(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3048(%rip)
2:
  pushq $381
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamCopyAttributes_ptsz
  .p2align 4
  .type cudaStreamCopyAttributes_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamCopyAttributes_ptsz
#endif
cudaStreamCopyAttributes_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3056(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3056(%rip)
2:
  pushq $382
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamCreate
  .p2align 4
  .type cudaStreamCreate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamCreate
#endif
cudaStreamCreate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3064(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3064(%rip)
2:
  pushq $383
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamCreateWithFlags
  .p2align 4
  .type cudaStreamCreateWithFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamCreateWithFlags
#endif
cudaStreamCreateWithFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3072(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3072(%rip)
2:
  pushq $384
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamCreateWithPriority
  .p2align 4
  .type cudaStreamCreateWithPriority, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamCreateWithPriority
#endif
cudaStreamCreateWithPriority:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3080(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3080(%rip)
2:
  pushq $385
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamDestroy
  .p2align 4
  .type cudaStreamDestroy, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamDestroy
#endif
cudaStreamDestroy:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3088(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3088(%rip)
2:
  pushq $386
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamEndCapture
  .p2align 4
  .type cudaStreamEndCapture, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamEndCapture
#endif
cudaStreamEndCapture:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3096(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3096(%rip)
2:
  pushq $387
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamEndCapture_ptsz
  .p2align 4
  .type cudaStreamEndCapture_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamEndCapture_ptsz
#endif
cudaStreamEndCapture_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3104(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3104(%rip)
2:
  pushq $388
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetAttribute
  .p2align 4
  .type cudaStreamGetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetAttribute
#endif
cudaStreamGetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3112(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3112(%rip)
2:
  pushq $389
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetAttribute_ptsz
  .p2align 4
  .type cudaStreamGetAttribute_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetAttribute_ptsz
#endif
cudaStreamGetAttribute_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3120(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3120(%rip)
2:
  pushq $390
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetCaptureInfo
  .p2align 4
  .type cudaStreamGetCaptureInfo, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetCaptureInfo
#endif
cudaStreamGetCaptureInfo:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3128(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3128(%rip)
2:
  pushq $391
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetCaptureInfo_ptsz
  .p2align 4
  .type cudaStreamGetCaptureInfo_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetCaptureInfo_ptsz
#endif
cudaStreamGetCaptureInfo_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3136(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3136(%rip)
2:
  pushq $392
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetCaptureInfo_v2
  .p2align 4
  .type cudaStreamGetCaptureInfo_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetCaptureInfo_v2
#endif
cudaStreamGetCaptureInfo_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3144(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3144(%rip)
2:
  pushq $393
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetCaptureInfo_v2_ptsz
  .p2align 4
  .type cudaStreamGetCaptureInfo_v2_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetCaptureInfo_v2_ptsz
#endif
cudaStreamGetCaptureInfo_v2_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3152(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3152(%rip)
2:
  pushq $394
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetCaptureInfo_v3
  .p2align 4
  .type cudaStreamGetCaptureInfo_v3, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetCaptureInfo_v3
#endif
cudaStreamGetCaptureInfo_v3:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3160(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3160(%rip)
2:
  pushq $395
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetCaptureInfo_v3_ptsz
  .p2align 4
  .type cudaStreamGetCaptureInfo_v3_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetCaptureInfo_v3_ptsz
#endif
cudaStreamGetCaptureInfo_v3_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3168(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3168(%rip)
2:
  pushq $396
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetDevice
  .p2align 4
  .type cudaStreamGetDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetDevice
#endif
cudaStreamGetDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3176(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3176(%rip)
2:
  pushq $397
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetDevice_ptsz
  .p2align 4
  .type cudaStreamGetDevice_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetDevice_ptsz
#endif
cudaStreamGetDevice_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3184(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3184(%rip)
2:
  pushq $398
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetFlags
  .p2align 4
  .type cudaStreamGetFlags, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetFlags
#endif
cudaStreamGetFlags:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3192(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3192(%rip)
2:
  pushq $399
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetFlags_ptsz
  .p2align 4
  .type cudaStreamGetFlags_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetFlags_ptsz
#endif
cudaStreamGetFlags_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3200(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3200(%rip)
2:
  pushq $400
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetId
  .p2align 4
  .type cudaStreamGetId, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetId
#endif
cudaStreamGetId:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3208(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3208(%rip)
2:
  pushq $401
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetId_ptsz
  .p2align 4
  .type cudaStreamGetId_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetId_ptsz
#endif
cudaStreamGetId_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3216(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3216(%rip)
2:
  pushq $402
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetPriority
  .p2align 4
  .type cudaStreamGetPriority, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetPriority
#endif
cudaStreamGetPriority:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3224(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3224(%rip)
2:
  pushq $403
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamGetPriority_ptsz
  .p2align 4
  .type cudaStreamGetPriority_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamGetPriority_ptsz
#endif
cudaStreamGetPriority_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3232(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3232(%rip)
2:
  pushq $404
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamIsCapturing
  .p2align 4
  .type cudaStreamIsCapturing, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamIsCapturing
#endif
cudaStreamIsCapturing:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3240(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3240(%rip)
2:
  pushq $405
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamIsCapturing_ptsz
  .p2align 4
  .type cudaStreamIsCapturing_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamIsCapturing_ptsz
#endif
cudaStreamIsCapturing_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3248(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3248(%rip)
2:
  pushq $406
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamQuery
  .p2align 4
  .type cudaStreamQuery, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamQuery
#endif
cudaStreamQuery:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3256(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3256(%rip)
2:
  pushq $407
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamQuery_ptsz
  .p2align 4
  .type cudaStreamQuery_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamQuery_ptsz
#endif
cudaStreamQuery_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3264(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3264(%rip)
2:
  pushq $408
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamSetAttribute
  .p2align 4
  .type cudaStreamSetAttribute, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamSetAttribute
#endif
cudaStreamSetAttribute:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3272(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3272(%rip)
2:
  pushq $409
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamSetAttribute_ptsz
  .p2align 4
  .type cudaStreamSetAttribute_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamSetAttribute_ptsz
#endif
cudaStreamSetAttribute_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3280(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3280(%rip)
2:
  pushq $410
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamSynchronize
  .p2align 4
  .type cudaStreamSynchronize, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamSynchronize
#endif
cudaStreamSynchronize:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3288(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3288(%rip)
2:
  pushq $411
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamSynchronize_ptsz
  .p2align 4
  .type cudaStreamSynchronize_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamSynchronize_ptsz
#endif
cudaStreamSynchronize_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3296(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3296(%rip)
2:
  pushq $412
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamUpdateCaptureDependencies
  .p2align 4
  .type cudaStreamUpdateCaptureDependencies, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamUpdateCaptureDependencies
#endif
cudaStreamUpdateCaptureDependencies:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3304(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3304(%rip)
2:
  pushq $413
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamUpdateCaptureDependencies_ptsz
  .p2align 4
  .type cudaStreamUpdateCaptureDependencies_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamUpdateCaptureDependencies_ptsz
#endif
cudaStreamUpdateCaptureDependencies_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3312(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3312(%rip)
2:
  pushq $414
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamUpdateCaptureDependencies_v2
  .p2align 4
  .type cudaStreamUpdateCaptureDependencies_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamUpdateCaptureDependencies_v2
#endif
cudaStreamUpdateCaptureDependencies_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3320(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3320(%rip)
2:
  pushq $415
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamUpdateCaptureDependencies_v2_ptsz
  .p2align 4
  .type cudaStreamUpdateCaptureDependencies_v2_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamUpdateCaptureDependencies_v2_ptsz
#endif
cudaStreamUpdateCaptureDependencies_v2_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3328(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3328(%rip)
2:
  pushq $416
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamWaitEvent
  .p2align 4
  .type cudaStreamWaitEvent, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamWaitEvent
#endif
cudaStreamWaitEvent:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3336(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3336(%rip)
2:
  pushq $417
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaStreamWaitEvent_ptsz
  .p2align 4
  .type cudaStreamWaitEvent_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaStreamWaitEvent_ptsz
#endif
cudaStreamWaitEvent_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3344(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3344(%rip)
2:
  pushq $418
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadExchangeStreamCaptureMode
  .p2align 4
  .type cudaThreadExchangeStreamCaptureMode, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadExchangeStreamCaptureMode
#endif
cudaThreadExchangeStreamCaptureMode:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3352(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3352(%rip)
2:
  pushq $419
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadExit
  .p2align 4
  .type cudaThreadExit, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadExit
#endif
cudaThreadExit:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3360(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3360(%rip)
2:
  pushq $420
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadGetCacheConfig
  .p2align 4
  .type cudaThreadGetCacheConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadGetCacheConfig
#endif
cudaThreadGetCacheConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3368(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3368(%rip)
2:
  pushq $421
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadGetLimit
  .p2align 4
  .type cudaThreadGetLimit, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadGetLimit
#endif
cudaThreadGetLimit:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3376(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3376(%rip)
2:
  pushq $422
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadSetCacheConfig
  .p2align 4
  .type cudaThreadSetCacheConfig, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadSetCacheConfig
#endif
cudaThreadSetCacheConfig:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3384(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3384(%rip)
2:
  pushq $423
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadSetLimit
  .p2align 4
  .type cudaThreadSetLimit, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadSetLimit
#endif
cudaThreadSetLimit:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3392(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3392(%rip)
2:
  pushq $424
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaThreadSynchronize
  .p2align 4
  .type cudaThreadSynchronize, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaThreadSynchronize
#endif
cudaThreadSynchronize:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3400(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3400(%rip)
2:
  pushq $425
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaUserObjectCreate
  .p2align 4
  .type cudaUserObjectCreate, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaUserObjectCreate
#endif
cudaUserObjectCreate:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3408(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3408(%rip)
2:
  pushq $426
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaUserObjectRelease
  .p2align 4
  .type cudaUserObjectRelease, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaUserObjectRelease
#endif
cudaUserObjectRelease:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3416(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3416(%rip)
2:
  pushq $427
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaUserObjectRetain
  .p2align 4
  .type cudaUserObjectRetain, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaUserObjectRetain
#endif
cudaUserObjectRetain:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3424(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3424(%rip)
2:
  pushq $428
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaVDPAUGetDevice
  .p2align 4
  .type cudaVDPAUGetDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaVDPAUGetDevice
#endif
cudaVDPAUGetDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3432(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3432(%rip)
2:
  pushq $429
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaVDPAUSetVDPAUDevice
  .p2align 4
  .type cudaVDPAUSetVDPAUDevice, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaVDPAUSetVDPAUDevice
#endif
cudaVDPAUSetVDPAUDevice:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3440(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3440(%rip)
2:
  pushq $430
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaWaitExternalSemaphoresAsync
  .p2align 4
  .type cudaWaitExternalSemaphoresAsync, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaWaitExternalSemaphoresAsync
#endif
cudaWaitExternalSemaphoresAsync:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3448(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3448(%rip)
2:
  pushq $431
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaWaitExternalSemaphoresAsync_ptsz
  .p2align 4
  .type cudaWaitExternalSemaphoresAsync_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaWaitExternalSemaphoresAsync_ptsz
#endif
cudaWaitExternalSemaphoresAsync_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3456(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3456(%rip)
2:
  pushq $432
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaWaitExternalSemaphoresAsync_v2
  .p2align 4
  .type cudaWaitExternalSemaphoresAsync_v2, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaWaitExternalSemaphoresAsync_v2
#endif
cudaWaitExternalSemaphoresAsync_v2:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3464(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3464(%rip)
2:
  pushq $433
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl cudaWaitExternalSemaphoresAsync_v2_ptsz
  .p2align 4
  .type cudaWaitExternalSemaphoresAsync_v2_ptsz, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden cudaWaitExternalSemaphoresAsync_v2_ptsz
#endif
cudaWaitExternalSemaphoresAsync_v2_ptsz:
  .cfi_startproc
  .cfi_def_cfa_offset 8  // Return address
  // Intel opt. manual says to
  // "make the fall-through code following a conditional branch be the likely target for a branch with a forward target"
  // to hint static predictor.
  cmpq $0, _libcudart_so_tramp_table+3472(%rip)
  je 2f
1:
  jmp *_libcudart_so_tramp_table+3472(%rip)
2:
  pushq $434
  .cfi_adjust_cfa_offset 8
  call _libcudart_so_save_regs_and_resolve
  addq $8, %rsp
  .cfi_adjust_cfa_offset -8
  jmp 1b
  .cfi_endproc
