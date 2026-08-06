/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // For RTLD_DEFAULT
#endif

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// Sanity check for ARM to avoid puzzling runtime crashes
#ifdef __arm__
# if defined __thumb__ && ! defined __THUMB_INTERWORK__
#   error "ARM trampolines need -mthumb-interwork to work in Thumb mode"
# endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK(cond, fmt, ...) do { \
    if(!(cond)) { \
      fprintf(stderr, "implib-gen: libcudart.so.12: " fmt "\n", ##__VA_ARGS__); \
    } \
  } while(0)

#define HAS_DLOPEN_CALLBACK 1
#define HAS_DLSYM_CALLBACK 1
#define NO_DLOPEN 0
#define LAZY_LOAD 1

static void *lib_handle;
static int do_dlclose;
static int is_lib_loading;

#if ! NO_DLOPEN
static void *load_library() {
  if(lib_handle)
    return lib_handle;

  is_lib_loading = 1;

#if HAS_DLOPEN_CALLBACK
  extern void *DP_cudart_dlopen(const char *lib_name);
  lib_handle = DP_cudart_dlopen("libcudart.so.12");
  CHECK(lib_handle, "failed to load library 'libcudart.so.12' via callback 'DP_cudart_dlopen'");
#else
  lib_handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
  CHECK(lib_handle, "failed to load library 'libcudart.so.12' via dlopen: %s", dlerror());
#endif

  do_dlclose = 1;
  is_lib_loading = 0;

  return lib_handle;
}

static void __attribute__((destructor)) unload_lib() {
  if(do_dlclose && lib_handle)
    dlclose(lib_handle);
}
#endif

#if ! NO_DLOPEN && ! LAZY_LOAD
static void __attribute__((constructor)) load_lib() {
  load_library();
}
#endif

// TODO: convert to single 0-separated string
static const char *const sym_names[] = {
  "__cudaGetKernel",
  "__cudaInitModule",
  "__cudaLaunchKernel",
  "__cudaLaunchKernel_ptsz",
  "__cudaPopCallConfiguration",
  "__cudaPushCallConfiguration",
  "__cudaRegisterFatBinary",
  "__cudaRegisterFatBinaryEnd",
  "__cudaRegisterFunction",
  "__cudaRegisterHostVar",
  "__cudaRegisterManagedVar",
  "__cudaRegisterUnifiedTable",
  "__cudaRegisterVar",
  "__cudaUnregisterFatBinary",
  "cudaArrayGetInfo",
  "cudaArrayGetMemoryRequirements",
  "cudaArrayGetPlane",
  "cudaArrayGetSparseProperties",
  "cudaChooseDevice",
  "cudaCreateChannelDesc",
  "cudaCreateSurfaceObject",
  "cudaCreateTextureObject",
  "cudaCtxResetPersistingL2Cache",
  "cudaDestroyExternalMemory",
  "cudaDestroyExternalSemaphore",
  "cudaDestroySurfaceObject",
  "cudaDestroyTextureObject",
  "cudaDeviceCanAccessPeer",
  "cudaDeviceDisablePeerAccess",
  "cudaDeviceEnablePeerAccess",
  "cudaDeviceFlushGPUDirectRDMAWrites",
  "cudaDeviceGetAttribute",
  "cudaDeviceGetByPCIBusId",
  "cudaDeviceGetCacheConfig",
  "cudaDeviceGetDefaultMemPool",
  "cudaDeviceGetGraphMemAttribute",
  "cudaDeviceGetLimit",
  "cudaDeviceGetMemPool",
  "cudaDeviceGetNvSciSyncAttributes",
  "cudaDeviceGetP2PAttribute",
  "cudaDeviceGetPCIBusId",
  "cudaDeviceGetSharedMemConfig",
  "cudaDeviceGetStreamPriorityRange",
  "cudaDeviceGetTexture1DLinearMaxWidth",
  "cudaDeviceGraphMemTrim",
  "cudaDeviceRegisterAsyncNotification",
  "cudaDeviceReset",
  "cudaDeviceSetCacheConfig",
  "cudaDeviceSetGraphMemAttribute",
  "cudaDeviceSetLimit",
  "cudaDeviceSetMemPool",
  "cudaDeviceSetSharedMemConfig",
  "cudaDeviceSynchronize",
  "cudaDeviceUnregisterAsyncNotification",
  "cudaDriverGetVersion",
  "cudaEGLStreamConsumerAcquireFrame",
  "cudaEGLStreamConsumerConnect",
  "cudaEGLStreamConsumerConnectWithFlags",
  "cudaEGLStreamConsumerDisconnect",
  "cudaEGLStreamConsumerReleaseFrame",
  "cudaEGLStreamProducerConnect",
  "cudaEGLStreamProducerDisconnect",
  "cudaEGLStreamProducerPresentFrame",
  "cudaEGLStreamProducerReturnFrame",
  "cudaEventCreate",
  "cudaEventCreateFromEGLSync",
  "cudaEventCreateWithFlags",
  "cudaEventDestroy",
  "cudaEventElapsedTime",
  "cudaEventElapsedTime_v2",
  "cudaEventQuery",
  "cudaEventRecord",
  "cudaEventRecordWithFlags",
  "cudaEventRecordWithFlags_ptsz",
  "cudaEventRecord_ptsz",
  "cudaEventSynchronize",
  "cudaExternalMemoryGetMappedBuffer",
  "cudaExternalMemoryGetMappedMipmappedArray",
  "cudaFree",
  "cudaFreeArray",
  "cudaFreeAsync",
  "cudaFreeAsync_ptsz",
  "cudaFreeHost",
  "cudaFreeMipmappedArray",
  "cudaFuncGetAttributes",
  "cudaFuncGetName",
  "cudaFuncGetParamInfo",
  "cudaFuncSetAttribute",
  "cudaFuncSetCacheConfig",
  "cudaFuncSetSharedMemConfig",
  "cudaGLGetDevices",
  "cudaGLMapBufferObject",
  "cudaGLMapBufferObjectAsync",
  "cudaGLRegisterBufferObject",
  "cudaGLSetBufferObjectMapFlags",
  "cudaGLSetGLDevice",
  "cudaGLUnmapBufferObject",
  "cudaGLUnmapBufferObjectAsync",
  "cudaGLUnregisterBufferObject",
  "cudaGetChannelDesc",
  "cudaGetDevice",
  "cudaGetDeviceCount",
  "cudaGetDeviceFlags",
  "cudaGetDeviceProperties",
  "cudaGetDeviceProperties_v2",
  "cudaGetDriverEntryPoint",
  "cudaGetDriverEntryPointByVersion",
  "cudaGetDriverEntryPointByVersion_ptsz",
  "cudaGetDriverEntryPoint_ptsz",
  "cudaGetErrorName",
  "cudaGetErrorString",
  "cudaGetExportTable",
  "cudaGetFuncBySymbol",
  "cudaGetKernel",
  "cudaGetLastError",
  "cudaGetMipmappedArrayLevel",
  "cudaGetSurfaceObjectResourceDesc",
  "cudaGetSymbolAddress",
  "cudaGetSymbolSize",
  "cudaGetTextureObjectResourceDesc",
  "cudaGetTextureObjectResourceViewDesc",
  "cudaGetTextureObjectTextureDesc",
  "cudaGraphAddChildGraphNode",
  "cudaGraphAddDependencies",
  "cudaGraphAddDependencies_v2",
  "cudaGraphAddEmptyNode",
  "cudaGraphAddEventRecordNode",
  "cudaGraphAddEventWaitNode",
  "cudaGraphAddExternalSemaphoresSignalNode",
  "cudaGraphAddExternalSemaphoresWaitNode",
  "cudaGraphAddHostNode",
  "cudaGraphAddKernelNode",
  "cudaGraphAddMemAllocNode",
  "cudaGraphAddMemFreeNode",
  "cudaGraphAddMemcpyNode",
  "cudaGraphAddMemcpyNode1D",
  "cudaGraphAddMemcpyNodeFromSymbol",
  "cudaGraphAddMemcpyNodeToSymbol",
  "cudaGraphAddMemsetNode",
  "cudaGraphAddNode",
  "cudaGraphAddNode_v2",
  "cudaGraphChildGraphNodeGetGraph",
  "cudaGraphClone",
  "cudaGraphConditionalHandleCreate",
  "cudaGraphCreate",
  "cudaGraphDebugDotPrint",
  "cudaGraphDestroy",
  "cudaGraphDestroyNode",
  "cudaGraphEventRecordNodeGetEvent",
  "cudaGraphEventRecordNodeSetEvent",
  "cudaGraphEventWaitNodeGetEvent",
  "cudaGraphEventWaitNodeSetEvent",
  "cudaGraphExecChildGraphNodeSetParams",
  "cudaGraphExecDestroy",
  "cudaGraphExecEventRecordNodeSetEvent",
  "cudaGraphExecEventWaitNodeSetEvent",
  "cudaGraphExecExternalSemaphoresSignalNodeSetParams",
  "cudaGraphExecExternalSemaphoresWaitNodeSetParams",
  "cudaGraphExecGetFlags",
  "cudaGraphExecHostNodeSetParams",
  "cudaGraphExecKernelNodeSetParams",
  "cudaGraphExecMemcpyNodeSetParams",
  "cudaGraphExecMemcpyNodeSetParams1D",
  "cudaGraphExecMemcpyNodeSetParamsFromSymbol",
  "cudaGraphExecMemcpyNodeSetParamsToSymbol",
  "cudaGraphExecMemsetNodeSetParams",
  "cudaGraphExecNodeSetParams",
  "cudaGraphExecUpdate",
  "cudaGraphExternalSemaphoresSignalNodeGetParams",
  "cudaGraphExternalSemaphoresSignalNodeSetParams",
  "cudaGraphExternalSemaphoresWaitNodeGetParams",
  "cudaGraphExternalSemaphoresWaitNodeSetParams",
  "cudaGraphGetEdges",
  "cudaGraphGetEdges_v2",
  "cudaGraphGetNodes",
  "cudaGraphGetRootNodes",
  "cudaGraphHostNodeGetParams",
  "cudaGraphHostNodeSetParams",
  "cudaGraphInstantiate",
  "cudaGraphInstantiateWithFlags",
  "cudaGraphInstantiateWithParams",
  "cudaGraphInstantiateWithParams_ptsz",
  "cudaGraphKernelNodeCopyAttributes",
  "cudaGraphKernelNodeGetAttribute",
  "cudaGraphKernelNodeGetParams",
  "cudaGraphKernelNodeSetAttribute",
  "cudaGraphKernelNodeSetParams",
  "cudaGraphLaunch",
  "cudaGraphLaunch_ptsz",
  "cudaGraphMemAllocNodeGetParams",
  "cudaGraphMemFreeNodeGetParams",
  "cudaGraphMemcpyNodeGetParams",
  "cudaGraphMemcpyNodeSetParams",
  "cudaGraphMemcpyNodeSetParams1D",
  "cudaGraphMemcpyNodeSetParamsFromSymbol",
  "cudaGraphMemcpyNodeSetParamsToSymbol",
  "cudaGraphMemsetNodeGetParams",
  "cudaGraphMemsetNodeSetParams",
  "cudaGraphNodeFindInClone",
  "cudaGraphNodeGetDependencies",
  "cudaGraphNodeGetDependencies_v2",
  "cudaGraphNodeGetDependentNodes",
  "cudaGraphNodeGetDependentNodes_v2",
  "cudaGraphNodeGetEnabled",
  "cudaGraphNodeGetType",
  "cudaGraphNodeSetEnabled",
  "cudaGraphNodeSetParams",
  "cudaGraphReleaseUserObject",
  "cudaGraphRemoveDependencies",
  "cudaGraphRemoveDependencies_v2",
  "cudaGraphRetainUserObject",
  "cudaGraphUpload",
  "cudaGraphUpload_ptsz",
  "cudaGraphicsEGLRegisterImage",
  "cudaGraphicsGLRegisterBuffer",
  "cudaGraphicsGLRegisterImage",
  "cudaGraphicsMapResources",
  "cudaGraphicsResourceGetMappedEglFrame",
  "cudaGraphicsResourceGetMappedMipmappedArray",
  "cudaGraphicsResourceGetMappedPointer",
  "cudaGraphicsResourceSetMapFlags",
  "cudaGraphicsSubResourceGetMappedArray",
  "cudaGraphicsUnmapResources",
  "cudaGraphicsUnregisterResource",
  "cudaGraphicsVDPAURegisterOutputSurface",
  "cudaGraphicsVDPAURegisterVideoSurface",
  "cudaHostAlloc",
  "cudaHostGetDevicePointer",
  "cudaHostGetFlags",
  "cudaHostRegister",
  "cudaHostUnregister",
  "cudaImportExternalMemory",
  "cudaImportExternalSemaphore",
  "cudaInitDevice",
  "cudaIpcCloseMemHandle",
  "cudaIpcGetEventHandle",
  "cudaIpcGetMemHandle",
  "cudaIpcOpenEventHandle",
  "cudaIpcOpenMemHandle",
  "cudaKernelSetAttributeForDevice",
  "cudaLaunchCooperativeKernel",
  "cudaLaunchCooperativeKernelMultiDevice",
  "cudaLaunchCooperativeKernel_ptsz",
  "cudaLaunchHostFunc",
  "cudaLaunchHostFunc_ptsz",
  "cudaLaunchKernel",
  "cudaLaunchKernelExC",
  "cudaLaunchKernelExC_ptsz",
  "cudaLaunchKernel_ptsz",
  "cudaLibraryEnumerateKernels",
  "cudaLibraryGetGlobal",
  "cudaLibraryGetKernel",
  "cudaLibraryGetKernelCount",
  "cudaLibraryGetManaged",
  "cudaLibraryGetUnifiedFunction",
  "cudaLibraryLoadData",
  "cudaLibraryLoadFromFile",
  "cudaLibraryUnload",
  "cudaMalloc",
  "cudaMalloc3D",
  "cudaMalloc3DArray",
  "cudaMallocArray",
  "cudaMallocAsync",
  "cudaMallocAsync_ptsz",
  "cudaMallocFromPoolAsync",
  "cudaMallocFromPoolAsync_ptsz",
  "cudaMallocHost",
  "cudaMallocManaged",
  "cudaMallocMipmappedArray",
  "cudaMallocPitch",
  "cudaMemAdvise",
  "cudaMemAdvise_v2",
  "cudaMemGetInfo",
  "cudaMemPoolCreate",
  "cudaMemPoolDestroy",
  "cudaMemPoolExportPointer",
  "cudaMemPoolExportToShareableHandle",
  "cudaMemPoolGetAccess",
  "cudaMemPoolGetAttribute",
  "cudaMemPoolImportFromShareableHandle",
  "cudaMemPoolImportPointer",
  "cudaMemPoolSetAccess",
  "cudaMemPoolSetAttribute",
  "cudaMemPoolTrimTo",
  "cudaMemPrefetchAsync",
  "cudaMemPrefetchAsync_ptsz",
  "cudaMemPrefetchAsync_v2",
  "cudaMemPrefetchAsync_v2_ptsz",
  "cudaMemRangeGetAttribute",
  "cudaMemRangeGetAttributes",
  "cudaMemcpy",
  "cudaMemcpy2D",
  "cudaMemcpy2DArrayToArray",
  "cudaMemcpy2DArrayToArray_ptds",
  "cudaMemcpy2DAsync",
  "cudaMemcpy2DAsync_ptsz",
  "cudaMemcpy2DFromArray",
  "cudaMemcpy2DFromArrayAsync",
  "cudaMemcpy2DFromArrayAsync_ptsz",
  "cudaMemcpy2DFromArray_ptds",
  "cudaMemcpy2DToArray",
  "cudaMemcpy2DToArrayAsync",
  "cudaMemcpy2DToArrayAsync_ptsz",
  "cudaMemcpy2DToArray_ptds",
  "cudaMemcpy2D_ptds",
  "cudaMemcpy3D",
  "cudaMemcpy3DAsync",
  "cudaMemcpy3DAsync_ptsz",
  "cudaMemcpy3DBatchAsync",
  "cudaMemcpy3DBatchAsync_ptsz",
  "cudaMemcpy3DPeer",
  "cudaMemcpy3DPeerAsync",
  "cudaMemcpy3DPeerAsync_ptsz",
  "cudaMemcpy3DPeer_ptds",
  "cudaMemcpy3D_ptds",
  "cudaMemcpyArrayToArray",
  "cudaMemcpyArrayToArray_ptds",
  "cudaMemcpyAsync",
  "cudaMemcpyAsync_ptsz",
  "cudaMemcpyBatchAsync",
  "cudaMemcpyBatchAsync_ptsz",
  "cudaMemcpyFromArray",
  "cudaMemcpyFromArrayAsync",
  "cudaMemcpyFromArrayAsync_ptsz",
  "cudaMemcpyFromArray_ptds",
  "cudaMemcpyFromSymbol",
  "cudaMemcpyFromSymbolAsync",
  "cudaMemcpyFromSymbolAsync_ptsz",
  "cudaMemcpyFromSymbol_ptds",
  "cudaMemcpyPeer",
  "cudaMemcpyPeerAsync",
  "cudaMemcpyToArray",
  "cudaMemcpyToArrayAsync",
  "cudaMemcpyToArrayAsync_ptsz",
  "cudaMemcpyToArray_ptds",
  "cudaMemcpyToSymbol",
  "cudaMemcpyToSymbolAsync",
  "cudaMemcpyToSymbolAsync_ptsz",
  "cudaMemcpyToSymbol_ptds",
  "cudaMemcpy_ptds",
  "cudaMemset",
  "cudaMemset2D",
  "cudaMemset2DAsync",
  "cudaMemset2DAsync_ptsz",
  "cudaMemset2D_ptds",
  "cudaMemset3D",
  "cudaMemset3DAsync",
  "cudaMemset3DAsync_ptsz",
  "cudaMemset3D_ptds",
  "cudaMemsetAsync",
  "cudaMemsetAsync_ptsz",
  "cudaMemset_ptds",
  "cudaMipmappedArrayGetMemoryRequirements",
  "cudaMipmappedArrayGetSparseProperties",
  "cudaOccupancyAvailableDynamicSMemPerBlock",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
  "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
  "cudaOccupancyMaxActiveClusters",
  "cudaOccupancyMaxPotentialClusterSize",
  "cudaPeekAtLastError",
  "cudaPointerGetAttributes",
  "cudaProfilerStart",
  "cudaProfilerStop",
  "cudaRuntimeGetVersion",
  "cudaSetDevice",
  "cudaSetDeviceFlags",
  "cudaSetDoubleForDevice",
  "cudaSetDoubleForHost",
  "cudaSetValidDevices",
  "cudaSignalExternalSemaphoresAsync",
  "cudaSignalExternalSemaphoresAsync_ptsz",
  "cudaSignalExternalSemaphoresAsync_v2",
  "cudaSignalExternalSemaphoresAsync_v2_ptsz",
  "cudaStreamAddCallback",
  "cudaStreamAddCallback_ptsz",
  "cudaStreamAttachMemAsync",
  "cudaStreamAttachMemAsync_ptsz",
  "cudaStreamBeginCapture",
  "cudaStreamBeginCaptureToGraph",
  "cudaStreamBeginCaptureToGraph_ptsz",
  "cudaStreamBeginCapture_ptsz",
  "cudaStreamCopyAttributes",
  "cudaStreamCopyAttributes_ptsz",
  "cudaStreamCreate",
  "cudaStreamCreateWithFlags",
  "cudaStreamCreateWithPriority",
  "cudaStreamDestroy",
  "cudaStreamEndCapture",
  "cudaStreamEndCapture_ptsz",
  "cudaStreamGetAttribute",
  "cudaStreamGetAttribute_ptsz",
  "cudaStreamGetCaptureInfo",
  "cudaStreamGetCaptureInfo_ptsz",
  "cudaStreamGetCaptureInfo_v2",
  "cudaStreamGetCaptureInfo_v2_ptsz",
  "cudaStreamGetCaptureInfo_v3",
  "cudaStreamGetCaptureInfo_v3_ptsz",
  "cudaStreamGetDevice",
  "cudaStreamGetDevice_ptsz",
  "cudaStreamGetFlags",
  "cudaStreamGetFlags_ptsz",
  "cudaStreamGetId",
  "cudaStreamGetId_ptsz",
  "cudaStreamGetPriority",
  "cudaStreamGetPriority_ptsz",
  "cudaStreamIsCapturing",
  "cudaStreamIsCapturing_ptsz",
  "cudaStreamQuery",
  "cudaStreamQuery_ptsz",
  "cudaStreamSetAttribute",
  "cudaStreamSetAttribute_ptsz",
  "cudaStreamSynchronize",
  "cudaStreamSynchronize_ptsz",
  "cudaStreamUpdateCaptureDependencies",
  "cudaStreamUpdateCaptureDependencies_ptsz",
  "cudaStreamUpdateCaptureDependencies_v2",
  "cudaStreamUpdateCaptureDependencies_v2_ptsz",
  "cudaStreamWaitEvent",
  "cudaStreamWaitEvent_ptsz",
  "cudaThreadExchangeStreamCaptureMode",
  "cudaThreadExit",
  "cudaThreadGetCacheConfig",
  "cudaThreadGetLimit",
  "cudaThreadSetCacheConfig",
  "cudaThreadSetLimit",
  "cudaThreadSynchronize",
  "cudaUserObjectCreate",
  "cudaUserObjectRelease",
  "cudaUserObjectRetain",
  "cudaVDPAUGetDevice",
  "cudaVDPAUSetVDPAUDevice",
  "cudaWaitExternalSemaphoresAsync",
  "cudaWaitExternalSemaphoresAsync_ptsz",
  "cudaWaitExternalSemaphoresAsync_v2",
  "cudaWaitExternalSemaphoresAsync_v2_ptsz",
  0
};

#define SYM_COUNT (sizeof(sym_names)/sizeof(sym_names[0]) - 1)

extern void *_libcudart_so_tramp_table[];

// Can be sped up by manually parsing library symtab...
void _libcudart_so_tramp_resolve(int i) {
  assert((unsigned)i < SYM_COUNT);

  CHECK(!is_lib_loading, "library function '%s' called during library load", sym_names[i]);

  void *h = 0;
#if NO_DLOPEN
  // Library with implementations must have already been loaded.
  if (lib_handle) {
    // User has specified loaded library
    h = lib_handle;
  } else {
    // User hasn't provided us the loaded library so search the global namespace.
#   ifndef IMPLIB_EXPORT_SHIMS
    // If shim symbols are hidden we should search
    // for first available definition of symbol in library list
    h = RTLD_DEFAULT;
#   else
    // Otherwise look for next available definition
    h = RTLD_NEXT;
#   endif
  }
#else
  h = load_library();
  CHECK(h, "failed to resolve symbol '%s', library failed to load", sym_names[i]);
#endif

#if HAS_DLSYM_CALLBACK
  extern void *DP_cudart_dlsym(void *handle, const char *sym_name);
  _libcudart_so_tramp_table[i] = DP_cudart_dlsym(h, sym_names[i]);
  CHECK(_libcudart_so_tramp_table[i], "failed to resolve symbol '%s' via callback DP_cudart_dlsym", sym_names[i]);
#else
  // Dlsym is thread-safe so don't need to protect it.
  _libcudart_so_tramp_table[i] = dlsym(h, sym_names[i]);
  CHECK(_libcudart_so_tramp_table[i], "failed to resolve symbol '%s' via dlsym: %s", sym_names[i], dlerror());
#endif
}

// Helper for user to resolve all symbols
void _libcudart_so_tramp_resolve_all(void) {
  size_t i;
  for(i = 0; i < SYM_COUNT; ++i)
    _libcudart_so_tramp_resolve(i);
}

// Allows user to specify manually loaded implementation library.
void _libcudart_so_tramp_set_handle(void *handle) {
  lib_handle = handle;
  do_dlclose = 0;
}

// Resets all resolved symbols. This is needed in case
// client code wants to reload interposed library multiple times.
void _libcudart_so_tramp_reset(void) {
  memset(_libcudart_so_tramp_table, 0, SYM_COUNT * sizeof(_libcudart_so_tramp_table[0]));
  lib_handle = 0;
  do_dlclose = 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif
