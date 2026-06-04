# Install script for directory: /data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_stable")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so"
         RPATH "\$ORIGIN:/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib:/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib/../../torch.libs:/data/app/intel/oneapi/mkl/2022.1.0/lib:/data/app/intel/oneapi/mkl/2022.1.0/lib/intel64:/data/app/intel/oneapi/mkl/2022.1.0/lib/intel64_win:/data/app/intel/oneapi/mkl/2022.1.0/lib/win-x64:/usr/local/cuda-12.4/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/api_cc/libdeepmd_cc.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so"
         OLD_RPATH "\$ORIGIN/../op/tf:\$ORIGIN/../op/pt:\$ORIGIN/../op/pd:/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib/../../torch.libs:/data/app/intel/oneapi/mkl/2022.1.0/lib:/data/app/intel/oneapi/mkl/2022.1.0/lib/intel64:/data/app/intel/oneapi/mkl/2022.1.0/lib/intel64_win:/data/app/intel/oneapi/mkl/2022.1.0/lib/win-x64:/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/lib:/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-12.4/lib64:"
         NEW_RPATH "\$ORIGIN:/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib:/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib/../../torch.libs:/data/app/intel/oneapi/mkl/2022.1.0/lib:/data/app/intel/oneapi/mkl/2022.1.0/lib/intel64:/data/app/intel/oneapi/mkl/2022.1.0/lib/intel64_win:/data/app/intel/oneapi/mkl/2022.1.0/lib/win-x64:/usr/local/cuda-12.4/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/api_cc/CMakeFiles/deepmd_cc.dir/install-cxx-module-bmi-release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/deepmd" TYPE FILE FILES
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/AtomMap.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DataModifier.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DataModifierTF.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepBaseModel.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepPot.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepPotJAX.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepPotPD.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepPotPT.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepPotPTExpt.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepPotTF.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepSpin.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepSpinPT.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepSpinTF.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepTensor.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepTensorPT.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/DeepTensorTF.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/common.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/commonPT.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/commonTF.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/tf_private.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/api_cc/include/tf_public.h"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/api_cc/version.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process( COMMAND /data/home/public/.conda/envs/mlip/bin/cmake -E create_symlink libdeepmd_cc.so /data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_stable/lib/libdeepmd_cc_low.so   )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/api_cc/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
