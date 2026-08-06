# Install script for directory: /root/code/deepmd-kit/source/api_cc

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
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
  set(CMAKE_INSTALL_SO_NO_EXE "1")
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
         RPATH "\$ORIGIN:/root/code/dp_dev/lib/python3.14/site-packages/torch/lib:/lib/intel64:/lib/intel64_win:/lib/win-x64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/root/code/deepmd-kit/source/build_local/api_cc/libdeepmd_cc.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so"
         OLD_RPATH "\$ORIGIN/../op/tf:\$ORIGIN/../op/pt:\$ORIGIN/../op/pd:/lib/intel64:/lib/intel64_win:/lib/win-x64:/root/code/deepmd-kit/source/build_local/lib:/root/code/dp_dev/lib/python3.14/site-packages/torch/lib:/root/code/deepmd-kit/source/build_local/lib/src/gpu:/root/code/deepmd-kit/source/build_local/lib/src/gpu/cudart:"
         NEW_RPATH "\$ORIGIN:/root/code/dp_dev/lib/python3.14/site-packages/torch/lib:/lib/intel64:/lib/intel64_win:/lib/win-x64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd_cc.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/root/code/deepmd-kit/source/build_local/api_cc/CMakeFiles/deepmd_cc.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/deepmd" TYPE FILE FILES
    "/root/code/deepmd-kit/source/api_cc/include/AtomMap.h"
    "/root/code/deepmd-kit/source/api_cc/include/DataModifier.h"
    "/root/code/deepmd-kit/source/api_cc/include/DataModifierTF.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepBaseModel.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepPot.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepPotJAX.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepPotPD.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepPotPT.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepPotPTExpt.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepPotTF.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepSpin.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepSpinPT.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepSpinTF.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepTensor.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepTensorPT.h"
    "/root/code/deepmd-kit/source/api_cc/include/DeepTensorTF.h"
    "/root/code/deepmd-kit/source/api_cc/include/common.h"
    "/root/code/deepmd-kit/source/api_cc/include/commonPT.h"
    "/root/code/deepmd-kit/source/api_cc/include/commonTF.h"
    "/root/code/deepmd-kit/source/api_cc/include/tf_private.h"
    "/root/code/deepmd-kit/source/api_cc/include/tf_public.h"
    "/root/code/deepmd-kit/source/build_local/api_cc/version.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process( COMMAND /root/code/dp_dev/lib/python3.14/site-packages/cmake/data/bin/cmake -E create_symlink libdeepmd_cc.so /usr/local/lib/libdeepmd_cc_low.so   )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/root/code/deepmd-kit/source/build_local/api_cc/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
