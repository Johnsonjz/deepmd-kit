# Install script for directory: /root/code/deepmd-kit/source/lib

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/root/code/deepmd-kit/source/build_local_fixed/lib/src/gpu/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so"
         RPATH "\$ORIGIN")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/root/code/deepmd-kit/source/build_local_fixed/lib/libdeepmd.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so"
         OLD_RPATH "/root/code/deepmd-kit/source/build_local_fixed/lib/src/gpu/cudart:"
         NEW_RPATH "\$ORIGIN")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdeepmd.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/root/code/deepmd-kit/source/build_local_fixed/lib/CMakeFiles/deepmd.dir/install-cxx-module-bmi-release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/deepmd" TYPE FILE FILES
    "/root/code/deepmd-kit/source/lib/include/ComputeDescriptor.h"
    "/root/code/deepmd-kit/source/lib/include/DeviceFunctor.h"
    "/root/code/deepmd-kit/source/lib/include/SimulationRegion.h"
    "/root/code/deepmd-kit/source/lib/include/SimulationRegion_Impl.h"
    "/root/code/deepmd-kit/source/lib/include/coord.h"
    "/root/code/deepmd-kit/source/lib/include/device.h"
    "/root/code/deepmd-kit/source/lib/include/env_mat.h"
    "/root/code/deepmd-kit/source/lib/include/env_mat_nvnmd.h"
    "/root/code/deepmd-kit/source/lib/include/errors.h"
    "/root/code/deepmd-kit/source/lib/include/ewald.h"
    "/root/code/deepmd-kit/source/lib/include/fmt_nlist.h"
    "/root/code/deepmd-kit/source/lib/include/gelu.h"
    "/root/code/deepmd-kit/source/lib/include/gpu_cuda.h"
    "/root/code/deepmd-kit/source/lib/include/gpu_rocm.h"
    "/root/code/deepmd-kit/source/lib/include/map_aparam.h"
    "/root/code/deepmd-kit/source/lib/include/neighbor_list.h"
    "/root/code/deepmd-kit/source/lib/include/neighbor_stat.h"
    "/root/code/deepmd-kit/source/lib/include/pair_tab.h"
    "/root/code/deepmd-kit/source/lib/include/pairwise.h"
    "/root/code/deepmd-kit/source/lib/include/prod_env_mat.h"
    "/root/code/deepmd-kit/source/lib/include/prod_env_mat_nvnmd.h"
    "/root/code/deepmd-kit/source/lib/include/prod_force.h"
    "/root/code/deepmd-kit/source/lib/include/prod_force_grad.h"
    "/root/code/deepmd-kit/source/lib/include/prod_virial.h"
    "/root/code/deepmd-kit/source/lib/include/prod_virial_grad.h"
    "/root/code/deepmd-kit/source/lib/include/region.h"
    "/root/code/deepmd-kit/source/lib/include/soft_min_switch.h"
    "/root/code/deepmd-kit/source/lib/include/soft_min_switch_force.h"
    "/root/code/deepmd-kit/source/lib/include/soft_min_switch_force_grad.h"
    "/root/code/deepmd-kit/source/lib/include/soft_min_switch_virial.h"
    "/root/code/deepmd-kit/source/lib/include/soft_min_switch_virial_grad.h"
    "/root/code/deepmd-kit/source/lib/include/switcher.h"
    "/root/code/deepmd-kit/source/lib/include/tabulate.h"
    "/root/code/deepmd-kit/source/lib/include/utilities.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/root/code/deepmd-kit/source/build_local_fixed/lib/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
