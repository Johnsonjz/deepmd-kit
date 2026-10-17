# Install script for directory: /data/home/public/jiangzhen/dp/deepmd-kit/source

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi")
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
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/rh/gcc-toolset-13/root/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/lib/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/op/pt/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/api_cc/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/api_c/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/lmp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/ipi/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/gmx/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDTargets.cmake"
         "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/CMakeFiles/Export/66cf505cce0b298b0999c10447b9289e/DeePMDTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDTargets.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD" TYPE FILE FILES "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/CMakeFiles/Export/66cf505cce0b298b0999c10447b9289e/DeePMDTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDTargets-release.cmake")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
      message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    file(INSTALL DESTINATION "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD" TYPE FILE FILES "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/CMakeFiles/Export/66cf505cce0b298b0999c10447b9289e/DeePMDTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDConfigVersion.cmake;/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD/DeePMDConfig.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_current_mpi/lib/cmake/DeePMD" TYPE FILE FILES
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/generated/DeePMDConfigVersion.cmake"
    "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/generated/DeePMDConfig.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_current_cuda_mpi/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
