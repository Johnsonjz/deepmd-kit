set (LMP_INSTALL_PREFIX "/root/code/deepmd-kit/source/build_local/USER-DEEPMD")
file(READ "/root/code/deepmd-kit/source/build_local/lmp/lammps_install_list.txt" files)
string(REGEX REPLACE "\n" "" files "${files}")

foreach (cur_file ${files})
  file (
    INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
    USE_SOURCE_PERMISSIONS
    TYPE FILE
    FILES "${cur_file}"
    )
endforeach ()

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
  TYPE FILE
  FILES "/root/code/deepmd-kit/source/build_local/lmp/env.sh"
)

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
  TYPE FILE
  FILES "/root/code/deepmd-kit/source/build_local/lmp/deepmd_version.h"
)
