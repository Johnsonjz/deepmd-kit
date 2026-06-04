set (LMP_INSTALL_PREFIX "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/USER-DEEPMD")
file(READ "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/lmp/lammps_install_list.txt" files)
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
  FILES "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/lmp/env.sh"
)

file (
  INSTALL DESTINATION "${LMP_INSTALL_PREFIX}"
  TYPE FILE
  FILES "/data/home/public/jiangzhen/dp/deepmd-kit/source/build_pt/lmp/deepmd_version.h"
)
