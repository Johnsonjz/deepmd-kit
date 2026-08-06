#define GIT_SUMM 
#define GIT_HASH 96ffc5b4
#define GIT_BRANCH sog-lmp
#define GIT_DATE 2026-07-15 08:41:27 +0000
#define DEEPMD_ROOT /usr/local
#define BACKEND_INCLUDE_DIRS /root/code/dp_dev/lib/python3.14/site-packages/torch/include;/root/code/dp_dev/lib/python3.14/site-packages/torch/include/torch/csrc/api/include
#define BACKEND_LIBRARY_PATH /root/code/dp_dev/lib/python3.14/site-packages/torch/lib
#define DPMD_CVT_STR(...) #__VA_ARGS__
#define DPMD_CVT_ASSTR(X) DPMD_CVT_STR(X)
#define STR_GIT_SUMM DPMD_CVT_ASSTR(GIT_SUMM)
#define STR_GIT_HASH DPMD_CVT_ASSTR(GIT_HASH)
#define STR_GIT_BRANCH DPMD_CVT_ASSTR(GIT_BRANCH)
#define STR_GIT_DATE DPMD_CVT_ASSTR(GIT_DATE)
#define STR_FLOAT_PREC DPMD_CVT_ASSTR(FLOAT_PREC)
#define STR_DEEPMD_ROOT DPMD_CVT_ASSTR(DEEPMD_ROOT)
#define STR_BACKEND_INCLUDE_DIRS DPMD_CVT_ASSTR(BACKEND_INCLUDE_DIRS)
#define STR_BACKEND_LIBRARY_PATH DPMD_CVT_ASSTR(BACKEND_LIBRARY_PATH)
