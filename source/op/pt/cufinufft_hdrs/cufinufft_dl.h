#pragma once
#include <glob.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <string>
#include <vector>
#include "cufinufft.h"

class CuFinufftLib {
public:
    static CuFinufftLib& getInstance() {
        static CuFinufftLib instance;
        return instance;
    }

    void* handle;
    
    // Type 1 Single Precision
    typedef int (*makeplan_f)(
        int,
        int,
        int64_t*,
        int,
        int,
        float,
        cufinufft_plan*,
        cufinufft_opts*);
    typedef int (*setpts_f)(
        cufinufft_plan,
        int64_t,
        float*,
        float*,
        float*,
        int64_t,
        float*,
        float*,
        float*);
    typedef int (*execute_f)(cufinufft_plan, float*, float*);

    // Type 1 Double Precision
    typedef int (*makeplan_d)(
        int,
        int,
        int64_t*,
        int,
        int,
        double,
        cufinufft_plan*,
        cufinufft_opts*);
    typedef int (*setpts_d)(
        cufinufft_plan,
        int64_t,
        double*,
        double*,
        double*,
        int64_t,
        double*,
        double*,
        double*);
    typedef int (*execute_d)(cufinufft_plan, double*, double*);

    typedef int (*destroy_f)(cufinufft_plan);
    typedef void (*default_opts_f)(cufinufft_opts*);
    typedef void (*default_opts_d)(cufinufft_opts*);

    makeplan_f makeplanf;
    setpts_f setptsf;
    execute_f executef;
    
    makeplan_d makepland;
    setpts_d setptsd;
    execute_d executed;

    destroy_f destroyf;
    default_opts_f default_optsf;
    default_opts_d default_optsd;

private:
    CuFinufftLib() {
        std::vector<std::string> candidates = {"libcufinufft.so"};
        const char* conda_prefix = std::getenv("CONDA_PREFIX");
        if (conda_prefix != nullptr) {
            std::string prefix(conda_prefix);
            candidates.push_back(prefix + "/lib/libcufinufft.so");

            glob_t py_glob{};
            std::string py_pattern =
                prefix + "/lib/python*/site-packages/cufinufft/libcufinufft.so";
            if (glob(py_pattern.c_str(), 0, nullptr, &py_glob) == 0) {
                for (size_t ii = 0; ii < py_glob.gl_pathc; ++ii) {
                    candidates.push_back(py_glob.gl_pathv[ii]);
                }
            }
            globfree(&py_glob);
        }

        for (const auto& path : candidates) {
            handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (handle != nullptr) {
                break;
            }
        }

        if (!handle) {
            std::cerr << "Warning: Could not load libcufinufft.so (" << dlerror() << ")\n";
            return;
        }

        makeplanf = (makeplan_f)dlsym(handle, "cufinufftf_makeplan");
        setptsf = (setpts_f)dlsym(handle, "cufinufftf_setpts");
        executef = (execute_f)dlsym(handle, "cufinufftf_execute");
        
        makepland = (makeplan_d)dlsym(handle, "cufinufft_makeplan");
        setptsd = (setpts_d)dlsym(handle, "cufinufft_setpts");
        executed = (execute_d)dlsym(handle, "cufinufft_execute");

        destroyf = (destroy_f)dlsym(handle, "cufinufft_destroy");
        default_optsf = (default_opts_f)dlsym(handle, "cufinufftf_default_opts");
        default_optsd = (default_opts_d)dlsym(handle, "cufinufft_default_opts");

        if (!makepland || !setptsd || !executed) {
            std::cerr << "Warning: Failed to map cufinufft double-precision symbols\n";
        }

        if (!makeplanf || !setptsf || !executef) {
            std::cerr << "Warning: Failed to map cufinufft single-precision symbols\n";
        }
    }
};
