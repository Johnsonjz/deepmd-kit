# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import platform
from ctypes import (
    CDLL,
    RTLD_GLOBAL,
)
from importlib import (
    metadata,
)
from pathlib import (
    Path,
)

import torch
from packaging.version import (
    Version,
)

from deepmd.env import (
    GLOBAL_CONFIG,
    SHARED_LIB_DIR,
)


_OP_LIB_OVERRIDE_ENV = "DEEPMD_OP_PT_LIB"
_ALLOW_PRELOADED_ENV = "DEEPMD_OP_PT_ALLOW_PRELOADED"


def _is_truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_module_file(
    module_name: str,
    prefix: str,
    ext: str,
) -> tuple[Path, bool]:
    override_path = os.environ.get(_OP_LIB_OVERRIDE_ENV)
    if override_path:
        return Path(override_path).expanduser().resolve(), True
    return (SHARED_LIB_DIR / (prefix + module_name)).with_suffix(ext).resolve(), False


def _loaded_library_hints() -> list[str]:
    loaded_libraries = sorted(torch.ops.loaded_libraries)
    deepmd_libraries = [
        lib
        for lib in loaded_libraries
        if "deepmd_op_pt" in lib or "libdeepmd_op_pt" in lib
    ]
    return deepmd_libraries if deepmd_libraries else loaded_libraries


def load_library(module_name: str) -> bool:
    """Load OP library.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    bool
        Whether the library is loaded successfully
    """
    if platform.system() == "Windows":
        ext = ".dll"
        prefix = ""
    else:
        ext = ".so"
        prefix = "lib"

    module_file, from_env_override = _resolve_module_file(module_name, prefix, ext)

    if not module_file.is_file():
        if from_env_override:
            raise RuntimeError(
                f"Environment variable {_OP_LIB_OVERRIDE_ENV} points to a non-existent file: {module_file}"
            )
        return False

    # Skip if this exact library path was already loaded by torch.ops.load_library.
    if str(module_file) in torch.ops.loaded_libraries:
        return True

    # If deepmd ops are already registered before this call, abort by default
    # to avoid silently using an unexpected preloaded library.
    if hasattr(torch.ops, "deepmd") and hasattr(torch.ops.deepmd, "enable_mpi"):
        if _is_truthy_env(_ALLOW_PRELOADED_ENV):
            return True
        loaded_hints = _loaded_library_hints()
        hint_text = "\n".join(loaded_hints) if loaded_hints else "(none reported by torch.ops.loaded_libraries)"
        raise RuntimeError(
            "DeepMD custom ops are already registered before deepmd.pt.cxx_op.load_library() "
            "could load the expected library path. This can indicate a mismatched or stale "
            "libdeepmd_op_pt.so in the current process.\n"
            f"Expected library path: {module_file}\n"
            f"Environment override ({_OP_LIB_OVERRIDE_ENV}): {os.environ.get(_OP_LIB_OVERRIDE_ENV, '(unset)')}\n"
            f"Loaded-library hints:\n{hint_text}\n"
            f"If this preloaded setup is intentional, set {_ALLOW_PRELOADED_ENV}=1 to bypass this check."
        )

    try:
        torch.ops.load_library(module_file)
    except OSError as e:
        # check: CXX11_ABI_FLAG; version
        # from our op
        PT_VERSION = GLOBAL_CONFIG["pt_version"]
        PT_CXX11_ABI_FLAG = int(GLOBAL_CONFIG["pt_cxx11_abi_flag"])
        # from torch
        # strip the local version
        pt_py_version = Version(torch.__version__).public
        pt_cxx11_abi_flag = int(torch.compiled_with_cxx11_abi())

        if PT_CXX11_ABI_FLAG != pt_cxx11_abi_flag:
            raise RuntimeError(
                "This deepmd-kit package was compiled with "
                f"CXX11_ABI_FLAG={PT_CXX11_ABI_FLAG}, but PyTorch runtime was compiled "
                f"with CXX11_ABI_FLAG={pt_cxx11_abi_flag}. These two library ABIs are "
                f"incompatible and thus an error is raised when loading {module_name}. "
                "You need to rebuild deepmd-kit against this PyTorch "
                "runtime."
            ) from e

        # different versions may cause incompatibility, see TF
        if PT_VERSION != pt_py_version:
            raise RuntimeError(
                "The version of PyTorch used to compile this "
                f"deepmd-kit package is {PT_VERSION}, but the version of PyTorch "
                f"runtime you are using is {pt_py_version}. These two versions are "
                f"incompatible and thus an error is raised when loading {module_name}. "
                f"You need to install PyTorch {PT_VERSION}, or rebuild deepmd-kit "
                f"against PyTorch {pt_py_version}.\nIf you are using a wheel from "
                "PyPI, you may consider to install deepmd-kit execuating "
                "`DP_ENABLE_PYTORCH=1 pip install deepmd-kit --no-binary deepmd-kit` "
                "instead."
            ) from e
        error_message = (
            "This deepmd-kit package is inconsistent with PyTorch "
            f"Runtime, thus an error is raised when loading {module_name}. "
            "You need to rebuild deepmd-kit against this PyTorch "
            "runtime."
        )
        if PT_CXX11_ABI_FLAG == 1:
            # #1791
            error_message += (
                "\nWARNING: devtoolset on RHEL6 and RHEL7 does not support _GLIBCXX_USE_CXX11_ABI=1. "
                "See https://bugzilla.redhat.com/show_bug.cgi?id=1546704"
            )
        raise RuntimeError(error_message) from e
    return True


def load_mpi_library() -> None:
    """Load MPI library.

    When building with cibuildwheel, the link to the MPI library is lost
    after the wheel is repaired.
    """
    if platform.system() == "Linux":
        libname = "libmpi.so.*"
    elif platform.system() == "Darwin":
        libname = "libmpi.*.dylib"
    else:
        raise RuntimeError("Unsupported platform")
    MPI_LIB = next(p for p in metadata.files("mpich") if p.match(libname)).locate()
    # use CDLL to load the library
    CDLL(MPI_LIB, mode=RTLD_GLOBAL)


if GLOBAL_CONFIG.get("cibuildwheel", "0") == "1" and platform.system() in (
    "Linux",
    "Darwin",
):
    load_mpi_library()

ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pt")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
]
