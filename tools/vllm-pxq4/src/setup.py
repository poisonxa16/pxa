"""setup.py — alternative to CMake for machines that have disk and a normal pip.

CMakeLists.txt is the primary path (it is what build.sh drives inside a throwaway container,
and it emits exactly the libpxq4_sm70.so filename the runtime package loads). This file exists
so the extension can also be built with `pip install -e .` during development.

Note the extension is a plain torch CUDAExtension with no vLLM dependency whatsoever: the ops
are published through TORCH_LIBRARY(pxq4, ...) and reached with torch.ops.load_library().
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name="pxq4_sm70",
    version="0.1.0",
    description="PXQ4 (ggml type 252) dequant + decode GEMV kernels for sm_70",
    ext_modules=[
        CUDAExtension(
            name="pxq4_sm70",
            sources=[
                os.path.join(HERE, "pxq4_kernel.cu"),
                os.path.join(HERE, "pxq4_kernel_torch.cpp"),
            ],
            include_dirs=[HERE],
            extra_compile_args={
                # No -ffast-math / -use_fast_math anywhere: contraction or reassociation
                # would break bit-exactness against the shipping llama.cpp kernels, which is
                # the entire correctness argument for this port.
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode", "arch=compute_70,code=sm_70",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
