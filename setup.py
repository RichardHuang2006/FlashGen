"""
setup.py — builds the flashgen._C PyBind11 extension via CMake.

Usage:
    pip install -e .                  # editable install (development)
    pip install .                     # regular install
    GPU_ARCH=86 pip install -e .      # override GPU architecture (RTX 30xx)
    GPU_ARCH=89 pip install -e .      # RTX 40xx
    GPU_ARCH=90 pip install -e .      # H100

The setup.py delegates CUDA compilation to CMakeLists.txt so that nvcc
flags, separable compilation, and TensorRT detection all live in one place.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Stub Extension that triggers cmake build instead of setuptools compiler."""

    def __init__(self, name: str, source_dir: str = "."):
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class CMakeBuildExt(build_ext):
    """Custom build_ext that calls cmake + make for CMakeExtension targets."""

    def build_extension(self, ext: CMakeExtension):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        gpu_arch = os.environ.get("GPU_ARCH", "80")
        build_type = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DGPU_ARCH={gpu_arch}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = ["--config", build_type, "--target", "flashgen_ext", "-j4"]

        subprocess.run(
            ["cmake", ext.source_dir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )


setup(
    name="flashgen",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "build*"]),
    ext_modules=[CMakeExtension("flashgen._C")],
    cmdclass={"build_ext": CMakeBuildExt},
    zip_safe=False,
    python_requires=">=3.9",
)
