# Adapted from https://pytorch.org/tutorials/advanced/cpp_extension.html
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="flashtrainext",
    ext_modules=[
        cpp_extension.CppExtension(
            "cuda_reistered_allocator", ["allocator.cpp"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
