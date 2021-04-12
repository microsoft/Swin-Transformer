from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='local_relation',
    ext_modules=[
        CUDAExtension('local_relation_cuda', [
            'src/local_relation_cuda.cpp',
            'src/local_relation_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
