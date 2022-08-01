from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='swin_window_process',
    ext_modules=[
        CUDAExtension('swin_window_process', [
            'swin_window_process.cpp',
            'swin_window_process_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})