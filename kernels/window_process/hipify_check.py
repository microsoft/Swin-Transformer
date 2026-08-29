# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Check that the CUDA sources translate cleanly to HIP for ROCm.

AMD GPUs do not compile CUDA. They compile HIP, a nearly identical language, and
PyTorch gets from one to the other with a source-to-source translator called
hipify: CUDA names go in, HIP names come out. CUDAExtension runs it on its own
when torch.version.hip is set, so a ROCm build needs no second copy of these
sources -- but it gives no signal about whether the translation came out
complete. This is that signal, and because hipify is pure Python it can be had
with no AMD GPU (with no GPU at all) on a CPU runner in CI.

Three things fail the check: a CUDA-only header that survives translation, a
name that is already portable but gets rewritten anyway, and a kernel launch
that reaches HIP without naming a stream. Symbols that translate, and the two
PyTorch spellings that are valid either way, are reported rather than judged.

    python hipify_check.py            # verify
    python hipify_check.py --diff     # verify, and print the generated HIP
"""

import argparse
import difflib
import os
import re
import shutil
import sys
import tempfile

from torch.utils.hipify import hipify_python


SOURCES = ['swin_window_process.cpp', 'swin_window_process_kernel.cu']

# CUDA header files, which simply do not exist on an AMD machine. hipify has to
# rewrite every one of them to its HIP counterpart -- cuda_runtime.h becomes
# hip/hip_runtime.h, and so on. If one of these names is still there afterwards,
# the ROCm build stops at "no such file", so finding one is a failure.
MUST_BE_TRANSLATED = [
    'cuda.h',
    'cuda_runtime.h',
    'cuda_fp16.h',
    'ATen/cuda/CUDAContext.h',
    'c10/cuda/CUDAException.h',
]

# PyTorch functions and macros that work on ROCm under either name. Some hipify
# versions rename them to an at::hip / C10_HIP spelling and some leave the CUDA
# spelling in place, which resolves to the HIP runtime anyway: torch 2.8 and 2.10
# rewrite them, 2.12 and 2.13 do not. Both spellings reach the current HIP
# stream -- the kept form is the one the MI300X build below actually compiled and
# ran, on torch 2.12 -- so the check makes no prediction and accepts either,
# reporting which one happened. That is why it passes unchanged on CI legs that
# pin different torch versions.
CUDA_COMPAT_SHIMS = [
    'at::cuda::getCurrentCUDAStream',
    'C10_CUDA_KERNEL_LAUNCH_CHECK',
]

# Names that mean the same thing in both languages: HIP spells blockIdx,
# __global__ and the rest exactly as CUDA does, so hipify is meant to leave them
# alone. One coming back renamed would mean the translation went wrong, so that
# is a failure too.
KNOWN_PORTABLE = [
    '__global__',
    'blockIdx',
    'threadIdx',
    'blockDim',
    'gridDim',
    'dim3',
    'AT_DISPATCH_FLOATING_TYPES_AND2',
    'at::ScalarType::BFloat16',
]

# __ldg is deliberately not in the list above. It is portable as a *symbol* --
# HIP defines it, and hipify leaves it alone -- but not for every type, which is
# the whole point of the caveat printed on success. Since the kernels read
# through SWIN_WP_LDG, the only __ldg left in the source is inside the #else
# branch of that macro, which hipcc never compiles: asserting that the token
# survives translation would be asserting something about dead code. The macro
# is reported instead, so the reader sees which arm the build will take.
LDG_MACRO = 'SWIN_WP_LDG'

# Printed on success, because passing this check proves less than it sounds like.
SYMBOL_LEVEL_CAVEAT = """\
Symbol level only: a symbol can translate and still not compile, because HIP and
CUDA do not always give it the same overload set. __ldg is the case in point --
it has no HIP overload for c10::Half. Only a real build finds that."""


def read(path):
    with open(path) as handle:
        return handle.read()


def hipify_into(destination):
    """Translate SOURCES out of place and return {source: hipified path}."""
    here = os.path.dirname(os.path.abspath(__file__))
    staging = os.path.join(destination, 'src')
    os.makedirs(staging)
    for name in SOURCES:
        shutil.copy(os.path.join(here, name), staging)

    results = hipify_python.hipify(
        project_directory=staging,
        output_directory=os.path.join(destination, 'out'),
        includes=('*',),
        is_pytorch_extension=True,
        out_of_place_only=True,
        show_progress=False,
    )

    mapping = {}
    for source, result in results.items():
        hipified = getattr(result, 'hipified_path', None) or source
        mapping[os.path.basename(source)] = hipified
    return staging, mapping


def report(staging, mapping, show_diff):
    """Classify each source's device-specific symbols and return the failures."""
    failures = []

    for name in SOURCES:
        original = os.path.join(staging, name)
        hipified = mapping.get(name)
        if hipified is None or not os.path.exists(hipified):
            failures.append(f'{name}: hipify produced no output')
            continue

        before = read(original)
        after = read(hipified)

        print(f'\n{name} -> {os.path.basename(hipified)}')

        translated, survived = [], []
        for token in MUST_BE_TRANSLATED:
            if token not in before:
                continue
            (survived if token in after else translated).append(token)

        for token in translated:
            print(f'  translated  {token}')
        for token in survived:
            print(f'  SURVIVED    {token}')
            failures.append(f'{name}: {token} was not translated')

        shims = [t for t in CUDA_COMPAT_SHIMS if t in before]
        for token in shims:
            how = 'rewritten' if token not in after else 'kept'
            print(f'  shim        {token}  ({how})')

        if LDG_MACRO in before:
            print(f'  macro       {LDG_MACRO}  '
                  '(__ldg on NVIDIA, plain load on AMD -- see the note below)')

        portable = [t for t in KNOWN_PORTABLE if t in before]
        for token in portable:
            if token in after:
                print(f'  portable    {token}  (native in HIP, left as is)')
            else:
                failures.append(f'{name}: {token} was rewritten but should be portable')

        if not translated and not survived and not shims and not portable \
                and LDG_MACRO not in before:
            print('  no symbol from the watch lists appears in this file')

        if show_diff:
            diff = difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                fromfile=name, tofile=os.path.basename(hipified))
            sys.stdout.writelines(diff)

    return failures


def check_launch_form(mapping):
    """Every kernel launch must still carry an explicit stream after translation.

    A launch that reaches HIP on the null stream is the ROCm form of the
    default-stream bug.

    The spelling of a launch is not fixed, so this accepts all of it: hipify may
    turn the triple chevron into hipLaunchKernelGGL or leave it alone, and may
    wrap the call over several lines. What it requires is narrow and textual: the
    launch must name a current-stream getter under either spelling. A launch
    handed a stream by any other route -- a variable, getStreamFromPool() -- is
    rejected too, which is strict rather than wrong for this source.
    """
    hipified = mapping.get('swin_window_process_kernel.cu')
    if hipified is None:
        return ['no hipified kernel source to inspect']

    text = read(hipified)
    calls = re.findall(r'hipLaunchKernelGGL\s*\((.*?)\)\s*;', text, re.DOTALL)
    calls += re.findall(r'<<<(.*?)>>>', text, re.DOTALL)
    if not calls:
        return ['no kernel launch found in the translated source']

    stream_getters = ('getCurrentHIPStream', 'getCurrentCUDAStream')
    failures = []
    for call in calls:
        if not any(g in call for g in stream_getters):
            failures.append('launch without an explicit stream: '
                            + ' '.join(call.split())[:80])
    if not failures:
        print(f'\n{len(calls)} kernel launches, all carrying an explicit stream')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--diff', action='store_true',
                        help='print the generated HIP source as a unified diff')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging, mapping = hipify_into(tmp)
        failures = report(staging, mapping, args.diff)
        failures += check_launch_form(mapping)

    if failures:
        print('\nFAILED')
        for failure in failures:
            print(f'  {failure}')
        raise SystemExit(1)

    print('\nOK: the CUDA sources translate to HIP with no unmapped symbols.')
    print(SYMBOL_LEVEL_CAVEAT)


if __name__ == '__main__':
    main()
