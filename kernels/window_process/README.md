# Fused window process kernels

`torch.roll` + `window_partition`, and its inverse, fused into a single pass.

Both are pure data movement, so the eager path materialises the tensor twice —
`torch.roll` writes a copy, then `window_partition` permutes and calls
`.contiguous()` for another — where the fused kernel writes once. Peak memory
drops with the copy that is no longer made.

Enable with `--fused_window_process` (see `get_started.md`), or by passing
`fused_window_process=True` to the model.

## Build

```bash
cd kernels/window_process
python setup.py install
```

The same command builds on ROCm; see [Building on AMD](#building-on-amd).

## Usage

```python
from window_process import WindowProcess, WindowProcessReverse

# (B, H, W, C) -> (B * nH * nW, window_h, window_w, C)
windows = WindowProcess.apply(x, B, H, W, C, -shift_size, window_size)

# and back
x = WindowProcessReverse.apply(windows, B, H, W, C, shift_size, window_size)
```

`shift_size` and `window_size` accept an `int` (isotropic) or an `(h, w)` pair:

```python
windows = WindowProcess.apply(x, B, H, W, C, (-2, -8), (4, 16))
```

`shift_size` is the negated `torch.roll` shift on the partition path, which is
why the call sites pass `-shift_size` forward and `+shift_size` back.

## Constraints

| | |
|---|---|
| dtype | float64, float32, float16, bfloat16 |
| layout | contiguous, channels last in memory: `(..., C)` |
| tiling | `H % window_h == 0` and `W % window_w == 0` |
| shape | `tensor.numel() == B * H * W * C` |
| shift | `abs(shift) < window size`, per axis |
| size | `B * H * W * C < 2^31` — offsets are computed in `int` |
| grid | `B * nH * nW <= 65535` and `H <= 65535` — CUDA `grid.z` and `grid.y` |

Shape, tiling, shift and size are checked with `TORCH_CHECK` before the launch;
device and contiguity by `CHECK_INPUT`. The channel-last layout is the shape
contract itself and is not separately verified.

Four of these are new guards over what was previously undetected: violating the
tiling, shape or size rules used to read the wrong element, or read out of
bounds, with no error raised anywhere, and exceeding the grid bounds used to
fail the launch asynchronously somewhere else. The `shift` row is different —
the kernels compute a larger roll correctly, for `window <= |shift| <= H` per
axis, and past that the operand of the modulo stops being non-negative and wraps
into a silently wrong read rather than an out-of-bounds one. It is enforced
because it is the contract the model already documents, not because it used to
break. dtype and layout were
already reported by the dispatch and by `CHECK_CONTIGUOUS`.

Note that `nH != nW` follows from `H != W` alone: with a square window,
`nH = H / window_size` and `nW = W / window_size`. A non-square window is not
required to reach that case, and `img_size` is documented as `int | tuple(int)`.

## Tests

| file | needs a GPU | what it covers |
|---|---|---|
| `test_index_math.py` | no | the index arithmetic of all four kernels, transcribed to PyTorch in `reference.py` |
| `hipify_check.py` | no | CUDA → HIP translation is complete, and every launch carries a stream |
| `unit_test.py` | yes | parity with the eager path across dtype × shape × shift, forward and backward |
| `test_model_parity.py` | yes | a `SwinTransformer` gives identical logits and gradients either way |
| `benchmark.py` | yes | wall time and peak memory, fused vs eager |

```bash
python test_index_math.py      # runs anywhere
python hipify_check.py         # runs anywhere
python unit_test.py            # skips without a GPU or the built extension
python test_model_parity.py
python benchmark.py --iters 200
```

The first two run in CI on a CPU runner, on every push that touches this
directory. `unit_test.py` and `test_model_parity.py` are executed there too, but
only to confirm they skip cleanly rather than error; `benchmark.py` is not, since
it exits non-zero without a device. The two CI legs pin different torch
versions, one that rewrites the stream getter and one that does not, so both
hipify behaviours are exercised on every run.

The kernels perform no arithmetic on the values, only gathers, so parity is
asserted with `torch.equal` on every dtype including float16 and bfloat16: any
deviation is an indexing error rather than a rounding one. For the same reason
`gradcheck` adds nothing, even though float64 does dispatch.

`reference.py` transcribes the kernels' `input_offset` computation to vectorised
PyTorch. That is not an approximation of the kernels — the offset computation is
their entire logic — which is what makes them testable on a CPU, with no
compiled extension and no GPU of either vendor. The two agree wherever the
operand of a modulo stays non-negative, which every input the launcher accepts
guarantees; outside that domain Python floors where the kernel wraps, and the
transcription is the more forgiving of the two.

## Validation

Parity is asserted against the eager path on an **RTX 3080** (torch
`2.8.0+cu129`, CUDA 12.9) and an **AMD Instinct MI300X** (gfx942, torch
`2.12.0+rocm7.14.0`): `unit_test.py` 15/15 test methods and
`test_model_parity.py` 4/4 on both, bit-exact, over square and non-square
images and windows including coprime `nH`/`nW`. `compute-sanitizer` (memcheck,
initcheck, synccheck) is clean on the RTX 3080 across all four kernels, forward
and backward; ROCm has no equivalent run. The MI300X run covers all 15 entries
in `unit_test.py`'s shape list; two of them were added after the RTX 3080 run
and have not been executed on NVIDIA.

## Performance

`benchmark.py`, forward + backward, batch 192, 200 iterations after 10 warm-up.
The `partition` direction is shown. The `merge` direction's fused time tracks it
within 6% at the configurations listed here on the RTX 3080, and within 1% at
every point of the full matrix on the MI300X. `benchmark.py` prints that matrix,
both directions and all four configurations.

**RTX 3080 (10 GB)**

| config | dtype | eager ms | fused ms | speedup | eager MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| stage 1 56×56 w7 | float32 | 7.137 | 2.398 | 2.98× | 1543.5 | 882.0 |
| stage 1 56×56 w7 | float16 | 5.223 | 2.083 | 2.51× | 771.8 | 441.0 |
| stage 1 56×56 w7 | bfloat16 | 5.074 | 1.948 | 2.60× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float32 | 3.711 | 1.156 | 3.21× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float16 | 2.552 | 0.700 | 3.65× | 392.0 | 224.0 |
| stage 2 28×28 w7 | bfloat16 | 2.783 | 0.726 | 3.84× | 392.0 | 224.0 |

Non-square configurations (32×16 with a square window, 16×64 with a 4×16 window)
land in the same 2.5–3.0× band.

**AMD Instinct MI300X**

| config | dtype | eager ms | fused ms | speedup | eager MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| stage 1 56×56 w7 | float32 | 0.911 | 0.598 | 1.52× | 1543.5 | 882.0 |
| stage 1 56×56 w7 | float16 | 0.652 | 0.588 | 1.11× | 771.8 | 441.0 |
| stage 1 56×56 w7 | bfloat16 | 0.646 | 0.589 | 1.10× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float32 | 0.449 | 0.188 | 2.38× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float16 | 0.323 | 0.152 | 2.12× | 392.0 | 224.0 |
| stage 2 28×28 w7 | bfloat16 | 0.320 | 0.152 | 2.10× | 392.0 | 224.0 |

Non-square configurations land in a 1.03–1.44× band.

Memory is identical across vendors, as it must be — the allocations are the
same. Time is not. The eager baseline is what moved: 7.137 ms at stage 1 on the
3080 against 0.911 ms here, so both paths compress toward a floor and the ratio
closes with them. The fused rows also stop scaling with dtype on the MI300X
(0.598 ms float32 against 0.588 float16, for half the bytes) while the eager
path still does, so at these sizes the fused kernel is no longer bandwidth bound
on CDNA. Block width is not the cause — that is measured below.

`bfloat16` is dispatched directly. The upstream kernel raises on a `bfloat16`
input, so the fused path previously needed an fp32 round trip.

## Building on AMD

`setup.py` needs no change. `CUDAExtension` hipifies its own sources when
`torch.version.hip` is set, substitutes `hipcc`, and derives `--offload-arch`
from `PYTORCH_ROCM_ARCH`:

```bash
PYTORCH_ROCM_ARCH=gfx942 python setup.py install
```

On AMD's ROCm 7.14 images that command fails until the SDK headers are in
place; the second bullet below has the two `export`s and the one package it
needs, and they have to come first.

Do not pass `--offload-arch` through `extra_compile_args`: PyTorch's
`_get_rocm_arch_flags()` skips its own detection as soon as it sees one, which
pins the build to a single GPU.

Two things are worth knowing before building:

- **`__ldg` has no HIP overload for `c10::Half`**, and half is in the dispatch.
  Reads go through the `SWIN_WP_LDG` macro for that reason: on NVIDIA it expands
  to the same `__ldg(ptr)` tokens as before, on AMD to a plain load. The hint is
  advisory on both, so no result changes. `hipify_check.py` cannot catch this
  and says so — it is a symbol-level check, and a symbol can translate cleanly
  and still not compile.

- **AMD's ROCm 7.14 PyTorch images ship the SDK as pip wheels** with no
  `/opt/rocm` tree, and those wheels carry no rocThrust, hipSPARSE, hipBLAS,
  hipBLASLt or hipSOLVER headers — which torch's own headers include when
  compiling device code. In that state *no* PyTorch HIP extension compiles; a
  file whose only content is `#include <ATen/ATen.h>` fails the same way. One
  dev metapackage supplies all of them, and touches no source:

  ```bash
  # Assumes AMD's Developer Cloud image, whose host already carries this repo's
  # signing key at /etc/apt/keyrings/amdrocm.gpg -- copy it into the container.
  # Anywhere else, install AMD's key under that path first: repo.amd.com serves
  # none over HTTP, and the repo.radeon.com key signs a different repository, so
  # apt-get update fails with NO_PUBKEY without it.
  echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main' \
       > /etc/apt/sources.list.d/rocm.list
  apt-get update && apt-get install -y amdrocm-core-dev7.14
  export CPLUS_INCLUDE_PATH=/opt/rocm/core-7.14/include
  export LIBRARY_PATH=/opt/rocm/core-7.14/lib
  ```

  `LIBRARY_PATH` is separate from the headers: the image ships
  `libamdhip64.so.7` with no development symlink, so the final link cannot
  resolve `-lamdhip64` without it. Two near misses are worth naming, because
  both fail far from their cause. Do not append `/usr/include` to
  `CPLUS_INCLUDE_PATH` — searching it ahead of the compiler's own directories
  defeats libstdc++'s `#include_next`, and the build dies on `stdlib.h: No such
  file or directory` before it reaches a single ROCm header. And do not reach
  for Ubuntu's `librocthrust-dev` / `librocprim-dev`: they are ROCm 5.7, and
  they install a 5.7 HIP into `/usr/include` that shadows the image's 7.14
  headers, producing a wall of errors inside `amd_warp_sync_functions.h`.

### Portability

The kernels use no shared memory, no `__syncthreads()` and no warp-level
primitives. Nothing depends on the warp or wavefront width, so CDNA's 64-wide
wavefront against NVIDIA's 32-wide warp affects occupancy and nothing else. The
three block widths in `best_block_dim()` are multiples of 64, so neither
platform schedules a partial wave.

Those thresholds were tuned on NVIDIA. What was measured on CDNA is the width
the heuristic actually picks at swin-tiny's `C` — 64 at both stages, since 96 and
192 are below the first threshold — by rebuilding with `-DSWIN_WP_BLOCK_DIM=N`
(stage 1 / stage 2, float32, MI300X, fused time):

| block width | stage 1 | stage 2 |
|---|---:|---:|
| 64 (what the heuristic picks here) | **0.598 ms** | **0.188 ms** |
| 256 | 0.632 ms | 0.212 ms |
| 1024 | 1.230 ms | 0.326 ms |

The NVIDIA-tuned choice wins on CDNA too, and widening hurts monotonically — at
1024 the fused path becomes *slower than eager* (0.74×). Both configurations sit
in the same branch, so the 384 and 1024 thresholds themselves remain untested on
CDNA. The grid is fixed by
the window geometry, so threads past `C` have nothing to do; swin-tiny's `C` is
96 at stage 1, and a 1024-wide block leaves 928 lanes idle.

The one substantive difference between the two platforms is the launch stream.
Upstream passes `0`, the null stream; this version passes
`at::cuda::getCurrentCUDAStream()`. Some hipify versions rewrite that to the HIP
spelling and some leave it, which resolves to the HIP runtime anyway — both
reach the current stream. torch 2.8 and 2.10 rewrite it; 2.12 and 2.13 leave it,
and 2.12 is the version the MI300X build used, so the kept spelling is the one
that compiled under hipcc and ran bit-exact above. `hipify_check.py` still makes
no prediction: it reports the one that happened rather than requiring either,
prints the full symbol mapping, and fails if anything is left untranslated.
