# vit.cpp

> 本目录是 vendored fork，新增了稳定的 C ABI（`vit_c.dll`）用于 Dart FFI / ctypes 调用。
> 上游来源、commit、许可证与本地改动见 [UPSTREAM.md](UPSTREAM.md)。

## 本地 fork 使用说明

### 目录

```
packages/vit.cpp/
  CMakeLists.txt
  include/vit_c.h          # C ABI（公开头文件）
  src/vit_c.cpp            # C ABI 实现 + 线程局部错误信息
  src/vit_c_bridge.cpp     # 上游 C++ API 桥接层
  src/vit_c_internal.h     # 桥接层内部声明
  tests/vit_c_smoke.cpp    # C ABI 冒烟程序（可选目标）
  vit.h / vit.cpp          # 上游推理实现（仅新增 apply_softmax 参数）
  convert-pth-to-ggml.py   # timm 权重转换
  quantize.cpp             # 量化 CLI
  ggml/                    # 内联 ggml 源码
```

### 构建（Windows / MSVC）

依赖 CMake 3.20+ 与 Visual Studio 2022（x64）。ggml 已内联，无需再初始化 submodule。

```powershell
cmake -S packages/vit.cpp -B packages/vit.cpp/build `
      -G "Visual Studio 17 2022" -A x64 `
      -DBUILD_SHARED_LIBS=OFF -DVIT_BUILD_SMOKE=ON
cmake --build packages/vit.cpp/build --config Release --parallel 8
```

产物：

| 路径 | 说明 |
| --- | --- |
| `build/bin/Release/vit.exe` | 上游 CLI（分类并打印 top-k 概率） |
| `build/bin/Release/quantize.exe` | 量化 CLI |
| `build/bin/Release/vit_c.dll` | C ABI 共享库 |
| `build/Release/vit_c.lib` | MSVC 导入库 |
| `build/bin/Release/vit_c_smoke.exe` | C ABI 冒烟程序（`VIT_BUILD_SMOKE=ON`） |

`-DBUILD_SHARED_LIBS=OFF` 让 ggml 静态链接进 `vit_c.dll`，运行时不需要额外的 `ggml.dll`。
构建选项：`VIT_BUILD_C_API`（默认 `ON`）、`VIT_BUILD_SMOKE`（默认 `OFF`）、
`BUILD_QUANTIZE`（默认 `ON`）、`BUILD_BENCHMARK`（默认 `OFF`）。

macOS / Linux 使用同一套 CMake 命令，去掉 `-G` / `-A` 即可；非 MSVC 会追加
`-O3 -march=native`。

### C ABI

完整声明见 [`include/vit_c.h`](include/vit_c.h)。

| 函数 | 说明 |
| --- | --- |
| `vit_model_load(path, n_threads, error)` | 加载模型，返回句柄；失败返回 `NULL` 并给出中文错误 |
| `vit_model_free(model)` | 释放句柄；`NULL` 为安全空操作 |
| `vit_classify(model, nchw, logits, capacity, error)` | NCHW float32 输入，输出 1000 类原始 logits |
| `vit_model_num_classes(model)` | 类别数（ViT-ImageNet1k 为 1000） |
| `vit_model_image_size(model)` | 输入边长（224 / 384 等） |
| `vit_model_patch_size(model)` | patch 大小（16 等） |
| `vit_model_class_label(model, index)` | 类别标签（借用指针，越界返回 `NULL`） |
| `vit_last_error()` | 当前线程最近一次错误（借用指针，永不为 `NULL`） |
| `vit_free_string(text)` | 释放 `error` 出参 |

约定：

- **输入**：`nchw` 按 `[C=3, H=size, W=size]` 排列，长度 `3 * size * size`；
  调用方负责 resize 与 ImageNet 归一化：
  `mean = {0.485, 0.456, 0.406}`、`std = {0.229, 0.224, 0.225}`。
- **输出**：`logits` 为未经 softmax 的原始分数，容量必须 `>= num_classes`；
  需要概率时由调用方自行 softmax。
- **线程**：`n_threads` 在 `vit_model_load` 时保存；同一句柄的推理必须串行，不同句柄可并发。
- **内存**：输入 / 输出缓冲区由调用方拥有；只有 `error` 出参需要 `vit_free_string` 释放。
- **错误**：返回码 `0` 成功，`1` 参数非法，`2` 文件不可读，`3` 模型格式错误，
  `4` 内存不足，`5` 推理内部错误。

最小 C 示例：

```c
#include "vit_c.h"

void run(const char * model_path, const float * nchw, float * logits) {
    char * err = NULL;
    vit_c_model * model = vit_model_load(model_path, 4, &err);
    if (model == NULL) {
        fprintf(stderr, "加载失败：%s\n", err != NULL ? err : vit_last_error());
        vit_free_string(err);
        return;
    }

    const int32_t classes = vit_model_num_classes(model);
    if (vit_classify(model, nchw, logits, classes, &err) != VIT_C_OK) {
        fprintf(stderr, "推理失败：%s\n", err != NULL ? err : vit_last_error());
        vit_free_string(err);
    }

    vit_model_free(model);
}
```

冒烟程序用法（内部完成 bilinear resize + 归一化，仅用于验证 ABI）：

```powershell
packages/vit.cpp/build/bin/Release/vit_c_smoke.exe model.gguf assets/tench.jpg
```

### 转换与量化

转换依赖 `torch` + `timm`（本机验证版本：torch 2.15.0+cpu、timm 1.0.31）。
输出文件固定写到当前目录的 `ggml-model-{f32,f16}.gguf`：

```powershell
cd packages/vit.cpp/build/models
python ..\..\convert-pth-to-ggml.py --model_name vit_tiny_patch16_224.augreg_in21k_ft_in1k --ftype 1
```

量化类型：`2=q4_0`、`3=q4_1`、`6=q5_0`、`7=q5_1`、`8=q8_0`：

```powershell
..\bin\Release\quantize.exe ggml-model-f16.gguf ggml-model-q8_0.gguf 8
```

F16 / Q8_0 冒烟（2026-11-15，MSVC 19.44，`assets/tench.jpg`，4 线程）：

| 模型 | 权重 / 文件大小 | CLI top-1 | C ABI top-1 logits |
| --- | --- | --- | --- |
| F16 | 11.04 MB / 11.61 MB（152 tensors） | `tench` 0.74 | `0 tench`，12.971826 |
| Q8_0 | 6.12 MB / 6.45 MB（152 tensors） | `tench` 0.74 | `0 tench`，12.977290 |

### 路由约定：纯 ViT 模型统一走 vit.cpp

- **纯 ViT 分类 / 特征模型**（timm 的 `vit_*` 系列、只有 patch embedding + Transformer
  encoder + 分类头的模型）统一走 `vit.cpp`，不新增 ONNX 运行时。
- **CLIP / Chinese-CLIP 双塔**、**EmbeddingGemma**、**Qwen3-VL 等 VLM 的视觉塔**不属于这里：
  它们属于各自的双塔 / 多模态运行时，分别由 `packages/clip.cpp`、`packages/llama.cpp` 承担。
- **CNN 模型族**（PP-OCR、SCRFD、ArcFace、MobileNetV2）走 `packages/vision.cpp`。
- 当前 App 没有现网纯 ViT 模型，本包只建立运行时与 C ABI 约定，未接入任何业务路由。
- 接入时约定：Dart 侧完成 resize + 归一化并传入 NCHW float32；模型文件由调用方分发，
  F16 为默认，Q8_0 用于体积受限场景。

Inference Vision Transformer (ViT) in plain C/C++ using ggml without any extra dependencies

## Description


This project presents a standalone implementation of the well known Vision Transformer (ViT) model family, used in a broad spectrum of applications and SOTA models like Large Multimodal Models(LMM). The primary goal is to develop a C/C++ inference engine tailored for ViT models, utilizing [ggml](https://github.com/ggerganov/ggml) to enhance performance, particularly on edge devices. Designed to be both lightweight and self-contained, this implementation can be run across diverse platforms.

<details>
<summary>Table of Contents</summary>

1. [Description](#Description)
2. [Features](#features)
3. [Vision Transformer Architecture](#vision-transformer-architecture)
4. [Quick Example](#quick-example)
5. [Convert PyTorch to GGUF](#convert-pytorch-to-gguf)
6. [Build](#build)
   - [Simple Build](#simple-build)
   - [Per Device Optimizations](#per-device-optimizations)
   - [OpenMP](#using-openmp)
7. [Run](#run)
8. [Benchmark against PyTorch](#benchmark-against-pytorch)
   - [ViT Inference](#vit-inference)
   - [Benchmark on Your Machine](#benchmark-on-your-machine)
9. [Quantization](#quantization)
10. [To-Do List](#to-do-list)
</details>


## Features


- Dependency-free and lightweight inference thanks to [ggml](https://github.com/ggerganov/ggml).
- 4-bit, 5-bit and 8-bit quantization support.
- Support for timm ViTs with different variants out of the box.

An important aspect of using `vit.cpp` is that it has short startup times compared to common DL frameworks, which makes it suitable for serverless deployments where the cold start is an issue.

## Vision Transformer architecture


The implemented architecture is based on the original Vision Transformer from:
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

<p align="center">
  <img src="assets/image.png" alt="Vision Transformer overview" width="60%" height="auto">
</p>
<p align="center">
  ViT architecture. Taken from the <a href="https://arxiv.org/abs/2010.11929">original paper</a>.
</p>

## Quick example


<details>
  <p align="center">
    <img src="assets/magpie.jpeg" alt="example input" width="50%" height="auto">
  </p>

  <summary>See output</summary>
  <pre>
  $ ./bin/vit -t 4 -m ../ggml-model-f16.gguf -i ../assets/magpie.jpeg -k 5
  main: seed = 1701176263
  main: n_threads = 4 / 8
  vit_model_load: loading model from &apos;../ggml-model-f16.gguf&apos; - please wait
  vit_model_load: hidden_size            = 192
  vit_model_load: num_hidden_layers      = 12
  vit_model_load: num_attention_heads    = 3
  vit_model_load: patch_size             = 16
  vit_model_load: img_size               = 224
  vit_model_load: num_classes            = 1000
  vit_model_load: ftype                  = 1
  vit_model_load: qntvr                  = 0
  operator(): ggml ctx size =  11.13 MB
  vit_model_load: ................... done
  vit_model_load: model size =    11.04 MB / num tensors = 152
  main: loaded image &apos;../assets/magpie.jpeg&apos; (500 x 470)
  vit_image_preprocess: scale = 2.232143
  processed, out dims : (224 x 224)


  &gt; magpie : 0.87
  &gt; goose : 0.02
  &gt; toucan : 0.01
  &gt; drake : 0.01
  &gt; king penguin, Aptenodytes patagonica : 0.01


  main:    model load time =    17.92 ms
  main:    processing time =   146.96 ms
  main:    total time      =   164.88 ms
  </pre>
</details>


## Convert PyTorch to GGUF


    # clone the repo recursively
    git clone --recurse-submodules https://github.com/staghado/vit.cpp.git

    cd vit.cpp

    # install torch and timm
    pip install torch timm

    # list available models if needed; note that not all models are supported
    python convert-pth-to-ggml.py --list

    # convert the weights to gguf : vit tiny with patch size of 16 and an image 
    # size of 384 pre-trained on ImageNet21k and fine-tuned on ImageNet1k
    python convert-pth-to-ggml.py --model_name vit_tiny_patch16_384.augreg_in21k_ft_in1k --ftype 1

> **Note:** You can also download the converted weights from [Hugging Face](https://huggingface.co/staghado/vit.cpp) directly.

> ```wget https://huggingface.co/staghado/vit.cpp/blob/main/tiny-ggml-model-f16.gguf```


## Build


### Simple build


    # build ggml and vit 
    mkdir build && cd build
    cmake .. && make -j4

    # run inference
    ./bin/vit -t 4 -m ../ggml-model-f16.gguf -i ../assets/tench.jpg

The optimal number of threads to use depends on many factors and more is not always better. Usually using a number of threads equal to the number of available physical cores gives the best performance in terms of speed.

### Per device optimizations


Generate per-device instructions that work best for the given machine rather than using general CPU instructions.
This can be done by specifying `-march=native` in the compiler flags.
  * Multi-threading and vectorization
  * Loop transformations(unrolling)

#### For AMD host processors


You can use a specialized compiler released by AMD to make full use of your specific processor's architecture.
Read more here : [AMD Optimizing C/C++ and Fortran Compilers (AOCC)](https://www.amd.com/en/developer/aocc.html)

You can follow the given instructions to install the AOCC compiler.

Note : For my AMD Ryzen 7 3700U, the improvements were not very significant but for more recent processors there could be a gain in using a specialized compiler.

### Using OpenMP


Additionally compile with OpenMP by specifying the `-fopenmp` flag to the compiler in the CMakeLists file,
allowing multithreaded runs. Make sure to also enable multiple threads when running, e.g.:

    OMP_NUM_THREADS=4 ./bin/vit -t 4 -m ../ggml-model-f16.bin -i ../assets/tench.jpg

## Run


    usage: ./bin/vit [options]

    options:
      -h, --help              show this help message and exit
      -s SEED, --seed SEED    RNG seed (default: -1)
      -t N, --threads N       number of threads to use during computation (default: 4)
      -m FNAME, --model FNAME model path (default: ../ggml-model-f16.bin)
      -i FNAME, --inp FNAME   input file (default: ../assets/tench.jpg)
      -k N, --topk N          top k classes to print (default: 5)
      -e FLOAT, --epsilon     epsilon (default: 0.000001)


## Benchmark against PyTorch


First experiments on Apple M1 show inference speedups(up to 6x faster for base model) compared to native PyTorch inference. 

### ViT inference


You can efficiently run ViT inference on the CPU.
Memory requirements and inference speed on AMD Ryzen 7 3700U(4 cores, 8 threads) for both native PyTorch and `vit.cpp`. 
Using 4 threads gives better results for my machine. The reported results of inference speed correspond to 10 runs averages for both PyTorch and `vit.cpp`.

| Model  | Max Mem(PyTorch)  | Max Mem            | Speed(PyTorch) | Speed          |
| :----: | :-----------: | :------------: | :------------: | :------------: |
| tiny   | ~780 MB       | **~20 MB**     | 431 ms         | **120 ms**     |
| small  | ~965 MB       | **~52 MB**     | 780 ms         | **463 ms**     |
| base   | ~1.61 GB      | **~179 MB**    | 2393 ms        | **1441 ms**    |
| large  | ~3.86 GB      | **~597 MB**    | 8151 ms        | **4892 ms**    |

> **Note:** The models used are of the form `vit_{size}_patch16_224.augreg_in21k_ft_in1k`.

### Benchmark on your machine


In order to test the inference speed on your machine, you can run the following scripts:

    chmod +x scripts/benchmark.*

    # install memory_profiler & threadpoolctl
    pip install memory_profiler threadpoolctl

    # run the benchmark of PyTorch
    python scripts/benchmark.py

    # run the benchmark of vit.cpp for non-qunatized model
    ./scripts/benchmark.sh

    # to run the benchamrk for qunatized models; 4 threads and quantize flag
    ./scripts/benchmark.sh 4 1

Both scripts use 4 threads by default. In Python, the `threadpoolctl` library is used to limit the number of threads used by PyTorch.

## Quantization


`vit.cpp` supports many quantization strategies from ggml such as q4_0, q4_1, q5_0, q5_1 and q8_0 types.
You can quantize a model in F32 (the patch embedding is in F16) to one of these types by using the `./bin/quantize` binary. 
```
usage: ./bin/quantize /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf type                              
  type = 2 - q4_0                                                                                                       
  type = 3 - q4_1                                                                                                       
  type = 6 - q5_0                                                                                                       
  type = 7 - q5_1                                                                                                       
  type = 8 - q8_0                                                                                                       
```

For example, you can run the following to convert the model to q5_1:

```shell
./bin/quantize ../tiny-ggml-model-f16.gguf ../tiny-ggml-model-f16-quant.gguf 7
```

Then you can use `tiny-ggml-model-f16-quant.gguf` just like the model in F16.

### Results

Here are the benchmarks for the different models and quantizations on my machine: 
For accurate estimation of run times, these benchmarks were run 100 times each.  

| Model  | Quantization | Speed (ms)    | Mem (MB)          |
| :----: | :----------: | :-----------: | :---------------: |
|   tiny |     q4_0     |    105 ms     |   12 MB        |
|   tiny |     q4_1     |    97 ms     |   12 MB        |
|   tiny |     q5_0     |    116 ms     |   13 MB        |
|   tiny |     q5_1     |    112 ms     |   13 MB        |
|   tiny |     q8_0     |    90 ms     |   15 MB        |
|   small |     q4_0     |    240 ms     |   23 MB        |
|   small |     q4_1     |    224 ms     |   24 MB        |
|   small |     q5_0     |    288 ms     |   25 MB        |
|   small |     q5_1     |    277 ms     |   27 MB        |
|   small |     q8_0     |    228 ms     |   33 MB        |
|   base |     q4_0     |    704 ms     |   61 MB        |
|   base |     q4_1     |    626 ms     |   66 MB        |
|   base |     q5_0     |    851 ms     |   71 MB        |
|   base |     q5_1     |    806 ms     |   76 MB        |
|   base |     q8_0     |    659 ms     |   102 MB        |
|   large |     q4_0     |    2189 ms     |   181 MB        |
|   large |     q4_1     |    1919 ms     |   199 MB        |
|   large |     q5_0     |    2676 ms     |   217 MB        |
|   large |     q5_1     |    2547 ms     |   235 MB        |
|   large |     q8_0     |    1994 ms     |   325 MB        |

## To-Do List


- **Evaluate performance on ImageNet1k**: 

  Run evaluation on ImageNet1k test set and analyze the performance of different quantization schemes.

This project was highly inspired by the following projects:
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=staghado/vit.cpp&type=Date)](https://star-history.com/#staghado/vit.cpp&Date)
