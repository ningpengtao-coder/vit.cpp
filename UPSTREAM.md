# vit.cpp (vendored)

本目录是 [vit.cpp](https://github.com/staghado/vit.cpp) 的 vendored 副本，并作为
[ningpengtao-coder/vit.cpp](https://github.com/ningpengtao-coder/vit.cpp) fork 的落地版本。
上游是只依赖 ggml 的纯 C/C++ Vision Transformer 推理实现，用于 ImageNet 分类类 ViT 模型。

## Upstream

- Fork 仓库：https://github.com/ningpengtao-coder/vit.cpp
- Fork commit：`a4841f6eeabb91c917e94b8f8f8d4f9e01043fbc`（`Add smoll README`）
- 原始上游：https://github.com/staghado/vit.cpp
- 许可证：MIT（见 [LICENSE](LICENSE)，Copyright 2023 Said Taghadouini）
- ggml：已内联到 [ggml/](ggml)，来自 `https://github.com/ggerganov/ggml`，
  commit `a5e4560425f83797327001f12fb780cf33606d87`（上游 `.gitmodules` 指定的版本）。
  克隆完成后已删除 `vit.cpp/.git` 与 `ggml/.git`，`.gitmodules` 仅作为来源记录保留。

### 拉取记录

本机直连 `github.com:443` 不可用，clone 通过 GitHub 镜像代理完成（文件内容与 commit 一致）：

```powershell
git clone --depth 1 https://gh-proxy.com/https://github.com/ningpengtao-coder/vit.cpp.git packages/vit.cpp
git -c url."https://gh-proxy.com/https://github.com/".insteadOf="https://github.com/" `
    -C packages/vit.cpp submodule update --init --depth 1
```

## Local changes

只补 C ABI、Windows/MSVC 可构建性与冒烟工具，未改变上游 CLI 参数与 timm 转换语义。

### 新增文件

- `include/vit_c.h`：稳定 C ABI（加载 / 释放 / logits 推理 / 错误信息）。
- `src/vit_c.cpp`：C ABI 实现与线程局部错误缓冲区。
- `src/vit_c_bridge.cpp` / `src/vit_c_internal.h`：C ABI 与上游 C++ API 的桥接层。
  上游 `vit.h` 已有 C++ 的 `vit_model_load`，因此 C 链接实现必须放在不包含 `vit.h`
  的翻译单元，桥接层使用 `vit_c_bridge_*` 前缀。
- `tests/vit_c_smoke.cpp`：只依赖 `vit_c.h` 的 NCHW 冒烟程序（可选目标 `VIT_BUILD_SMOKE`）。

### 修改文件

- `CMakeLists.txt`
  - `-O3 -march=native` 仅在非 MSVC 下追加；MSVC 使用 Release 默认 `/O2`。
  - 新增共享库目标 `vit_c`（`vit.cpp` + `src/vit_c.cpp` + `src/vit_c_bridge.cpp`）。
    MSVC 上开启 `WINDOWS_EXPORT_ALL_SYMBOLS`，同时导出 C ABI 与上游 C++ 符号（供冒烟程序复用）。
  - 新增 `VIT_BUILD_C_API`（默认 ON）与 `VIT_BUILD_SMOKE`（默认 OFF）选项。
- `vit.h` / `vit.cpp`
  - `vit_encode_image()` 增加 `bool apply_softmax = true` 参数。默认值保持上游 CLI 行为
    （输出 softmax 概率），C ABI 传 `false` 以获得未经 softmax 的 1000 类原始 logits。
- `main.cpp`
  - 在首次 `ggml_time_us()` 前调用 `ggml_time_init()`。Windows 上未初始化计时频率会
    因除零（0xC0000094）直接崩溃；CLI 参数与输出格式不变。
  - `quantize.cpp` 上游已经在计时前用一次空的 `ggml_init()` 初始化了计时器，无需改动。

## Known limitations

- C ABI 只支持单图、NCHW float32 输入；预处理（resize + ImageNet 归一化）由调用方完成。
- 同一个模型句柄上的 `vit_classify` 必须串行调用；不同句柄可并发。
- 仅覆盖上游支持的 ViT 结构（patch embedding + Pre-LN Transformer block + GELU MLP + CLS 头），
  不支持 QK-norm、SwiGLU、register token 等变体。
- 模型为上游自定义 ggml 二进制格式（magic `ggml`，非 GGUF），扩展名沿用 `*.gguf`。
- 量化只量化 2D 权重（`.*weight` 且 `n_dims == 2`），LayerNorm / bias / pos_embed / cls_token
  保持原类型，与上游 `quantize.cpp` 一致。
