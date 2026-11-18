// C ABI 与上游 C++ 实现之间的桥接层。
//
// 本文件是唯一同时接触上游 vit.h（C++ API）与 C ABI 内部结构的翻译单元。
// 由于上游已有 C++ 函数 vit_model_load，这里只导出带 vit_c_bridge_ 前缀的 C++ 符号，
// 真正带 C 链接的 vit_model_load 定义在 src/vit_c.cpp。

#include "vit.h"

#include "vit_c_internal.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <vector>

struct vit_c_model
{
    vit_model model;
    vit_state state;
    int32_t n_threads = 4;
};

namespace
{

void set_error(const std::string & message)
{
    vit_c_record_error(message.c_str());
}

// 与上游 main.cpp 中 state.context 的尺寸保持一致。
const size_t k_state_ctx_size = 3u * 1024 * 1024;

const size_t k_tensor_alignment = 32;

} // namespace

vit_c_model * vit_c_bridge_load(const char * path, int32_t n_threads)
{
    if (path == nullptr || path[0] == '\0')
    {
        set_error("模型路径为空");
        return nullptr;
    }
    if (n_threads <= 0)
    {
        set_error("线程数必须为正数");
        return nullptr;
    }

    {
        std::ifstream probe(path, std::ios::binary);
        if (!probe)
        {
            set_error(std::string("模型文件不存在或不可读：") + path);
            return nullptr;
        }
    }

    // Windows 上 ggml_time_us 依赖 ggml_time_init 初始化计时频率。
    ggml_time_init();

    vit_c_model * ctx = new (std::nothrow) vit_c_model();
    if (ctx == nullptr)
    {
        set_error("内存不足：无法创建模型句柄");
        return nullptr;
    }
    ctx->state.ctx = nullptr;
    ctx->state.prediction = nullptr;
    ctx->state.gallocr = nullptr;

    if (!vit_model_load(path, ctx->model))
    {
        if (ctx->model.ctx != nullptr)
        {
            ggml_free(ctx->model.ctx);
            ctx->model.ctx = nullptr;
        }
        delete ctx;
        set_error(std::string("加载模型失败：文件格式不受支持或已损坏（") + path + "）");
        return nullptr;
    }

    ctx->n_threads = n_threads;

    struct ggml_init_params params = {
        /*.mem_size   =*/k_state_ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };
    ctx->state.ctx = ggml_init(params);
    if (ctx->state.ctx == nullptr)
    {
        ggml_free(ctx->model.ctx);
        delete ctx;
        set_error("内存不足：无法创建推理上下文");
        return nullptr;
    }

    ctx->state.prediction = ggml_new_tensor_4d(ctx->state.ctx, GGML_TYPE_F32, ctx->model.hparams.num_classes, 1, 1, 1);
    if (ctx->state.prediction == nullptr)
    {
        ggml_free(ctx->state.ctx);
        ggml_free(ctx->model.ctx);
        delete ctx;
        set_error("内存不足：无法创建输出张量");
        return nullptr;
    }

    return ctx;
}

void vit_c_bridge_free(vit_c_model * model)
{
    if (model == nullptr)
    {
        return;
    }
    if (model->state.gallocr != nullptr)
    {
        ggml_gallocr_free(model->state.gallocr);
        model->state.gallocr = nullptr;
    }
    if (model->state.ctx != nullptr)
    {
        ggml_free(model->state.ctx);
        model->state.ctx = nullptr;
    }
    if (model->model.ctx != nullptr)
    {
        ggml_free(model->model.ctx);
        model->model.ctx = nullptr;
    }
    delete model;
}

int32_t vit_c_bridge_num_classes(const vit_c_model * model)
{
    return (model != nullptr) ? model->model.hparams.num_classes : 0;
}

int32_t vit_c_bridge_image_size(const vit_c_model * model)
{
    return (model != nullptr) ? model->model.hparams.img_size : 0;
}

int32_t vit_c_bridge_patch_size(const vit_c_model * model)
{
    return (model != nullptr) ? model->model.hparams.patch_size : 0;
}

const char * vit_c_bridge_class_label(const vit_c_model * model, int32_t index)
{
    if (model == nullptr)
    {
        return nullptr;
    }
    const auto it = model->model.hparams.id2label.find(index);
    if (it == model->model.hparams.id2label.end())
    {
        return nullptr;
    }
    return it->second.c_str();
}

int32_t vit_c_bridge_classify(vit_c_model * model, const float * nchw, float * logits, int32_t logits_capacity)
{
    if (model == nullptr)
    {
        set_error("模型句柄为空");
        return VIT_C_BRIDGE_ERR_INVALID_ARGUMENT;
    }
    if (nchw == nullptr)
    {
        set_error("输入指针为空");
        return VIT_C_BRIDGE_ERR_INVALID_ARGUMENT;
    }
    if (logits == nullptr)
    {
        set_error("输出 logits 指针为空");
        return VIT_C_BRIDGE_ERR_INVALID_ARGUMENT;
    }

    const int32_t num_classes = model->model.hparams.num_classes;
    if (logits_capacity < num_classes)
    {
        set_error("输出缓冲区容量不足：需要 " + std::to_string(num_classes) +
                  " 个 float，实际 " + std::to_string(logits_capacity));
        return VIT_C_BRIDGE_ERR_INVALID_ARGUMENT;
    }

    const int32_t n_img_size = model->model.hparams.img_size;
    const int32_t n_pixels = n_img_size * n_img_size;

    // 上游 image_f32 使用 HWC 交错布局，这里把调用方的 NCHW 重排过去。
    image_f32 img;
    img.nx = n_img_size;
    img.ny = n_img_size;
    img.data.resize(static_cast<size_t>(3) * n_pixels);
    for (int32_t y = 0; y < n_img_size; ++y)
    {
        for (int32_t x = 0; x < n_img_size; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                img.data[3 * (y * n_img_size + x) + c] =
                    nchw[static_cast<size_t>(c) * n_pixels + y * n_img_size + x];
            }
        }
    }

    // 图分配器按 CPU 缓冲类型创建，模型句柄存活期间复用。
    if (model->state.gallocr == nullptr)
    {
        model->state.gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (model->state.gallocr == nullptr)
        {
            set_error("推理失败：无法创建图分配器");
            return VIT_C_BRIDGE_ERR_INTERNAL;
        }
    }

    // 元数据（张量描述/图结构）放在预分配的 no_alloc 上下文里。
    model->state.buf_compute_img_enc.resize(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());

    struct ggml_tensor * inp = nullptr;
    struct ggml_cgraph * gf = vit_encode_image(model->model, model->state, img, false, &inp);
    if (gf == nullptr)
    {
        set_error("推理失败：无法构建计算图");
        return VIT_C_BRIDGE_ERR_INTERNAL;
    }

    if (!ggml_gallocr_alloc_graph(model->state.gallocr, gf))
    {
        set_error("推理失败：无法分配计算图内存");
        return VIT_C_BRIDGE_ERR_INTERNAL;
    }

    vit_set_input(inp, img);
    ggml_graph_compute_helper(model->state.work_buffer, gf, model->n_threads);

    model->state.work_buffer.clear();

    const float * output = ggml_get_data_f32(model->state.prediction);
    if (output == nullptr)
    {
        set_error("推理失败：输出张量无数据");
        return VIT_C_BRIDGE_ERR_INTERNAL;
    }

    std::memcpy(logits, output, sizeof(float) * num_classes);
    return VIT_C_BRIDGE_OK;
}
