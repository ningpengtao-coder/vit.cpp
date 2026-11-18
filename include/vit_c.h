#ifndef VIT_C_H
#define VIT_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(VIT_C_SHARED)
#  if defined(VIT_C_BUILD)
#    define VIT_C_API __declspec(dllexport)
#  else
#    define VIT_C_API __declspec(dllimport)
#  endif
#else
#  define VIT_C_API
#endif

/* 不透明模型句柄，内部持有 GGUF 权重、推理上下文与线程数。 */
typedef struct vit_c_model vit_c_model;

/* 返回码。0 表示成功，其余为错误分类。 */
typedef enum vit_c_status
{
    VIT_C_OK = 0,                   /* 成功 */
    VIT_C_ERR_INVALID_ARGUMENT = 1, /* 参数非法（空指针、容量不足、线程数 <= 0 等） */
    VIT_C_ERR_IO = 2,               /* 模型文件不存在或不可读 */
    VIT_C_ERR_MODEL = 3,            /* GGUF 格式不受支持或权重损坏 */
    VIT_C_ERR_OOM = 4,              /* 内存分配失败 */
    VIT_C_ERR_INTERNAL = 5,         /* 计算图构建或推理内部错误 */
} vit_c_status;

/*
 * 加载 ViT GGUF 模型。
 *
 * 参数：
 *   path      模型文件路径（UTF-8 / 本地编码，交由 std::ifstream 打开）
 *   n_threads 推理线程数，必须 > 0；保存到句柄，后续 vit_classify 使用
 *   error     失败时返回堆分配的中文错误消息，可为 NULL；调用方用 vit_free_string 释放
 *
 * 返回：成功返回模型句柄；失败返回 NULL。加载过程中的进度日志由上游打印到 stdout/stderr。
 */
VIT_C_API vit_c_model * vit_model_load(const char * path, int32_t n_threads, char ** error);

/* 释放模型句柄。NULL 与重复释放同一非空句柄均为安全空操作（但释放后请置空指针）。 */
VIT_C_API void vit_model_free(vit_c_model * model);

/* 模型类别数，通常是 1000（ImageNet-1k）。model 为 NULL 时返回 0。 */
VIT_C_API int32_t vit_model_num_classes(const vit_c_model * model);

/* 模型期望的输入边长（正方形，224 / 384 等）。model 为 NULL 时返回 0。 */
VIT_C_API int32_t vit_model_image_size(const vit_c_model * model);

/* 模型 patch 大小（8 / 16 等）。model 为 NULL 时返回 0。 */
VIT_C_API int32_t vit_model_patch_size(const vit_c_model * model);

/*
 * 返回类别下标对应的标签字符串（借用语义，模型释放后失效）。
 * 下标越界或 model 为 NULL 时返回 NULL。
 */
VIT_C_API const char * vit_model_class_label(const vit_c_model * model, int32_t index);

/*
 * 对单张图执行分类推理，输出未经 softmax 的原始 logits。
 *
 * 输入由调用方（Dart / Python 等）完成预处理：
 *   nchw  按 NCHW 排列，长度必须为 3 * size * size
 *         其中 size = vit_model_image_size(model)
 *         值应为 ImageNet 归一化结果：(pixel/255 - mean) / std
 *         mean = {0.485, 0.456, 0.406}，std = {0.229, 0.224, 0.225}
 * 输出：
 *   logits            长度至少为 num_classes 的 float 缓冲区，由调用方拥有
 *   logits_capacity   缓冲区容量（float 个数），必须 >= num_classes
 * 线程：
 *   使用 vit_model_load 时传入的线程数；同一个模型句柄上的调用必须串行，
 *   不同模型句柄可以并发。
 *
 * 返回：VIT_C_OK 或错误码；错误消息通过 error 返回（可为 NULL）。
 */
VIT_C_API int32_t vit_classify(vit_c_model * model,
                               const float * nchw,
                               float * logits,
                               int32_t logits_capacity,
                               char ** error);

/*
 * 返回当前线程最近一次错误的借用指针，永不为 NULL。
 * 每次 C ABI 调用开始时清空；多线程互不干扰。
 */
VIT_C_API const char * vit_last_error(void);

/* 释放由本库分配的错误字符串（vit_model_load / vit_classify 的 error 出参）。 */
VIT_C_API void vit_free_string(char * text);

#ifdef __cplusplus
}
#endif

#endif /* VIT_C_H */
