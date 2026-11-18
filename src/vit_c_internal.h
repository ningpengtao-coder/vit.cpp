#ifndef VIT_C_INTERNAL_H
#define VIT_C_INTERNAL_H

#include <cstdint>

/*
 * 内部桥接层。
 *
 * 注意：本头文件同时被 src/vit_c.cpp（C ABI）和 src/vit_c_bridge.cpp（包含上游 vit.h）使用，
 * 因此不能包含 include/vit_c.h —— 上游 vit.h 已有 C++ 的 vit_model_load，若同一翻译单元
 * 再声明同名 extern "C" 函数会与 C 链接名冲突。
 */

struct vit_c_model;

/* 与 include/vit_c.h 的 vit_c_status 数值保持一致；桥接层不能直接包含该头文件。 */
enum vit_c_bridge_status
{
    VIT_C_BRIDGE_OK = 0,
    VIT_C_BRIDGE_ERR_INVALID_ARGUMENT = 1,
    VIT_C_BRIDGE_ERR_IO = 2,
    VIT_C_BRIDGE_ERR_MODEL = 3,
    VIT_C_BRIDGE_ERR_OOM = 4,
    VIT_C_BRIDGE_ERR_INTERNAL = 5,
};

/* 返回码与 include/vit_c.h 的 vit_c_status 数值保持一致。 */
vit_c_model * vit_c_bridge_load(const char * path, int32_t n_threads);
void vit_c_bridge_free(vit_c_model * model);
int32_t vit_c_bridge_classify(vit_c_model * model, const float * nchw, float * logits, int32_t logits_capacity);
int32_t vit_c_bridge_num_classes(const vit_c_model * model);
int32_t vit_c_bridge_image_size(const vit_c_model * model);
int32_t vit_c_bridge_patch_size(const vit_c_model * model);
const char * vit_c_bridge_class_label(const vit_c_model * model, int32_t index);

/* 定义在 src/vit_c.cpp：写入当前线程的错误消息。 */
void vit_c_record_error(const char * message);

#endif /* VIT_C_INTERNAL_H */
