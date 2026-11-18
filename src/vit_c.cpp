#include "vit_c.h"
#include "vit_c_internal.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace
{

const char * k_unknown_error = "未知错误";

thread_local std::string g_last_error;

char * duplicate_string(const std::string & text)
{
    char * buffer = static_cast<char *>(std::malloc(text.size() + 1));
    if (buffer == nullptr)
    {
        return nullptr;
    }
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return buffer;
}

char * take_last_error()
{
    const std::string text = g_last_error.empty() ? std::string(k_unknown_error) : g_last_error;
    return duplicate_string(text);
}

} // namespace

void vit_c_record_error(const char * message)
{
    g_last_error = (message != nullptr) ? message : "";
}

extern "C"
{

    VIT_C_API vit_c_model * vit_model_load(const char * path, int32_t n_threads, char ** error)
    {
        g_last_error.clear();
        if (error != nullptr)
        {
            *error = nullptr;
        }

        vit_c_model * model = vit_c_bridge_load(path, n_threads);
        if (model == nullptr && error != nullptr)
        {
            *error = take_last_error();
        }
        return model;
    }

    VIT_C_API void vit_model_free(vit_c_model * model)
    {
        vit_c_bridge_free(model);
    }

    VIT_C_API int32_t vit_model_num_classes(const vit_c_model * model)
    {
        return vit_c_bridge_num_classes(model);
    }

    VIT_C_API int32_t vit_model_image_size(const vit_c_model * model)
    {
        return vit_c_bridge_image_size(model);
    }

    VIT_C_API int32_t vit_model_patch_size(const vit_c_model * model)
    {
        return vit_c_bridge_patch_size(model);
    }

    VIT_C_API const char * vit_model_class_label(const vit_c_model * model, int32_t index)
    {
        return vit_c_bridge_class_label(model, index);
    }

    VIT_C_API int32_t vit_classify(vit_c_model * model,
                                   const float * nchw,
                                   float * logits,
                                   int32_t logits_capacity,
                                   char ** error)
    {
        g_last_error.clear();
        if (error != nullptr)
        {
            *error = nullptr;
        }

        const int32_t status = vit_c_bridge_classify(model, nchw, logits, logits_capacity);
        if (status != VIT_C_OK && error != nullptr)
        {
            *error = take_last_error();
        }
        return status;
    }

    VIT_C_API const char * vit_last_error(void)
    {
        return g_last_error.c_str();
    }

    VIT_C_API void vit_free_string(char * text)
    {
        std::free(text);
    }

} // extern "C"
