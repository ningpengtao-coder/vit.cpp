// vit_c.dll 冒烟测试：读取 JPEG/PNG，按 NCHW + ImageNet 归一化预处理后调用 C ABI。
// 只依赖 include/vit_c.h 与 stb_image，用于验证 ABI 契约本身。

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "vit_c.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace
{

void report_error(const char * stage, char * error)
{
    const char * message = (error != nullptr) ? error : vit_last_error();
    std::fprintf(stderr, "%s失败：%s\n", stage, message);
    vit_free_string(error);
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "用法: vit_c_smoke <model.gguf> <image>\n");
        return 2;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char * pixels = stbi_load(argv[2], &width, &height, &channels, 3);
    if (pixels == nullptr)
    {
        std::fprintf(stderr, "读取图像失败：%s\n", argv[2]);
        return 1;
    }

    char * error = nullptr;
    vit_c_model * model = vit_model_load(argv[1], 4, &error);
    if (model == nullptr)
    {
        report_error("加载模型", error);
        stbi_image_free(pixels);
        return 1;
    }

    const int32_t size = vit_model_image_size(model);
    const int32_t num_classes = vit_model_num_classes(model);

    // 与上游 vit_image_preprocess_bilinear 相同的缩放与归一化，但直接写成 NCHW。
    const float mean[3] = {123.675f, 116.280f, 103.530f};
    const float stdv[3] = {58.395f, 57.120f, 57.375f};
    const float x_scale = static_cast<float>(width) / size;
    const float y_scale = static_cast<float>(height) / size;

    std::vector<float> nchw(static_cast<size_t>(3) * size * size);
    for (int32_t y = 0; y < size; ++y)
    {
        for (int32_t x = 0; x < size; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                const float sx = (x + 0.5f) * x_scale - 0.5f;
                const float sy = (y + 0.5f) * y_scale - 0.5f;

                const int32_t x0 = std::max(0, static_cast<int32_t>(std::floor(sx)));
                const int32_t y0 = std::max(0, static_cast<int32_t>(std::floor(sy)));
                const int32_t x1 = std::min(x0 + 1, width - 1);
                const int32_t y1 = std::min(y0 + 1, height - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const float v00 = pixels[3 * (y0 * width + x0) + c];
                const float v01 = pixels[3 * (y0 * width + x1) + c];
                const float v10 = pixels[3 * (y1 * width + x0) + c];
                const float v11 = pixels[3 * (y1 * width + x1) + c];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;
                const float v = std::min(std::max(v0 * (1.0f - dy) + v1 * dy, 0.0f), 255.0f);

                nchw[static_cast<size_t>(c) * size * size + y * size + x] = (v - mean[c]) / stdv[c];
            }
        }
    }
    stbi_image_free(pixels);

    std::vector<float> logits(num_classes);
    const int32_t status = vit_classify(model, nchw.data(), logits.data(), num_classes, &error);
    if (status != VIT_C_OK)
    {
        report_error("推理", error);
        vit_model_free(model);
        return 1;
    }

    std::vector<int32_t> order(num_classes);
    std::iota(order.begin(), order.end(), 0);
    const int32_t topk = std::min<int32_t>(5, num_classes);
    std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                      [&logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    for (int32_t i = 0; i < topk; ++i)
    {
        const char * label = vit_model_class_label(model, order[i]);
        std::printf("%d\t%s\t%.6f\n", order[i], (label != nullptr) ? label : "(无标签)", logits[order[i]]);
    }
    std::printf("top1_index=%d\n", order[0]);

    vit_model_free(model);
    return 0;
}
