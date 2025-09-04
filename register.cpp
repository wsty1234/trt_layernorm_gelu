#include "NvInferPlugin.h"
#include "layernorm_plugin.h"
#include "gelu_plugin.h"
#include <cstdio>

using namespace nvinfer1;


extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace)
{
    std::fprintf(stderr, "[custom_trt_plugins] initLibNvInferPlugins called\n");
    auto* reg = getPluginRegistry();
    if (!reg) return false;

    static comexample::LayerNormPluginCreator lnCreator;
    static comexample::GELUPluginCreator geluCreator;

    static comexample::LayerNormPluginCreator lnCreator1;
    static comexample::GELUPluginCreator geluCreator1;

    lnCreator1.setPluginNamespace("");
    geluCreator1.setPluginNamespace("");

    lnCreator.setPluginNamespace("com.example");
    geluCreator.setPluginNamespace("com.example");

    reg->registerCreator(lnCreator, "com.example");
    reg->registerCreator(geluCreator, "com.example");

    reg->registerCreator(lnCreator1, "");
    reg->registerCreator(geluCreator1, "");

    std::fprintf(stderr, "[custom_trt_plugins] Registered plugins: %s v%s, %s v%s\n",
        lnCreator.getPluginName(), lnCreator.getPluginVersion(),
        geluCreator.getPluginName(), geluCreator.getPluginVersion());

    std::fprintf(stderr, "[custom_trt_plugins] registered namespace: %s\n", lnCreator.getPluginNamespace());

    return true;
}

// 添加一个全局构造函数来确保插件被注册
__attribute__((constructor))
void register_plugins_at_init() {
    std::fprintf(stderr, "[custom_trt_plugins] Global constructor called\n");
    initLibNvInferPlugins(nullptr, "com.example");
}