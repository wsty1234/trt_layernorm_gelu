// file: plugin/layernorm_plugin.h
#pragma once
#include "NvInfer.h"
#include <string>
#include <vector>

namespace comexample {

class LayerNormPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    LayerNormPlugin(float epsilon, std::string dataFormat, int axis);
    LayerNormPlugin(const void* data, size_t length);

    // IPluginV2
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    float mEps;
    std::string mDataFormat; // "channels_last" | "channels_first"
    int mAxis;               // -1 or 1
    std::string mNamespace;
};

class LayerNormPluginCreator : public nvinfer1::IPluginCreator {
public:
    LayerNormPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFC;
};

} // namespace comexample