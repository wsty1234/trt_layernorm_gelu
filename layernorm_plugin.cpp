#include "layernorm_plugin.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstring>

using namespace nvinfer1;

namespace comexample {

namespace {
static const char* PLUGIN_NAME = "LayerNorm";
static const char* PLUGIN_VERSION = "1";
}

LayerNormPlugin::LayerNormPlugin(float epsilon, std::string dataFormat, int axis)
    : mEps(epsilon), mDataFormat(std::move(dataFormat)), mAxis(axis) {}

LayerNormPlugin::LayerNormPlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    std::memcpy(&mEps, d, sizeof(mEps)); d += sizeof(mEps);
    int fmtLen; std::memcpy(&fmtLen, d, sizeof(int)); d += sizeof(int);
    mDataFormat.assign(d, d + fmtLen); d += fmtLen;
    std::memcpy(&mAxis, d, sizeof(mAxis)); d += sizeof(mAxis);
}

const char* LayerNormPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }
const char* LayerNormPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
int LayerNormPlugin::getNbOutputs() const noexcept { return 1; }
int LayerNormPlugin::initialize() noexcept { return 0; }
void LayerNormPlugin::terminate() noexcept {}
size_t LayerNormPlugin::getSerializationSize() const noexcept {
    return sizeof(mEps) + sizeof(int) + mDataFormat.size() + sizeof(mAxis);
}
void LayerNormPlugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    std::memcpy(d, &mEps, sizeof(mEps)); d += sizeof(mEps);
    int fmtLen = (int)mDataFormat.size();
    std::memcpy(d, &fmtLen, sizeof(int)); d += sizeof(int);
    std::memcpy(d, mDataFormat.data(), fmtLen); d += fmtLen;
    std::memcpy(d, &mAxis, sizeof(mAxis)); d += sizeof(mAxis);
}
void LayerNormPlugin::destroy() noexcept { delete this; }
IPluginV2DynamicExt* LayerNormPlugin::clone() const noexcept {
    auto* p = new LayerNormPlugin(mEps, mDataFormat, mAxis);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}
void LayerNormPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* LayerNormPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

DataType LayerNormPlugin::getOutputDataType(int, const DataType* inputTypes, int) const noexcept {
    return inputTypes[0];
}

DimsExprs LayerNormPlugin::getOutputDimensions(int, const DimsExprs* inputs, int, IExprBuilder&) noexcept {
    return inputs[0]; // same shape as input
}

bool LayerNormPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // inputs: x (0), gamma (1), beta (2); outputs: y (3)
    const PluginTensorDesc& d = inOut[pos];
    if (pos == 0) {
        return d.format == TensorFormat::kLINEAR && (d.type == DataType::kFLOAT || d.type == DataType::kHALF);
    }
    if (pos == 1 || pos == 2) {
        return d.format == inOut[0].format && d.type == inOut[0].type;
    }
    if (pos == 3) {
        return d.format == inOut[0].format && d.type == inOut[0].type;
    }
    return false;
}

void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc*, int, const DynamicPluginTensorDesc*, int) noexcept {}
size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc*, int, const PluginTensorDesc*, int) const noexcept { return 0; }

int LayerNormPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    const void* x = inputs[0];
    const void* gamma = inputs[1];
    const void* beta  = inputs[2];
    void* y = outputs[0];

    auto dt = inputDesc[0].type;
    auto dims = inputDesc[0].dims;

    if (mDataFormat == "channels_last") {
        int nbDims = dims.nbDims;
        int C = dims.d[nbDims - 1];
        int M = 1;
        for (int i=0; i<nbDims-1; ++i) M *= dims.d[i];

        dim3 grid(M);
        dim3 block(256);
        if (dt == DataType::kFLOAT) {
            layernorm_lastdim_kernel<float><<<grid, block, 0, stream>>>(
                (const float*)x, (const float*)gamma, (const float*)beta, (float*)y, M, C, mEps);
        } else {
            layernorm_lastdim_kernel<half><<<grid, block, 0, stream>>>(
                (const half*)x, (const half*)gamma, (const half*)beta, (half*)y, M, C, mEps);
        }
    } else {
        // channels_first: N,C,H,W
        if (dims.nbDims != 4) return 1;
        int N=dims.d[0], C=dims.d[1], H=dims.d[2], W=dims.d[3];
        int M = N*H*W;
        dim3 grid(M);
        dim3 block(256);
        if (dt == DataType::kFLOAT) {
            layernorm_channel_kernel<float><<<grid, block, 0, stream>>>(
                (const float*)x, (const float*)gamma, (const float*)beta, (float*)y, N,C,H,W, mEps);
        } else {
            layernorm_channel_kernel<half><<<grid, block, 0, stream>>>(
                (const half*)x, (const half*)gamma, (const half*)beta, (half*)y, N,C,H,W, mEps);
        }
    }
    return 0;
}

/************ Creator ************/
LayerNormPluginCreator::LayerNormPluginCreator() {
    mFields.clear();
    mFields.emplace_back(nvinfer1::PluginField{"epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    mFields.emplace_back(nvinfer1::PluginField{"data_format", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
    mFields.emplace_back(nvinfer1::PluginField{"axis", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFC.nbFields = (int)mFields.size();
    mFC.fields = mFields.data();
}
const char* LayerNormPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }
const char* LayerNormPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
const nvinfer1::PluginFieldCollection* LayerNormPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    float eps = 1e-6f;
    std::string fmt = "channels_last";
    int axis = -1;
    for (int i=0; i<fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (!strcmp(f.name, "epsilon") && f.type == nvinfer1::PluginFieldType::kFLOAT32) {
            eps = *reinterpret_cast<const float*>(f.data);
        } else if (!strcmp(f.name, "data_format") && f.type == nvinfer1::PluginFieldType::kCHAR) {
            fmt.assign(reinterpret_cast<const char*>(f.data), f.length);
        } else if (!strcmp(f.name, "axis") && f.type == nvinfer1::PluginFieldType::kINT32) {
            axis = *reinterpret_cast<const int*>(f.data);
        }
    }
    auto* p = new LayerNormPlugin(eps, fmt, axis);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::IPluginV2* LayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    auto* p = new LayerNormPlugin(serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}
void LayerNormPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }
const char* LayerNormPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace comexample