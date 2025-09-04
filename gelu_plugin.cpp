
#include "gelu_plugin.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstring>

using namespace nvinfer1;

namespace comexample {

namespace {
static const char* PLUGIN_NAME = "Gelu";
static const char* PLUGIN_VERSION = "1";
}

GELUPlugin::GELUPlugin(std::string approximate) : mApproximate(std::move(approximate)) {}

GELUPlugin::GELUPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    int len = 0;
    std::memcpy(&len, d, sizeof(int)); d += sizeof(int);
    mApproximate.assign(d, d + len); d += len;
}

const char* GELUPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }
const char* GELUPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
int GELUPlugin::getNbOutputs() const noexcept { return 1; }
int GELUPlugin::initialize() noexcept { return 0; }
void GELUPlugin::terminate() noexcept {}

size_t GELUPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) + mApproximate.size();
}

void GELUPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    int len = static_cast<int>(mApproximate.size());
    std::memcpy(d, &len, sizeof(int)); d += sizeof(int);
    std::memcpy(d, mApproximate.data(), len);
}

void GELUPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt* GELUPlugin::clone() const noexcept
{
    auto* p = new GELUPlugin(mApproximate);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void GELUPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* GELUPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

DataType GELUPlugin::getOutputDataType(int, const DataType* inputTypes, int) const noexcept
{
    return inputTypes[0];
}

DimsExprs GELUPlugin::getOutputDimensions(
    int, const DimsExprs* inputs, int, IExprBuilder&) noexcept
{
    return inputs[0];
}

bool GELUPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int, int) noexcept
{
    if (pos == 0) {
        return inOut[0].format == TensorFormat::kLINEAR &&
               (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF);
    }
    if (pos == 1) {
        return inOut[1].format == inOut[0].format && inOut[1].type == inOut[0].type;
    }
    return false;
}

void GELUPlugin::configurePlugin(
    const DynamicPluginTensorDesc*, int, const DynamicPluginTensorDesc*, int) noexcept {}

size_t GELUPlugin::getWorkspaceSize(
    const PluginTensorDesc*, int, const PluginTensorDesc*, int) const noexcept
{
    return 0;
}

int GELUPlugin::enqueue(
    const PluginTensorDesc* inputDesc, const PluginTensorDesc*, const void* const* inputs,
    void* const* outputs, void*, cudaStream_t stream) noexcept
{
    const void* x = inputs[0];
    void* y = outputs[0];

    auto dt = inputDesc[0].type;
    auto dims = inputDesc[0].dims;

    int n = 1;
    for (int i = 0; i < dims.nbDims; ++i) n *= dims.d[i];

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bool tanhApprox = (mApproximate == std::string("tanh"));

    if (dt == DataType::kFLOAT) {
        gelu_kernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(x), static_cast<float*>(y), n, tanhApprox);
    } else {
        gelu_kernel<half><<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(x), static_cast<half*>(y), n, tanhApprox);
    }
    return 0;
}

/******************** Creator ********************/
GELUPluginCreator::GELUPluginCreator()
{
    mFields.clear();
    mFields.emplace_back(nvinfer1::PluginField{"approximate", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
    mFC.nbFields = static_cast<int>(mFields.size());
    mFC.fields = mFields.data();
}

const char* GELUPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }
const char* GELUPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
const nvinfer1::PluginFieldCollection* GELUPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2* GELUPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    std::string approx = "erf";
    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (!strcmp(f.name, "approximate") && f.type == nvinfer1::PluginFieldType::kCHAR) {
            approx.assign(reinterpret_cast<const char*>(f.data), f.length);
        }
    }
    auto* p = new GELUPlugin(approx);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::IPluginV2* GELUPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    auto* p = new GELUPlugin(serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void GELUPluginCreator::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }
const char* GELUPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace comexample