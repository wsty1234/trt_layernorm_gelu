# verfy_plugin.py
import os, ctypes, tensorrt as trt

so = os.path.abspath("./build/libcustom_trt_plugins.so")  # 改成你的真实路径
print("Loading:", so)
lib = ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
print("Loaded OK")


reg = trt.get_plugin_registry()

c = reg.get_plugin_creator("LayerNorm", "1", "com.example")
print("creator:", c)  # 
# for c in reg.plugin_creator_list:
#     print(f"name= {c.name} ver= {c.plugin_version} ns= {c.plugin_namespace}")


# 解析ONNX模型
def parse_onnx_model(onnx_model_path):
    # 创建TensorRT构建器和网络
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建ONNX解析器
    parser = trt.OnnxParser(network, logger)
    
    # 加载ONNX模型
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        input = network.get_input(i)
        print(f"  Input {i}: {input.name} - Shape: {input.shape}")
    
    print(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        output = network.get_output(i)
        print(f"  Output {i}: {output.name} - Shape: {output.shape}")
    
    print(f"Network layers: {network.num_layers}")
    return network, builder

# 构建TensorRT引擎
def build_engine(network, builder):
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    profile = builder.create_optimization_profile()
    # 为动态输入设置优化维度（根据您的模型调整）
    # 示例：profile.set_shape("input_name", min_shape, opt_shape, max_shape)
    
    if profile.nb_optimization_profiles > 0:
        config.add_optimization_profile(profile)
    
    engine = builder.build_engine(network, config)
    return engine

# 主函数
def main():

    
    # 如果您有ONNX模型文件，可以这样解析：
    onnx_model_path = "/root/quant/custom.onnx"
    network, builder = parse_onnx_model(onnx_model_path)
    if network:
        engine = build_engine(network, builder)
        if engine:
            print("Engine built successfully")
        else:
            print("Failed to build engine")

if __name__ == "__main__":
    main()