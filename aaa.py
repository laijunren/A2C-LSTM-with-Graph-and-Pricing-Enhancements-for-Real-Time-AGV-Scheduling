import torch, gc


print(f"Torch版本: {torch.__version__}")
print(f"Torch配置信息: {torch.__config__}")
print(f"Torch所有可用设备: {torch.cuda.device_count()}")
print(f"当前设备索引:{torch.cuda.current_device()}")
print(f"当前设备的名称: {torch.cuda.get_device_name()}")
print(f"当前设备的属性: {torch.cuda.get_device_properties(device=None)}")
print(f"是否支持CUDA: {torch.cuda.is_available()}")
print(f"CUDA版本号:{torch.version.cuda}")
print(f"cudnn版本：{torch.backends.cudnn.version()}")



gc.collect()
torch.cuda.empty_cache()