# import torch
# import os
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_capability(i))
#     print(torch.cuda.get_device_name(i))
#     print(torch.cuda.get_device_properties(i))
#     print('\n')
#
# s = os.system('lspci | grep -i nvidia')
# print(s)

#########################################################
from pynvml import *
nvmlInit()     #初始化
print("Driver: ", nvmlSystemGetDriverVersion())  #显示驱动信息
#>>> Driver: 384.xxx

#查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))
#>>>
#GPU 0 : b'GeForce GTX 1080 Ti'
#GPU 1 : b'GeForce GTX 1080 Ti'

#查看显存、温度、风扇、电源
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)

print("Memory Total: ",info.total / 1024**3)
print("Memory Free: ",info.free/ 1024**3)
print("Memory Used: ",info.used/ 1024**3)

print("Temperature is %d C"% nvmlDeviceGetTemperature(handle,0))
print("Fan speed is ", nvmlDeviceGetFanSpeed(handle))
print("Power ststus",nvmlDeviceGetPowerState(handle))


for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(handle)

    print("Memory Total: ", info.total / 1024 ** 3)
    print("Memory Free: ", info.free / 1024 ** 3)
    print("Memory Used: ", info.used / 1024 ** 3)

    # print("Temperature is %d C" % nvmlDeviceGetTemperature(handle, 0))
    # print("Fan speed is ", nvmlDeviceGetFanSpeed(handle))
    # print("Power ststus", nvmlDeviceGetPowerState(handle))

#最后要关闭管理工具
nvmlShutdown()