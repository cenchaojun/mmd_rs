import multiprocessing
import time
from pynvml import *
import numpy as np

# def sleep(name, sleep_time, total_rsc):
#     for i in range(sleep_time):
#         print('%s has sleep %d / %d' % (name, i, sleep_time))
#         time.sleep(1)
#     total_rsc.value += 1
#
# def show_num(name, num_list, total_rsc):
#     for num in num_list:
#         print('%s : %d' % (name, num))
#         time.sleep(1)
#     total_rsc.value += 1

def modify_pycmd(cmd, gpus=None, ctrl=None):
    """

    :param cmd: python py_cmd.py model_name -m train
    ，不包含-d的信息
    :param gpus:
    :return:
    """
    assert gpus
    for id in gpus:
        ctrl[id] = 1
    gpu_str = str(gpus).strip('[]')
    cmd = cmd.strip()
    cmd += ' -d ' + gpu_str
    os.system(cmd)

    for id in gpus:
        ctrl[id] = 0

def get_gpu_infos(deviceCount):
    infos = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        infos.append(dict(total=info.total / 1024 ** 3,
                          free=info.free / 1024 ** 3,
                          used=info.used / 1024 ** 3))
    return infos

def get_available_gpu_ids(deviceCount, max_used=1):
    infos = get_gpu_infos(deviceCount)
    av_ids = []
    for id, info in enumerate(infos):
        if info['used'] < max_used:
            av_ids.append(id)
    return av_ids


if __name__ == "__main__":
    MAX_TASK = 2

    nvmlInit()  # 初始化
    print("Driver: ", nvmlSystemGetDriverVersion())  # 显示驱动信息
    # >>> Driver: 384.xxx

    # 查看设备
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", nvmlDeviceGetName(handle))

    # 已经使用的GPU列表(已经分配的)
    used_gpu = multiprocessing.Array("i", np.zeros(deviceCount, dtype=np.int).tolist())
    # running_task =  multiprocessing.Array("i", np.zeros(deviceCount, dtype=np.int).tolist())

    # 在这里设置队伍列表
    tasks = [
        dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DIOR_libra_faster_rcnn_full -m test',ctrl=used_gpu), used_gpu=1),
        dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DIOR_pafpn_full -m test',ctrl=used_gpu), used_gpu=1),
    ]
    task_id = 0
    p_list = []

    # 开始运行
    while(True):
        if task_id == len(tasks):
            break
        task = tasks[task_id]
        task_used = task['used_gpu']

        # 可以使用的 和 已经使用的集合 的 并集
        available_gpu_ids = get_available_gpu_ids(deviceCount)
        used_gpu_ids = np.arange(deviceCount)[np.array(list(used_gpu), dtype=np.bool)].tolist()
        available_gpu_ids = list(set(available_gpu_ids) - set(used_gpu_ids))

        # 如果没有足够GPU可以使用，则继续检查
        if len(available_gpu_ids) < task_used:
            print('Wait')
            time.sleep(1)
            continue
        if np.sum(np.array(list(used_gpu))) >= 2:
            print('Wait')
            time.sleep(1)
            continue

        # 设置GPU数量
        task['kwargs']['gpus'] = available_gpu_ids[0:task_used]
        print("available_gpu_ids: %s, used: %s" %
              (available_gpu_ids, task['kwargs']['gpus']))

        # 启动进程
        p = multiprocessing.Process(target=task['fun'], kwargs=task['kwargs'])
        p_list.append(p)
        p_list[-1].start()
        # 等待一秒，让进程设置一下used_gpu
        time.sleep(1)


        task_id += 1
    for p in p_list:
        p.join()

    print('Done')
    nvmlShutdown()


