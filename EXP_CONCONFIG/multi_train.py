import multiprocessing
import time
from pynvml import *
import numpy as np

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
    # 自动化测试命令
    from EXP_CONCONFIG.model_DIOR_full_config import DIOR_cfgs
    from EXP_CONCONFIG.model_DIOR_full_ms_test_config import DIOR_ms_test_cfgs
    cfgs = DIOR_ms_test_cfgs
    cfgs.update(DIOR_cfgs)
    cmds = []
    # 筛选出已经训练好的模型，并形成commands
    for model_name, cfg in cfgs.items():
        work_dir = cfg['work_dir']
        # 模型存在
        if os.path.exists(work_dir):
            work_files = os.listdir(work_dir)
            # 模型训练完毕
            if os.path.exists(cfg['cp_file']):
                # print(cfg['result'])
                # 已经存在结果
                if not os.path.exists(cfg['result']):
                    print('%-80s \tDO Inference' % model_name)
                    cmds.append('python py_cmd.py %s -m test' % model_name)
                else:
                    print('%-80s \tLOAD RESULTS'%model_name)
                    cmds.append('python py_cmd.py %s -m test -load_results' % model_name)
        ##############################
        # work_dir = cfg['work_dir']
        # # 模型存在
        # if os.path.exists(work_dir):
        #     work_files = os.listdir(work_dir)
        #     # 模型训练完毕
        #     if os.path.exists(cfg['cp_file']):
        #         print(cfg['result'])
        #         # 不存在result文件
        #         if not os.path.exists(cfg['result']):
        #             cmds.append('python py_cmd.py %s -m test' % model_name)

    for cmd in cmds:
        print(cmd)

    # 最大同时存在的task数目
    MAX_TASK = 4

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
    # tasks = [
    #     dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DIOR_libra_faster_rcnn_full -m test',ctrl=used_gpu), used_gpu=1),
    #     dict(fun=modify_pycmd, kwargs=dict(cmd='python py_cmd.py DIOR_pafpn_full -m test',ctrl=used_gpu), used_gpu=1),
    # ]
    tasks = [dict(fun=modify_pycmd, kwargs=dict(cmd=cmd, ctrl=used_gpu),used_gpu=1) for cmd in cmds]
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
        if np.sum(np.array(list(used_gpu))) >= MAX_TASK:
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


