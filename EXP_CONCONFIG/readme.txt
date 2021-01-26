model_DIOR_full_config: DIOR数据集的config，包含模型名称，模型work_dir，模型结果和评估结果等，被py_cmd调用
model_DOTA_hbb_tv_config:    DOTA的水平框config
model_DOTA_obb_tv_config:    DOTA的旋转框config
multi_test.py:          多任务测试。多进程、多GPU的调度系统，可以设置最大任务数量等，自动监测GPU使用情况并分配。
py_cmd.py:              执行train_dota.py和test_dota.py的python脚本。
                        便于实现全自动化训练、全自动化测试、模型和实验结果整理等。
MARK：标记添加的代码。代码中使用MRAK的变量，用来标记更改或者调试代码。