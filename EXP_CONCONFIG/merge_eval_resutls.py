import json
import os
import os.path as osp
import pandas as pd
from collections import OrderedDict
results = dict()

# from EXP_CONCONFIG.model_DOTA_hbb_tv_config import cfgs, show_dict
# from EXP_CONCONFIG.model_DOTA_obb_tv_config import obb_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_config import DIOR_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_ms_test_config import DIOR_ms_test_cfgs
from EXP_CONCONFIG.CONFIGS.model_NWPU_VHR_10_config import NV10_cfgs
from EXP_CONCONFIG.model_DIOR_full_voc_test_config import DIOR_voc_cfgs
# cfgs.update(obb_cfgs)
# cfgs.update(DIOR_cfgs)
cfgs = DIOR_voc_cfgs
# cfgs.update(DIOR_ms_test_cfgs)
# cfgs.update(NV10_cfgs)

eval_results = dict()
for model_name, cfg in cfgs.items():
    work_dir = cfg['work_dir']
    # 模型评估结果存在
    if os.path.exists(cfg['eval_results']):
        with open(cfg['eval_results'], 'r') as f:
            s = f.read()
            print(model_name, s)
            results = eval(s)
            if type(results) == OrderedDict:
                reuslts = dict(results)
            if not reuslts:
                continue
            if 'bbox_mAP_copypaste' in results.keys():
                results.pop('bbox_mAP_copypaste')
            if 'results_per_category' in results.keys():
                cat_ap = results.pop('results_per_category')
                for (cat, ap) in cat_ap:
                    results[cat] = ap

            for k, v in results.items():
                results[k] = float('%.3f' % float(v))

        eval_results[model_name] = results

df = pd.DataFrame(eval_results)
df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)  # 转置
writer = pd.ExcelWriter('./DIOR_voc_epoch12_eval_results.xlsx')#创建数据存放路径
df2.to_excel(writer)
writer.save()#文件保存
writer.close()#文件关闭