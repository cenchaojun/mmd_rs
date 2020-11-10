import json
import os
import os.path as osp
from copy import deepcopy
import numpy as np
results = dict()

def cal_mean(d):
    vs = np.array(list(d.values()))
    return float(np.mean(vs))


def get_eval_res(root):
    if not osp.isdir(root):
        return
    print(root)
    for f in os.listdir(root):
        fp = osp.join(root, f)
        if osp.isdir(fp):
            get_eval_res(fp)
            continue
        if f == 'dota_eval_results.json':
            with open(fp, 'r') as file:
                r = json.load(file)
                assert 'map' in r.keys()
                assert 'precisions' in r.keys()
                assert 'recalls' in r.keys()
                ps = r.pop('precisions')
                rs = r.pop('recalls')
                map = r.pop('map')
                r_pre = dict(mean=cal_mean(ps))
                print(cal_mean(ps), map)
                r_pre.update(ps)
                r_rec = dict(mean=cal_mean(rs))
                r_rec.update(rs)
            results[osp.split(root)[1] + '_P'] = r_pre
            results[osp.split(root)[1] + '_R'] = r_rec


def merge_eval_res(rs):
    m_rs = dict(
        name=[]
    )
    for model, r in rs.items():
        m_rs['name'].append(model)
        for k, v in r.items():
            if k not in m_rs:
                m_rs[k] = []
            m_rs[k].append(v)
    return m_rs

# with open('../reuslts/retinanet_hbb_tv/eval_reuslts.txt', 'r') as f:
#     r = eval(f.read())
get_eval_res('../results')
print(results)
merged_results = merge_eval_res(results)
print(merged_results)

n_model = len(merged_results['name'])
for i, l in merged_results.items():
    assert len(l) == n_model

import pandas as pd
df = pd.DataFrame(merged_results,
                  index=range(0, n_model),
                  columns=merged_results.keys())

writer = pd.ExcelWriter('./eval_results.xlsx')#创建数据存放路径
df.to_excel(writer)
writer.save()#文件保存
writer.close()#文件关闭