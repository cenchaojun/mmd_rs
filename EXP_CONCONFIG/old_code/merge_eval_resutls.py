import json
import os
import os.path as osp
results = dict()

def get_eval_res(root, white_list=None):
    if not osp.isdir(root):
        return
    print(root)
    for f in os.listdir(root):
        fp = osp.join(root, f)
        if osp.isdir(fp):
            get_eval_res(fp, white_list)
            continue
        if f == 'eval_results.txt':
            with open(fp, 'r') as file:
                r = eval(file.read())
            results[osp.split(root)[1]] = r

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