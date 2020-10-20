import json
import os
import os.path as osp
results = dict()

def merge_eval_res(root):
    if not osp.isdir(root):
        return
    for f in os.listdir(root):
        fp = osp.join(root, f)
        if osp.isdir(f):
            merge_eval_res(fp)
            continue
        if f == 'eval_results.txt':
            with open(fp, 'r') as f:
                r = eval(f.read())
            results[f] = r

# with open('../reuslts/retinanet_hbb_tv/eval_reuslts.txt', 'r') as f:
#     r = eval(f.read())
merge_eval_res('../results')
print(results)