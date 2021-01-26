import pandas as pd
import numpy as np

from commonlibs.drawing_tools.diagram import *
from commonlibs.common_tools import *

def dfs2arrays(df):
    values =df.iloc[:, 2:].values
    cols = list(df.columns)[2:]
    rows = list(df['name'])
    return [rows, cols, values]

def plot_and_save(df,
                  n_cat_file='./sheets/val_cat_nums.json',
                  img_path='./test_val.png',
                  result_path='./sheets/hbb_tv_pre.xlsx'
                  ):
    pre = dfs2arrays(df)

    cat_nums = jsonload(n_cat_file)
    total = sum(list(cat_nums.values()))
    num_ratios = []
    for name in pre[1]:
        if name in cat_nums.keys():
            num_ratios.append(float(cat_nums[name]) / total)
        else:
            num_ratios.append(0)
    print(num_ratios)

    pre[0].append('cat_nums')
    num_ratios = np.array(num_ratios).reshape(1, len(num_ratios)) * 2
    pre[2] = np.concatenate((pre[2], num_ratios), axis=0)
    print(pre[1])

    mean_model_ap = np.mean(pre[2], axis=0)
    var_model_ap = np.var(pre[2], axis=0)
    name_map = [(m, v, name)
                for name, m, v
                in zip(pre[1], mean_model_ap, var_model_ap)]
    name_map = sorted(name_map)
    map = np.array([[m, v] for m, v, n in name_map]).T

    maps = pd.DataFrame(map, index=['map', 'var'],
                        columns=[n for m,v,n in name_map])

    writer = pd.ExcelWriter(result_path)  # 创建数据存放路径
    maps.to_excel(writer, sheet_name='map')

    writer.save()  # 文件保存
    writer.close()
    for i in name_map:
        print(i)

    simple_plot_mul_compare(img_path,
                            range(len(pre[1])),
                            pre[2],
                            pre[0])


data = pd.read_excel('./sheets/.1_dota_eval_results.xlsx', sheetname=None)
pre1 = data['Precision']
data2 = pd.read_excel('./sheets/.4_dota_eval_results.xlsx', sheetname=None)
pre2 = data2['Precision']
pre = pd.concat([pre1,pre2])

writer = pd.ExcelWriter('./sheets/hbb_tv_pre.xlsx')  # 创建数据存放路径
pre.to_excel(writer, sheet_name='Precision')
writer.save()
writer.close()

plot_and_save(pre, img_path='./hbb_tv.png',
              result_path='./sheets/hbb_tv_map.xlsx')


data = pd.read_excel('./sheets/obb.xlsx', sheetname=None)
pre = data['Precision']
plot_and_save(pre, img_path='./obb.png',
              result_path='./sheets/obb_map.xlsx')



