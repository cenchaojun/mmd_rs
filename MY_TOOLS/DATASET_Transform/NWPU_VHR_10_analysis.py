import cv2
import os
data_root = '../../data/NWPU_VHR_10'
ana_root = data_root + '/ground truth'
cat_ids = []

for gt_f in os.listdir(ana_root):
    with open(ana_root + '/' + gt_f) as f:
        anns = list(f.readlines())
        anns = [ann.strip('\n') for ann in anns]
        anns = [ann for ann in anns if len(ann) > 0]

        # print(anns)
        anns = [eval(ann) for ann in anns]
        for ann in anns:
            (x1, y1) = ann[0]
            (x2, y2) = ann[1]
            cat_id = ann[2]
            cat_ids.append(cat_id)
            if x2 - x1 <= 0 or y2 - y1<=0:
                print(x1, y1, x2, y2, cat_id)

print(cat_ids)
print(set(cat_ids))
