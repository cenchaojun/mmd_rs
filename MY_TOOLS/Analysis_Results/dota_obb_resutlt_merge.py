import os

def parse_dota_res(result_folder):
    res_files = os.listdir(result_folder)

    img_infos = dict()
    for res_file in res_files:
        category = os.path.splitext(res_file)[0]
        print(category)
        with open(result_folder + '/' + res_file, 'r') as f:
            while True:
                det_info = f.readline().strip()
                if not det_info:
                    break
                det_info = det_info.split()
                img_file = det_info[0] + '.png'
                score = float(det_info[1])
                x1, y1, x2, y2 ,\
                    x3, y3, x4, y4 = float(det_info[2]), \
                                 float(det_info[3]), \
                                 float(det_info[4]), \
                                 float(det_info[5]), \
                                 float(det_info[6]), \
                                 float(det_info[7]), \
                                 float(det_info[8]), \
                                 float(det_info[9])
                if img_file not in img_infos.keys():
                    img_infos[img_file] = dict(
                        bboxes=[],
                        scores=[],
                        cats=[]
                    )
                img_infos[img_file]['bboxes'].append([x1, y1, x2, y2,
                                                      x3, y3, x4, y4])
                img_infos[img_file]['scores'].append(score)
                img_infos[img_file]['cats'].append(category)
    print(len(img_infos))
    return img_infos

gt_ann_file = '/home/huangziyue/data/dota/val/val_coco.json'
dt_ann_folder = '/home/huangziyue/mmdet_results/retinanet_obb_tv/Task1_results_nms'

