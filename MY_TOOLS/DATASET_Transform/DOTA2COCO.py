
from DOTA_devkit.DOTA2COCO import DOTA2COCOTrain, DOTA2COCOTest
import os
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

if __name__ == '__main__':
    root = '/home/huangziyue/data/dota_1_1024_824'

    DOTA2COCOTrain(root + '/train',
                  root + '/train/train_coco_ann.json',
                   wordname_15, difficult='-1')
    DOTA2COCOTrain(root + '/val',
                  root + '/val/val_coco_ann.json',
                   wordname_15, difficult='-1')
    DOTA2COCOTrain(root + '/train_val',
                  root + '/train_val/train_val_coco_ann.json',
                   wordname_15, difficult='-1')
    DOTA2COCOTest(root + '/test',
                  root + '/test/test_coco_ann.json',
                  wordname_15)

    # DOTA2COCOTrain(r'../../data/dota/train',
    #                r'../../data/dota/train/train_coco.json',
    #                wordname_15)
    # DOTA2COCOTrain(r'../../data/dota/test',
    #                r'../../data/dota/train/train_coco.json',
    #                wordname_15)
    # DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024',
    #               r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024/DOTA_test1024.json',
    #               wordname_15)
