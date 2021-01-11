
from DOTA_devkit.DOTA2COCO import DOTA2COCOTrain, DOTA2COCOTest

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

if __name__ == '__main__':

    DOTA2COCOTrain(r'../data/dota/train',
                   r'../data/dota/train/train_coco.json',
                   wordname_15)
    # DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024',
    #               r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024/DOTA_test1024.json',
    #               wordname_15)
