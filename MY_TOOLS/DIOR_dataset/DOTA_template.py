import os

class DOTA_COCOTmp:
    def __int__(self, basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'labelTxt')
        self.imagepath = os.path.join(basepath, 'images')

    def add_ann(self, img_infos):
        ann_name = 0



















