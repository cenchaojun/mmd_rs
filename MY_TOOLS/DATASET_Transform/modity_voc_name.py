import os

img_root = '../../data/DIOR_VOC/JPEGImages'
target_format= '000000003611'
for img_file in os.listdir(img_root):
    img_path = img_root + '/' + img_file
    img_id = int(os.path.splitext(img_file)[0])
    img_ext = os.path.splitext(img_file)[1]

    target_img_name = str('%d'%img_id).zfill(12) + img_ext
    target_path =  img_root + '/' + target_img_name
    os.rename(img_path, target_path)
    print('%s --> %s' % (img_path, target_path))