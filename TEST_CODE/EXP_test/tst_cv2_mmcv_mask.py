import pycocotools.mask as maskUtils
import cv2
import numpy as np
def mmd_poly2mask(mask_anns, img_h, img_w):
    masks = []
    for mask_ann in mask_anns:
        if isinstance(mask_ann, list):
            mask_ann = np.reshape(mask_ann, (1, -1)).tolist()
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        masks.append(masks)
    return masks

def cv2_poly2mask(mask_ann, img_h, img_w):
    masks = []
    for i, poly in enumerate(mask_ann):
        mask = np.zeros((img_h, img_w))
        cv2.fillPoly(mask, [np.array(poly)], color=1)
        masks.append(mask)
    return masks

def generate_point(W, H):
    wl = int((W-1)/4)
    wh = int(3*(W-1)/4)
    hl = int((H-1)/4)
    hh = int(3*(H-1)/4)

    return [np.random.randint(wl,wh), np.random.randint(hl,hh)]

H = 300
W = 300
poly_list = []
for i in range(100000):
    ps = []
    for j in range(4):
        ps.append(generate_point(H, W))
    poly_list.append(ps)
# poly_list = np.array(poly_list)
cv2_poly2mask(poly_list, H, W)
mmd_poly2mask(poly_list, H, W)