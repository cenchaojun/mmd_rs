import pycocotools.mask as maskUtils
import cv2
import numpy as np
def resize(mask_anns, img_h, img_w, ratios):
    multi_masks = []
    for r in ratios:
        new_masks = []
        for mask in mask_anns:
            r_h = int(img_h * r)
            r_w = int(img_w * r)
            new_masks.append(cv2.resize(mask, (r_h, r_w)))
        multi_masks.append(new_masks)

    return multi_masks

def repoly(poly_anns, img_h, img_w, ratios):
    multi_masks = []

    for r in ratios:
        new_masks = []

        masks = []
        for i, poly in enumerate(poly_anns):
            scale_poly = np.array(poly).reshape(-1, 2)
            scale_poly = (scale_poly * r).astype(np.int32)
            r_h = int(img_h * r)
            r_w = int(img_w * r)
            mask = np.zeros((r_h, r_w))
            cv2.fillPoly(mask, [scale_poly], color=1)
            masks.append(mask)
        multi_masks.append(masks)

    return multi_masks


def generate_hull(W, H, n_rand):
    img = np.zeros([W, H], dtype=np.uint8)
    points = np.array([generate_point(W, H) for i in range(n_rand)])
    hull = cv2.convexHull(points)
    hull = np.reshape(hull, [-1, 2])
    # print(hull)
    cv2.fillPoly(img, [hull], 255)
    return img, hull

def generate_point(W, H):
    wl = int((W-1)/4)
    wh = int(3*(W-1)/4)
    hl = int((H-1)/4)
    hh = int(3*(H-1)/4)

    return [np.random.randint(wl,wh), np.random.randint(hl,hh)]

H = 600
W = 600
masks = []
polys = []
ratios = [0.75, 0.5, 1 / 4, 1 / 8, 1 / 16]

for i in range(10000):
    mask, poly = generate_hull(W, H, 5)
    masks.append(mask)
    polys.append(poly)
ml1 = resize(masks, H, W, ratios)
ml2 = repoly(polys, H, W, ratios)

a = 0
