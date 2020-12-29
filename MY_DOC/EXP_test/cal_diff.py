import cv2
import numpy as np



def generate_point(W, H):
    wl = int((W-1)/4)
    wh = int(3*(W-1)/4)
    hl = int((H-1)/4)
    hh = int(3*(H-1)/4)

    return [np.random.randint(wl,wh), np.random.randint(hl,hh)]

def generate_hull(W, H, n_rand):
    img = np.zeros([W, H], dtype=np.uint8)
    points = np.array([generate_point(W, H) for i in range(n_rand)])
    hull = cv2.convexHull(points)
    hull = np.reshape(hull, [-1, 2])
    # print(hull)
    cv2.fillPoly(img, [hull], 255)
    return img, hull
    # cv2.imshow('DDD', img)
    # cv2.waitKey(10000)

W = 800
H = 800

size_diff = dict()
for i in range(100):
    org_img, points = generate_hull(W, H, 5)
    for r in [0.75, 0.5, 1 / 4, 1 / 8, 1 / 16]:
        resize_img = cv2.resize(org_img, (int(W * r), int(W * r)))
        resize_img[resize_img > 0] = 10

        hull = (points * r).astype(np.int)
        rehull_img = np.zeros([int(W * r), int(W * r)], dtype=np.uint8)
        cv2.fillPoly(rehull_img, [hull], 10)


        diff_img = resize_img - rehull_img # np.abs(rehull_img - resize_img)
        resize_gt_rehull = np.sum(diff_img>0)
        resize_lt_rehull = np.sum(diff_img<0)
        print(resize_gt_rehull, resize_lt_rehull)

        diff_img[diff_img>0] = 1
        diff_img[diff_img<=0] = 0

        diff_ratio = np.sum(diff_img) / (W * H * r * r)
        print(diff_ratio)
        if r not in size_diff.keys():
            size_diff[r] = []
        size_diff[r].append(diff_ratio)

        diff_img[diff_img>0] = 255
        # cv2.imshow('DDD', cv2.resize(diff_img, (W, H),
        #                              interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(1000)

    # resize_recover_img = cv2.resize(resize_img, (W, H))
for r, diffs in size_diff.items():
    print(r, np.mean(diffs), np.std(diffs))