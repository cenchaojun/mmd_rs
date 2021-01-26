# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import xml.etree.ElementTree as ET
import os
#import cPickle
import numpy as np
import matplotlib.pyplot as plt

def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ')  for x in lines]
        splitlines = splitlines[2:len(splitlines)]
        for splitline in splitlines:
            object_struct = {}
            object_struct['name'] = splitline[8]
            if (len(splitline) == 9):
                object_struct['difficult'] = 0
            elif (len(splitline) == 10):
                object_struct['difficult'] = int(splitline[9])
            # object_struct['difficult'] = 0#
            ############################################################
            # object_struct['bbox'] = [int(float(splitline[0])),
            #                              int(float(splitline[1])),
            #                              int(float(splitline[4])),
            #                              int(float(splitline[5]))]
            #
            # w = int(float(splitline[4])) - int(float(splitline[0]))
            # h = int(float(splitline[5])) - int(float(splitline[1]))

            x1, y1, x2, y2, x3, y3, x4, y4 = int(float(splitline[0])),\
                                         int(float(splitline[1])),\
                                         int(float(splitline[2])),\
                                         int(float(splitline[3])),\
                                         int(float(splitline[4])),\
                                         int(float(splitline[5])),\
                                         int(float(splitline[6])),\
                                         int(float(splitline[7]))




            xmin, ymin, xmax, ymax = min([x1, x2, x3, x4]), min([y1,y2,y3,y4]), \
                                     max([x1, x2, x3, x4]), max([y1,y2,y3,y4])
            object_struct['bbox'] = [xmin, ymin, xmax, ymax]
            w = xmax - xmin
            h = ymax - ymin
            #############################################################

            object_struct['area'] = w * h
            #print('area:', object_struct['area'])
            # if object_struct['area'] < (15 * 15):
            #     #print('area:', object_struct['area'])
            #     object_struct['difficult'] = 1
            objects.append(object_struct)
    return objects
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        #if i % 100 == 0:
         #   print ('Reading annotation for {:d}/{:d}'.format(
          #      i + 1, len(imagenames)) )
        # save
        #print ('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'w') as f:
         #   cPickle.dump(recs, f)
    #else:
        # load
        #with open(cachefile, 'r') as f:
         #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if not lines:
        return 0, 0, 0
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            ## if there exist 2
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
                   # print('filename:', image_ids[d])
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def evaluate(detpath, annopath, imagesetfile, eval_result_path):

    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    classaps = []
    map = 0
    recalls = dict()
    precisions = dict()
    full_data = dict()
    for classname in classnames:
        ###################
        # classname = 'ship'
        ###################
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False)
        recalls[classname] = float(np.max(rec))
        precisions[classname] = float(ap)
        if isinstance(rec, np.ndarray):
            rec = rec.tolist()
            prec = prec.tolist()
            full_data[classname] = dict(rec=rec, prec=prec)
        else:
            full_data[classname] = dict(rec=0, prec=0)


        # print('rc: ', np.max(rec))
        map = map + ap
        print('ap: ', ap)
        classaps.append(ap)

        ## uncomment to plot p-r curve for each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
        # plt.show()
    map = map/len(classnames)
    result = dict(
        map=map,
        precisions=precisions,
        recalls=recalls,
        # full_data=full_data
    )
    print(result)
    import json
    with open(eval_result_path, 'wt+') as f:
        json.dump(result, f)

    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('model', help='model name')
    parser.add_argument('-d', help='devices id, 0~9')
    parser.add_argument('-c', help='control, list->list models, state->model的状态')
    parser.add_argument('-resume', help='latest -> latest or int -> epoch')

    parser.add_argument('-load_results',
                        action='store_true',
                        help=' does not inference ,just evaluate, default=True')


    parser.add_argument('-m', help='mode, train or test', default='train')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    os.chdir('../')
    args = parse_args()

    annopath = r'./data/dota/val/labelTxt/{:s}.txt'# change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'./data/dota/val/val_sets.txt'

    from EXP_CONCONFIG.model_hbb_tv_config import cfgs
    for name, cfg in cfgs.items():
        # print(cfg)
        output_path = cfg['work_dir']
        detpath = cfg['Task2_results'] + '/{:s}.txt'
        eval_result_path = cfg['work_dir'] + '/dota_eval_results.json'

        if os.path.exists(cfg['Task2_results']):
            evaluate(detpath, annopath, imagesetfile, eval_result_path)
            print(name + ' Done!')

        else:
            print(name + ' Pass!')
