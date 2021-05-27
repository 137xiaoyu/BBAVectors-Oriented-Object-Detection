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
import polyiou
from functools import partial
import cv2

def reorder_pts(tt, rr, bb, ll):
    pts = np.asarray([tt,rr,bb,ll],np.float32)
    l_ind = np.argmin(pts[:,0])
    r_ind = np.argmax(pts[:,0])
    t_ind = np.argmin(pts[:,1])
    b_ind = np.argmax(pts[:,1])
    tt_new = pts[t_ind,:]
    rr_new = pts[r_ind,:]
    bb_new = pts[b_ind,:]
    ll_new = pts[l_ind,:]
    return tt_new,rr_new,bb_new,ll_new

def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                
                # generate ground truth of direction
                
                # result direction: det_directional_BBAvec compared with det_BBAvec
                # gt direction: ann_directional_BBAvec compared with ann_BBAvec
                
                # ann_directional_BBAvec
                # directional BBA vectors: pts from ann, ct from minAreaRect
                ann_pts = np.asarray(object_struct['bbox'], np.float32).reshape((-1, 2))
                pt_0 = ann_pts[0,:]
                pt_1 = ann_pts[1,:]            
                direction_vec = (np.asarray(pt_0,np.float32)+np.asarray(pt_1,np.float32))/2
                
                # ann_BBAvec
                # BBA vectors: pts from minAreaRect, ct from minAreaRect
                rect = cv2.minAreaRect(ann_pts)
                (cen_x, cen_y), (bbox_w, bbox_h), theta = rect
                pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2
            
                bl = pts_4[0,:]
                tl = pts_4[1,:]
                tr = pts_4[2,:]
                br = pts_4[3,:]

                tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
                rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
                bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
                ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2
                
                # reorder BBA vectors
                if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                    tt,rr,bb,ll = reorder_pts(tt,rr,bb,ll)

                # compute all BBA vectors and main direction vector
                tt = 100*(tt - np.asarray([cen_x, cen_y], np.float32))
                rr = 100*(rr - np.asarray([cen_x, cen_y], np.float32))
                bb = 100*(bb - np.asarray([cen_x, cen_y], np.float32))
                ll = 100*(ll - np.asarray([cen_x, cen_y], np.float32))
                direction_vec = 100*(direction_vec - np.asarray([cen_x, cen_y], np.float32))
                
                # compute cos and direction (0 to 3: tt rr bb ll)
                norm_tt = np.linalg.norm(tt)
                norm_rr = np.linalg.norm(rr)
                norm_bb = np.linalg.norm(bb)
                norm_ll = np.linalg.norm(ll)
                norm_direction_vec = np.linalg.norm(direction_vec)
                
                cos_tt = np.sum(tt*direction_vec)/norm_tt/norm_direction_vec
                cos_rr = np.sum(rr*direction_vec)/norm_rr/norm_direction_vec
                cos_bb = np.sum(bb*direction_vec)/norm_bb/norm_direction_vec
                cos_ll = np.sum(ll*direction_vec)/norm_ll/norm_direction_vec
                
                cos_all = np.asarray([cos_tt,cos_rr,cos_bb,cos_ll], np.float32)
                direction = np.argmax(cos_all)

                # another way (not recommended, but has higher accuracy)
                # center_pt_x = np.mean(np.array([float(splitlines[0]), float(splitlines[2]),
                #                                 float(splitlines[4]), float(splitlines[6])]))
                # center_pt_y = np.mean(np.array([float(splitlines[1]), float(splitlines[3]),
                #                                 float(splitlines[5]), float(splitlines[7])]))
                # direction_vec_x = (float(splitlines[0]) + float(splitlines[2]))/2 - center_pt_x
                # direction_vec_y = (float(splitlines[1]) + float(splitlines[3]))/2 - center_pt_y
                
                # if direction_vec_x >= 0 and direction_vec_y < 0:
                #     direction = 0   # tt
                # elif direction_vec_x < 0 and direction_vec_y <= 0:
                #     direction = 3   # ll
                # elif direction_vec_x <= 0 and direction_vec_y > 0:
                #     direction = 2   # bb
                # elif direction_vec_x > 0 and direction_vec_y >= 0:
                #     direction = 1   # rr
                # else:
                #     raise ValueError('direction error')
                
                object_struct['direction'] = int(direction)
                
                objects.append(object_struct)
            else:
                break
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
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        directions = np.array([x['direction'] for x in R]).astype(np.int32)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'direction': directions,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    directions = np.array([int(float(x[2])) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[3:]] for x in splitlines])
    
    # if no detected target
    if BB.shape[0] == 0:
        print('no ' + classname)
        return 0, 0, 0, 0

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    directions = [directions[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    t_direction = np.zeros(nd)
    count = 0
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        direction = directions[d].astype(int)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        direction_gt = R['direction'].astype(int)
        
        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    # compare direction
                    if direction == direction_gt[jmax]:
                        t_direction[d] = 1
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
            count = count + 1

    # compute the accuracy of direction
    t_direction = np.sum(t_direction)
    if t_direction == 0:
        acc_direction = float(0)
    else:
        acc_direction = t_direction / float(np.sum(tp))

    # compute precision recall
    print('check fp:', fp)
    print('check tp', tp)
    print('npos num:', npos)
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, acc_direction

def main():

    # ##TODO: wrap the code in the main
    # detpath = r'/home/dingjian/Documents/Research/experiments/light_head_faster_rotbox_best_point/Task1_results_0.1_nms_epoch18/results/Task1_{:s}.txt'
    # annopath = r'/home/dingjian/code/DOTA/DOTA/media/OrientlabelTxt-utf-8/{:s}.txt'# change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'/home/dingjian/code/DOTA/DOTA/media/testset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

    detpath = r'D:\\137\\workspace\\python_projects\\BBAVectors-Oriented-Object-Detection\\result_dota\\Task1_{:s}.txt'
    annopath = r'D:\\137\\dataset\\MunichDatasetVehicleDetection-2015-old\\DOTA_TrainVal\\labelTxt\\{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'D:\\137\\dataset\\MunichDatasetVehicleDetection-2015-old\\DOTA_TrainVal\\valset.txt'

    # For DOTA-v1.5
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    # For DOTA-v1.0
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    
    classnames = ['pkw', 'pkw_trail', 'truck', 'truck_trail', 'van_trail', 'cam', 'bus']
    
    classaps = []
    class_acc = []
    map = 0
    macc = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap, acc_direction = voc_eval(detpath,
                                                annopath,
                                                imagesetfile,
                                                classname,
                                                ovthresh=0.5,
                                                use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)
        
        # count acc_direction
        macc = macc + acc_direction
        print('accuracy_direction: ', acc_direction)
        class_acc.append(acc_direction)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
       # plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    
    # show acc_direction
    macc = macc/len(classnames)
    print('mean acc_direction: ', macc)
    class_acc = 100*np.array(class_acc)
    print('class acc_direction: ', class_acc)
    
if __name__ == '__main__':
    main()
