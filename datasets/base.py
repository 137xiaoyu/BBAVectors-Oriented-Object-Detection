import torch.utils.data as data
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
from . import data_augment

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 500
        self.image_distort =  data_augment.PhotometricDistort()

    def load_img_ids(self):
        """
        Definition: generate self.img_ids
        Usage: index the image properties (e.g. image name) for training, testing and evaluation
        Format: self.img_ids = [list]
        Return: self.img_ids
        """
        return None

    def load_image(self, index):
        """
        Definition: read images online
        Input: index, the index of the image in self.img_ids
        Return: image with H x W x 3 format
        """
        return None

    def load_annoFolder(self, img_id):
        """
        Return: the path of annotation
        Note: You may not need this function
        """
        return None

    def load_annotation(self, index):
        """
        Return: dictionary of {'pts': float np array of [bl, tl, tr, br], 
                                'cat': int np array of class_index}
        Explaination:
                bl: bottom left point of the bounding box, format [x, y]
                tl: top left point of the bounding box, format [x, y]
                tr: top right point of the bounding box, format [x, y]
                br: bottom right point of the bounding box, format [x, y]
                class_index: the category index in self.category
                    example: self.category = ['ship]
                             class_index of ship = 0
        """
        return None

    def dec_evaluation(self, result_path):
        return None

    def data_transform(self, image, annotation):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
        image, gt_pts, crop_center = random_flip(image, annotation['pts'], crop_center)
        if crop_center is None:
            crop_center = np.asarray([float(image.shape[1])/2, float(image.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        if annotation['pts'].shape[0]:
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)

        out_annotations = {}
        size_thresh = 3
        out_rects = []
        out_cat = []
        pts_new = []
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any():
                pt_new = pt_old.copy()
                pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                if iou>0.6:
                    rect = cv2.minAreaRect(pt_new/self.down_ratio)
                    if rect[1][0]>size_thresh and rect[1][1]>size_thresh:
                        out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                        out_cat.append(cat)
                        pts_new.append(pt_new/self.down_ratio)
            else:
                rect = cv2.minAreaRect(pt_old/self.down_ratio)
                if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                    continue
                out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                out_cat.append(cat)
                pts_new.append(pt_old/self.down_ratio)
        out_annotations['pts'] = np.asarray(pts_new)
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, out_annotations

    def __len__(self):
        return len(self.img_ids)

    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1


    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
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
    

    # def get_pts_pos(self, pts):
    #     argsorted_pts_x = np.argsort(pts[:, 0], axis=0)
    #     argsorted_pts_y = np.argsort(pts[:, 1], axis=0)
    #     sorted_pts = np.zeros(pts.shape, dtype=np.int32)
    #     for i, pt in enumerate(sorted_pts):
    #         x_pos = np.argwhere(argsorted_pts_x == i)
    #         y_pos = np.argwhere(argsorted_pts_y == i)
    #         sorted_pts[i] = [x_pos, y_pos]
    #     pts_pos = []
    #     for i, pt in enumerate(pts):
    #         if sorted_pts[i, 0] > 1 and sorted_pts[i, 1] <= 1:
    #             pos = 0
    #         elif sorted_pts[i, 0] <= 1 and sorted_pts[i, 1] <= 1:
    #             pos = 1
    #         elif sorted_pts[i, 0] <= 1 and sorted_pts[i, 1] > 1:
    #             pos = 2
    #         elif sorted_pts[i, 0] > 1 and sorted_pts[i, 1] > 1:
    #             pos = 3
    #         pts_pos.append(pos)

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 0)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 1 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] > pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 1
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 0
    #         elif 3 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] < pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 3
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 0

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 1)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 0 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] < pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 0
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 1
    #         elif 2 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] < pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 2
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 1

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 2)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 3 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] < pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 3
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 2
    #         elif 1 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] > pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 1
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 2

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 3)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 2 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] > pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 2
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 3
    #         elif 0 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] > pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 0
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 3

    #     if 0 not in pts_pos or 1 not in pts_pos or 2 not in pts_pos or 3 not in pts_pos:
    #         raise ValueError("pts_pos error")

    #     return pts_pos


    # def get_pts_pos(self, pts):
    #     cen_pt = np.mean(pts, axis=0)
    #     pts_pos = []
    #     for i, pt in enumerate(pts):
    #         if pt[0] - cen_pt[0] >= 0 and pt[1] - cen_pt[1] < 0:
    #             pos = 0
    #         elif pt[0] - cen_pt[0] < 0 and pt[1] - cen_pt[1] <= 0:
    #             pos = 1
    #         elif pt[0] - cen_pt[0] <= 0 and pt[1] - cen_pt[1] > 0:
    #             pos = 2
    #         elif pt[0] - cen_pt[0] > 0 and pt[1] - cen_pt[1] >= 0:
    #             pos = 3
    #         pts_pos.append(pos)

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 0)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 1 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] > pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 1
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 0
    #         elif 3 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] < pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 3
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 0

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 1)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 0 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] < pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 0
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 1
    #         elif 2 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] < pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 2
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 1

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 2)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 3 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] < pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 3
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 2
    #         elif 1 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] > pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 1
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 1
    #                 pts_pos[repeated_pos_ind[1]] = 2

    #     repeated_pos_ind = np.argwhere(np.array(pts_pos) == 3)[:, 0]
    #     if repeated_pos_ind.shape[0] > 1:
    #         if 2 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][0] > pts[repeated_pos_ind[1]][0]:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 2
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 2
    #                 pts_pos[repeated_pos_ind[1]] = 3
    #         elif 0 not in pts_pos:
    #             if pts[repeated_pos_ind[0]][1] > pts[repeated_pos_ind[1]][1]:
    #                 pts_pos[repeated_pos_ind[0]] = 3
    #                 pts_pos[repeated_pos_ind[1]] = 0
    #             else:
    #                 pts_pos[repeated_pos_ind[0]] = 0
    #                 pts_pos[repeated_pos_ind[1]] = 3

    #     if 0 not in pts_pos or 1 not in pts_pos or 2 not in pts_pos or 3 not in pts_pos:
    #         raise ValueError("pts_pos error")

    #     return pts_pos


    # def sort_pts(self, pts, ann_pts_pos):
    #     pts_pos = self.get_pts_pos(pts)
        
    #     pt1_ind = pts_pos.index(ann_pts_pos[0])
    #     pt2_ind = pts_pos.index(ann_pts_pos[1])
    #     pt3_ind = pts_pos.index(ann_pts_pos[2])
    #     pt4_ind = pts_pos.index(ann_pts_pos[3])
        
    #     pt1_new = pts[pt1_ind]
    #     pt2_new = pts[pt2_ind]
    #     pt3_new = pts[pt3_ind]
    #     pt4_new = pts[pt4_ind]
        
    #     pts_new = np.asarray([pt1_new,pt2_new,pt3_new,pt4_new],np.float32)
    #     return pts_new
    
    
    # def get_direction(self, pts, ann_pts):
    #     pts_pos = self.get_pts_pos(pts)
    #     ann_pts_pos = self.get_pts_pos(ann_pts)
        
    #     pt1_ind = pts_pos.index(ann_pts_pos[0])
    #     pt2_ind = pts_pos.index(ann_pts_pos[1])
        
    #     if (pt1_ind == 0 and pt2_ind == 1) or (pt1_ind == 1 and pt2_ind == 0):
    #         direction = 0
    #     elif (pt1_ind == 1 and pt2_ind == 2) or (pt1_ind == 2 and pt2_ind == 1):
    #         direction = 1
    #     elif (pt1_ind == 2 and pt2_ind == 3) or (pt1_ind == 3 and pt2_ind == 2):
    #         direction = 2
    #     elif (pt1_ind == 3 and pt2_ind == 0) or (pt1_ind == 0 and pt2_ind == 3):
    #         direction = 3
    #     else:
    #         raise ValueError("direction error")
        
    #     # ann_pts_pos = self.get_pts_pos(ann_pts)
        
    #     return direction
    
    
    # def get_direction_vec_RBB(self, ann_pts, ct):
    #     bl = ann_pts[0, :]
    #     tl = ann_pts[1, :]
    #     tr = ann_pts[2, :]
    #     br = ann_pts[3, :]

    #     tt = (np.asarray(tl, np.float32)+np.asarray(tr, np.float32))/2
    #     rr = (np.asarray(tr, np.float32)+np.asarray(br, np.float32))/2
    #     bb = (np.asarray(bl, np.float32)+np.asarray(br, np.float32))/2
    #     ll = (np.asarray(tl, np.float32)+np.asarray(bl, np.float32))/2

    #     direction_vec = ll - ct

    #     return direction_vec
    
    
    # def get_direction_vec_HBB(self, ann_pts, ct, w, h):
    #     bl = ann_pts[0, :]
    #     tl = ann_pts[1, :]
    #     tr = ann_pts[2, :]
    #     br = ann_pts[3, :]

    #     tt = (np.asarray(tl, np.float32)+np.asarray(tr, np.float32))/2
    #     rr = (np.asarray(tr, np.float32)+np.asarray(br, np.float32))/2
    #     bb = (np.asarray(bl, np.float32)+np.asarray(br, np.float32))/2
    #     ll = (np.asarray(tl, np.float32)+np.asarray(bl, np.float32))/2

    #     direction_vec = ll - ct
        
    #     if np.fabs(direction_vec[0]) > np.fabs(direction_vec[1]):
    #         if direction_vec[0] > 0:
    #             direction_vec[0] = w/2
    #             direction_vec[1] = 0
    #         else:
    #             direction_vec[0] = -w/2
    #             direction_vec[1] = 0
    #     else:
    #         if direction_vec[1] > 0:
    #             direction_vec[0] = 0
    #             direction_vec[1] = h/2
    #         else:
    #             direction_vec[0] = 0
    #             direction_vec[1] = -h/2

    #     return direction_vec


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        
        # 10 + 8
        wh = np.zeros((self.max_objs, 18), dtype=np.float32)
        
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # copy_image1 = cv2.resize(image, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)
            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            
            # generate ground_truth of BBA vectors
            # BBA vectors: pts from minAreaRect, ct from minAreaRect
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
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            
            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            
            # generate ground_truth of directional BBA vectors
            # directional BBA vectors: pts from ann, ct from minAreaRect
            pts_4 = annotation['pts'][k, :]
            
            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2
            
            # do not reorder directional BBA vectors
            # if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
            #     tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            
            # rotational channel
            wh[k, 10:12] = tt - ct
            wh[k, 12:14] = rr - ct
            wh[k, 14:16] = bb - ct
            wh[k, 16:18] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (cen_x, cen_y), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            #####################################################################################
            # horizontal channel
            # wh from ann (better performance than the one from minAreaRect)
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1
        # ###################################### view Images #####################################
        # # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
        # # copy_image = cv2.addWeighted(np.uint8(copy_image), 0.4, hm_show, 0.8, 0)
        #     if jaccard_score>0.95:
        #         print(theta, jaccard_score, cls_theta[k, 0])
        #         cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
        #         cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
        #         key = cv2.waitKey(0)&0xFF
        #         if key==ord('q'):
        #             cv2.destroyAllWindows()
        #             exit()
        # #########################################################################################

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               }
        return ret

    def __getitem__(self, index):
        image = self.load_image(index)
        image_h, image_w, c = image.shape
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}

        elif self.phase == 'train':
            annotation = self.load_annotation(index)
            image, annotation = self.data_transform(image, annotation)
            data_dict = self.generate_ground_truth(image, annotation)
            return data_dict


