import torch.nn.functional as F
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes):
        self.K = K
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, pr_decs):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat)
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        wh = self._tranpose_and_gather_feat(wh, inds)
        
        # 10 + 8
        wh = wh.view(batch, self.K, 18)
        # add
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)
        mask = (cls_theta>0.8).float().view(batch, self.K, 1)

        # quadrant 1 to quadrants 4: tt ll bb rr
        # for each quadrant: (start:end]
        
        # End of BBA vectors
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)
        
        # End of directional BBA vectors
        # tt_x = xs+wh[..., 10:11]
        # tt_y = ys+wh[..., 11:12]
        # rr_x = xs+wh[..., 12:13]
        # rr_y = ys+wh[..., 13:14]
        # bb_x = xs+wh[..., 14:15]
        # bb_y = ys+wh[..., 15:16]
        # ll_x = xs+wh[..., 16:17]
        # ll_y = ys+wh[..., 17:18]
        
        # get all BBA vectors and main direction vector
        tt = 100*torch.cat((tt_x - xs, tt_y - ys), dim=2)
        rr = 100*torch.cat((rr_x - xs, rr_y - ys), dim=2)
        bb = 100*torch.cat((bb_x - xs, bb_y - ys), dim=2)
        ll = 100*torch.cat((ll_x - xs, ll_y - ys), dim=2)
        direction_vec = 100*wh[..., 16:18]
        
        # calculate cos and direction (0 to 3: tt rr bb ll)
        norm_tt = torch.linalg.norm(tt, dim=2, keepdim=True)
        norm_rr = torch.linalg.norm(rr, dim=2, keepdim=True)
        norm_bb = torch.linalg.norm(bb, dim=2, keepdim=True)
        norm_ll = torch.linalg.norm(ll, dim=2, keepdim=True)
        norm_direction_vec = torch.linalg.norm(direction_vec, dim=2, keepdim=True)
        
        cos_tt = torch.sum(tt.mul(direction_vec), dim=2, keepdim=True)/norm_tt/norm_direction_vec
        cos_rr = torch.sum(rr.mul(direction_vec), dim=2, keepdim=True)/norm_rr/norm_direction_vec
        cos_bb = torch.sum(bb.mul(direction_vec), dim=2, keepdim=True)/norm_bb/norm_direction_vec
        cos_ll = torch.sum(ll.mul(direction_vec), dim=2, keepdim=True)/norm_ll/norm_direction_vec
        
        cos_all = torch.cat((cos_tt,cos_rr,cos_bb,cos_ll))
        direction = torch.argmax(cos_all, dim=0, keepdim=True)
        
        # #
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y,
                                scores,
                                clses,
                                direction],
                               dim=2)

        index = (scores>self.conf_thresh).squeeze(0).squeeze(1)
        detections = detections[:,index,:]
        return detections.data.cpu().numpy()