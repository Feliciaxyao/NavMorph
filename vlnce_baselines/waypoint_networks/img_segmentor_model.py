import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgSegmentor(nn.Module):
    
    def __init__(self, segmentation_model, loss_scale):
        super(ImgSegmentor, self).__init__()
        self._segmentation_model = segmentation_model
        self.loss_scale = loss_scale
        
        self.cel_loss = nn.CrossEntropyLoss()



    def forward(self, batch):
        
        imgs = batch['images']
        B, T, _, H, W = imgs.shape

        pred_segm_raw = self._segmentation_model(imgs)

        C = pred_segm_raw.shape[1]

        # Get a prob distribution over the labels
        pred_segm_raw = pred_segm_raw.view(B,T,C,H,W)
        pred_segm = F.softmax(pred_segm_raw, dim=2)

        output = {'pred_segm_raw':pred_segm_raw,
                  'pred_segm':pred_segm}

        return output
        

    def loss_cel(self, batch, pred_outputs):
        pred_segm_raw = pred_outputs['pred_segm_raw']
        B, T, C, H, W = pred_segm_raw.shape

        gt_segm = batch['gt_segm']
        pred_segm_loss = self.cel_loss(input=pred_segm_raw.view(B*T,C,H,W), target=gt_segm.view(B*T,H,W))
        
        pred_segm_err = pred_segm_loss.clone().detach()

        output={}
        output['pred_segm_err'] = pred_segm_err
        output['pred_segm_loss'] = self.loss_scale * pred_segm_loss
        return output