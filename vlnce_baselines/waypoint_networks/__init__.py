

from .resnetUnet import ResNetUNet
from .img_segmentor_model import ImgSegmentor

def get_img_segmentor_from_options(n_object_classes,img_segm_loss_scale):
    return ImgSegmentor(segmentation_model=ResNetUNet(n_channel_in=3, n_class_out=n_object_classes),
                        loss_scale=img_segm_loss_scale)

'''
Model ResNetUnet taken from:
https://github.com/usuyama/pytorch-unet
'''