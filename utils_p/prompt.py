import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=224):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        start = time.time()
        _, _, imgH, imgW = x.size() # image size

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1)) # fft for image

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft) # amp: 振幅谱，表示频率大小；pha：相位谱，相位偏移
        amp_src = torch.fft.fftshift(amp_src) # 将振幅谱的低频分量移动到图像的中心

        # F.pad: 填充self.data_prompt, 将其居中于输入图像尺寸
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous() # self.data:预定义的提示，通过填充操作使其与输入图像的尺寸匹配。填充操作确保只处理频谱中的特定部分（低频部分）

        amp_src_ = amp_src * prompt # 振幅谱与提示相乘，以专注于特定的低频
        amp_src_ = torch.fft.ifftshift(amp_src_) # prompt调制后，将频率成分移回其原始排列方式

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size] # 提取低频成分

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW) #重构图像
        end = time.time()
        T = end - start
        #print(T)
        
        return src_in_trg, amp_low_


    
    def enhance(self, x, retrieve_p):
        
        _, _, imgH, imgW = x.size() # image size

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1)) # fft for image

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft) # amp: 振幅谱，表示频率大小；pha：相位谱，相位偏移
        amp_src = torch.fft.fftshift(amp_src) # 将振幅谱的低频分量移动到图像的中心

        # F.pad: 填充self.data_prompt, 将其居中于输入图像尺寸
        prompt = F.pad(retrieve_p.cuda(), [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous() # self.data:预定义的提示，通过填充操作使其与输入图像的尺寸匹配。填充操作确保只处理频谱中的特定部分（低频部分）

        amp_src_ = amp_src * prompt # 振幅谱与提示相乘，以专注于特定的低频
        amp_src_ = torch.fft.ifftshift(amp_src_) # prompt调制后，将频率成分移回其原始排列方式

        # amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size] # 提取低频成分

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW) #重构图像
        #end = time.time()
        #T = end - start
        #print(T)
        
        return src_in_trg #, amp_low_