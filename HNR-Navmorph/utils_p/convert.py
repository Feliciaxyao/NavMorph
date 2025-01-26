import torch.nn as nn
import numpy as np


class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3), keepdims=True)
        cur_std = x.std((2, 3), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return new_x


def convert_encoder_to_target(net, norm, start=0, end=5, verbose=True, bottleneck=False, input_size=512, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(net.conv1.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 0:
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx, fea_size=input_size // 2)
            idx += 1
        else:
            down_sample = 2 ** (1 + i)

            for j, block in enumerate(layer):
                block.bn1 = convert_norm(block.bn1, norm, block.bn1.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                block.bn2 = convert_norm(block.bn2, norm, block.bn2.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                if bottleneck:
                    block.bn3 = convert_norm(block.bn3, norm, block.bn3.num_features, idx, fea_size=input_size // down_sample)
                    idx += 1
                if block.downsample is not None:
                    block.downsample[1] = convert_norm(block.downsample[1], norm, block.downsample[1].num_features, idx, fea_size=input_size // down_sample)
                    idx += 1
    return net


def convert_decoder_to_target(net, norm, start=0, end=5, verbose=True, input_size=512, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [net[0], net[1], net[2], net[3], net[4]]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 4:
            net[4] = convert_norm(layer, norm, layer.num_features, idx, input_size)
            idx += 1
        else:
            down_sample = 2 ** (4 - i)
            layer.bn = convert_norm(layer.bn, norm, layer.bn.num_features, idx, input_size // down_sample)
            idx += 1
    return net

