import torch.nn as nn


class Sample(nn.Module):
    def __init__(self, s_shape, t_shape):
        super(Sample, self).__init__()
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        self.channelSample = nn.Conv2d(s_C, t_C, kernel_size=1)
        self.spatailSample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, x):
        x = self.channelSample(x)
        x = self.spatailSample(x)
        return x

# class UpSample(nn.Module):
#     def __init__(self, s_shape, t_shape):
#         super(UpSample, self).__init__()
#         s_N, s_C, s_H, s_W = s_shape
#         t_N, t_C, t_H, t_W = t_shape
#         self.sample = nn.ConvTranspose2d(s_C, t_C, kernel_size=int(t_H/s_H), stride=int(t_H/s_H))
#
#     def forward(self, x):
#         x = self.sample(x)
#         return x
#
#
# class DownSample(nn.Module):
#     def __init__(self, s_shape, t_shape):
#         super(DownSample, self).__init__()
#         s_N, s_C, s_H, s_W = s_shape
#         t_N, t_C, t_H, t_W = t_shape
#         self.sample = nn.Conv2d(s_C, t_C, kernel_size=1, stride=int(s_H/t_H))
#
#     def forward(self, x):
#         x = self.sample(x)
#         return x