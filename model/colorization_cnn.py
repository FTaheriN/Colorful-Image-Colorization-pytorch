import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BasicConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, **kwargs),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.conv(X)


class colorizationCNN(nn.Module):

    def __init__(self):
        super(colorizationCNN, self).__init__()

        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

        layer1  = [BasicConvBlock(1 , 64, kernel_size=3, stride=1, padding=1, bias=True),]
        layer1 += [BasicConvBlock(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        layer1 += [nn.BatchNorm2d(64),]

        layer2  = [BasicConvBlock(64 ,128, kernel_size=3, stride=1, padding=1, bias=True),]
        layer2 += [BasicConvBlock(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        layer2 += [nn.BatchNorm2d(128),]

        layer3  = [BasicConvBlock(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        layer3 += [BasicConvBlock(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        layer3 += [BasicConvBlock(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        layer3 += [nn.BatchNorm2d(256),]

        layer4  = [BasicConvBlock(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        layer4 += [BasicConvBlock(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        layer4 += [BasicConvBlock(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        layer4 += [nn.BatchNorm2d(512),]

        layer5  = [BasicConvBlock(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        layer5 += [BasicConvBlock(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        layer5 += [BasicConvBlock(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        layer5 += [nn.BatchNorm2d(512),]

        layer6  = [BasicConvBlock(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        layer6 += [BasicConvBlock(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        layer6 += [BasicConvBlock(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        layer6 += [nn.BatchNorm2d(512),]

        layer7  = [BasicConvBlock(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        layer7 += [BasicConvBlock(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        layer7 += [BasicConvBlock(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        layer7 += [nn.BatchNorm2d(512),]

        layer8  = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        layer8 += [nn.ReLU(True),]
        layer8 += [BasicConvBlock(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        layer8 += [BasicConvBlock(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]

        layer8 +=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)
        self.layer6 = nn.Sequential(*layer6)
        self.layer7 = nn.Sequential(*layer7)
        self.layer8 = nn.Sequential(*layer8)

        # self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def normalize_l(self, in_l):
        return (in_l-self.l_cent)/self.l_norm

    def unnormalize_l(self, in_l):
        return in_l*self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab/self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab*self.ab_norm

    def forward(self, x):
        conv1_2 = self.layer1(self.normalize_l(x))
        conv2_2 = self.layer2(conv1_2)
        conv3_3 = self.layer3(conv2_2)
        conv4_3 = self.layer4(conv3_3)
        conv5_3 = self.layer5(conv4_3)
        conv6_3 = self.layer6(conv5_3)
        conv7_3 = self.layer7(conv6_3)
        conv8_3 = self.layer8(conv7_3)
        out_reg = self.model_out(conv8_3)

        return self.unnormalize_ab(self.upsample4(out_reg))