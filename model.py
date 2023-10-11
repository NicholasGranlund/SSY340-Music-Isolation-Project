import torch
from torch import nn


class UnetModel(nn.Module):

    def __init__(self, ngf=64, nc=1):
        super(UnetModel, self).__init__()

        # initialize layers
        self.audionet_convlayer1 = self.unet_conv(nc, ngf)
        self.audionet_convlayer2 = self.unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = self.unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = self.unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = self.unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = self.unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = self.unet_conv(ngf * 8, ngf * 8)
        self.upconvlayer1 = self.unet_upconv(ngf * 8, ngf * 8)
        self.upconvlayer2 = self.unet_upconv(ngf * 16, ngf * 8)
        self.upconvlayer3 = self.unet_upconv(ngf * 16, ngf * 8)
        self.upconvlayer4 = self.unet_upconv(ngf * 16, ngf * 4)
        self.upconvlayer5 = self.unet_upconv(ngf * 8, ngf * 2)
        self.upconvlayer6 = self.unet_upconv(ngf * 4, ngf)
        self.upconvlayer7 = self.unet_upconv(ngf * 2, nc, True)

    def forward(self, x):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_upconv1feature = self.upconvlayer1(audio_conv7feature)
        audio_upconv2feature = self.upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        audio_upconv3feature = self.upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        audio_upconv4feature = self.upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        audio_upconv5feature = self.upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        audio_upconv6feature = self.upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        mask_prediction = self.upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        return mask_prediction

    @staticmethod
    def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])

    @staticmethod
    def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
        upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(output_nc)
        if not outermost:
            return nn.Sequential(*[upconv, upnorm, uprelu])
        else:
            return nn.Sequential(*[upconv, nn.Sigmoid()])

    def transfer_weights(self, weight_path, device):
        print('Loading weights for UNet')
        trained_model_state_dict = torch.load(weight_path, map_location=device)
        trained_trim = {k: v for k, v in trained_model_state_dict.items() if not k.startswith('audionet_upconvlayer')}
        self.load_state_dict(trained_trim, strict=False)

