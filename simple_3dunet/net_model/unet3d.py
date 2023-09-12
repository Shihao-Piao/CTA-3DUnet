from .unet3d_parts import *


# 定义Encoder
class EncoderUNet3D(nn.Module):
    def __init__(self, channel_list=[1, 8, 16, 32, 64, 64]):
        super(EncoderUNet3D, self).__init__()
        self.input = DoubleConv3D(channel_list[0], channel_list[1])
        self.down1 = DownBlock3D(channel_list[1], channel_list[2])
        self.down2 = DownBlock3D(channel_list[2], channel_list[3])
        self.down3 = DownBlock3D(channel_list[3], channel_list[4])
        self.down4 = DownBlock3D(channel_list[4], channel_list[5])

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


# 定义Decoder
class DecoderUNet3D(nn.Module):
    def __init__(self, channel_list=[64, 64, 32, 16, 8, 1]):
        super(DecoderUNet3D, self).__init__()
        self.up4 = UpBlock3D(channel_list[0], channel_list[1], shortcut_ch=channel_list[1])
        self.up3 = UpBlock3D(channel_list[1], channel_list[2], shortcut_ch=channel_list[2])
        self.up2 = UpBlock3D(channel_list[2], channel_list[3], shortcut_ch=channel_list[3])
        self.up1 = UpBlock3D(channel_list[3], channel_list[4], shortcut_ch=channel_list[4])
        self.output = nn.Conv3d(channel_list[4], channel_list[5], kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.output(x)
        return torch.sigmoid(x)


# 定义3DUNet网络结构
class UNet3D(nn.Module):
    def __init__(self, input_channels, n_classes, f_channel):
        super(UNet3D, self).__init__()
        self.input = DoubleConv3D(input_channels, f_channel)
        self.down1 = DownBlock3D(f_channel, f_channel * 2)
        self.down2 = DownBlock3D(f_channel * 2, f_channel * 4)
        self.down3 = DownBlock3D(f_channel * 4, f_channel * 8)
        # self.down4 = DownBlock3D(f_channel * 8, f_channel * 8)
        # self.up4 = UpBlock3D(f_channel * 8, f_channel * 8, shortcut_ch=f_channel * 8)
        self.up3 = UpBlock3D(f_channel * 8, f_channel * 4, shortcut_ch=f_channel * 4)
        self.up2 = UpBlock3D(f_channel * 4, f_channel * 2, shortcut_ch=f_channel * 2)
        self.up1 = UpBlock3D(f_channel * 2, f_channel, shortcut_ch=f_channel)
        self.output = nn.Conv3d(f_channel, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up4(x5, x4)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.output(x)
        return torch.sigmoid(x)
