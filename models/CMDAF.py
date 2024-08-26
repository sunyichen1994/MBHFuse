import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gelu = nn.GELU().cuda()

    sub_vi_ir = vi_feature - ir_feature
    # Define your convolutional layer to replace power_average_pool
    conv1 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv2 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv3 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    sub_vi_ir_conv1 = conv1(sub_vi_ir)
    sub_vi_ir_conv2 = conv2(sub_vi_ir_conv1)
    sub_vi_ir_conv3 = conv3(sub_vi_ir_conv2)
    vi_ir_div = sub_vi_ir * gelu(sub_vi_ir_conv3)

    sub_ir_vi = ir_feature - vi_feature
    # Define another convolutional layer
    conv4 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv5 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv6 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    sub_ir_vi_conv4 = conv4(sub_ir_vi)
    sub_ir_vi_conv5 = conv5(sub_ir_vi_conv4)
    sub_ir_vi_conv6 = conv6(sub_ir_vi_conv5)
    ir_vi_div = sub_ir_vi * gelu(sub_ir_vi_conv6)


    vi_ir_div1 = CoordAtt(inp=ir_feature.shape[1], oup=ir_feature.shape[1]).to(ir_feature.device)(ir_vi_div)
    ir_vi_div1 = CoordAtt(inp=vi_feature.shape[1], oup=vi_feature.shape[1]).to(vi_feature.device)(vi_ir_div)

    vi_feature += ir_vi_div1
    ir_feature += vi_ir_div1

    return vi_feature, ir_feature

def power_average_pool(x, power):
    # 幂平均池化函数
    batch_size, channels, height, width = x.size()
    pooled = nn.functional.avg_pool2d(x, (height, width))  # 全局平均池化
    pooled = torch.pow(pooled, power)  # 取幂
    pooled = torch.mean(pooled, dim=(2, 3), keepdim=True)  # 平均值
    return pooled

def CMDAF1(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gelu = nn.GELU().cuda()

    sub_vi_ir = vi_feature - ir_feature

    conv1 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv2 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv3 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    sub_vi_ir_conv1 = conv1(sub_vi_ir)
    sub_vi_ir_conv2 = conv2(sub_vi_ir_conv1)
    sub_vi_ir_conv3 = conv3(sub_vi_ir_conv2)
    vi_ir_div = sub_vi_ir * gelu(sub_vi_ir_conv3)

    sub_ir_vi = ir_feature - vi_feature

    conv4 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv5 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv6 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    sub_ir_vi_conv4 = conv4(sub_ir_vi)
    sub_ir_vi_conv5 = conv5(sub_ir_vi_conv4)
    sub_ir_vi_conv6 = conv6(sub_ir_vi_conv5)
    ir_vi_div = sub_ir_vi * gelu(sub_ir_vi_conv6)


    vi_channel = vi_feature.mean(3).mean(2)  # shape: (batch_size, channels)
    ir_channel = ir_feature.mean(3).mean(2)  # shape: (batch_size, channels)
    vi_channel_weight = torch.unsqueeze(sigmoid(vi_channel), dim=2).unsqueeze(dim=3)
    ir_channel_weight = torch.unsqueeze(sigmoid(ir_channel), dim=2).unsqueeze(dim=3)
    vi_feature = vi_feature * vi_channel_weight
    ir_feature = ir_feature * ir_channel_weight


    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature

def CMDAF2(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gelu = nn.GELU().cuda()

    sub_vi_ir = vi_feature - ir_feature

    conv1 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv2 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv3 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    sub_vi_ir_conv1 = conv1(sub_vi_ir)
    sub_vi_ir_conv2 = conv2(sub_vi_ir_conv1)
    sub_vi_ir_conv3 = conv3(sub_vi_ir_conv2)
    vi_ir_div = sub_vi_ir * gelu(sub_vi_ir_conv3)

    sub_ir_vi = ir_feature - vi_feature

    conv4 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv5 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv6 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    sub_ir_vi_conv4 = conv4(sub_ir_vi)
    sub_ir_vi_conv5 = conv5(sub_ir_vi_conv4)
    sub_ir_vi_conv6 = conv6(sub_ir_vi_conv5)
    ir_vi_div = sub_ir_vi * gelu(sub_ir_vi_conv6)


    batch_size, channels, height, width = vi_feature.size()

    vi_pos_weight = torch.sigmoid(torch.arange(height * width, device=vi_feature.device).view(1, 1, height, width))
    ir_pos_weight = torch.sigmoid(torch.arange(height * width, device=ir_feature.device).view(1, 1, height, width))

    vi_feature = vi_feature * vi_pos_weight
    ir_feature = ir_feature * ir_pos_weight

    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature

def CMDAF3(vi_feature, ir_feature):

    sigmoid = nn.Sigmoid()
    gelu = nn.GELU().cuda()

    sub_vi_ir = vi_feature - ir_feature

    conv1 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv2 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    conv3 = nn.Conv2d(in_channels=sub_vi_ir.shape[1], out_channels=sub_vi_ir.shape[1], kernel_size=3, padding=1).cuda()
    sub_vi_ir_conv1 = conv1(sub_vi_ir)
    sub_vi_ir_conv2 = conv2(sub_vi_ir_conv1)
    sub_vi_ir_conv3 = conv3(sub_vi_ir_conv2)
    vi_ir_div = sub_vi_ir * gelu(sub_vi_ir_conv3)

    sub_ir_vi = ir_feature - vi_feature

    conv4 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv5 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    conv6 = nn.Conv2d(in_channels=sub_ir_vi.shape[1], out_channels=sub_ir_vi.shape[1], kernel_size=3, padding=1).cuda()
    sub_ir_vi_conv4 = conv4(sub_ir_vi)
    sub_ir_vi_conv5 = conv5(sub_ir_vi_conv4)
    sub_ir_vi_conv6 = conv6(sub_ir_vi_conv5)
    ir_vi_div = sub_ir_vi * gelu(sub_ir_vi_conv6)



    vi_corners = torch.stack([vi_feature[:, :, 0, 0], vi_feature[:, :, 0, -1],
                              vi_feature[:, :, -1, 0], vi_feature[:, :, -1, -1]], dim=2)
    ir_corners = torch.stack([ir_feature[:, :, 0, 0], ir_feature[:, :, 0, -1],
                              ir_feature[:, :, -1, 0], ir_feature[:, :, -1, -1]], dim=2)
    vi_corner_weight = torch.unsqueeze(sigmoid(vi_corners.mean(dim=2)), dim=2).unsqueeze(dim=3)
    ir_corner_weight = torch.unsqueeze(sigmoid(ir_corners.mean(dim=2)), dim=2).unsqueeze(dim=3)
    vi_feature = vi_feature * vi_corner_weight
    ir_feature = ir_feature * ir_corner_weight

    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature