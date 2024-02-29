import torch
import torch.nn as nn


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    #NOTE: Res_CBAM_block就是cbrcb，然后空间注意一下，然后再通道注意一下，再残差连接
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # TODO: 消融实验1 ca/sa
        out = self.ca(out) * out    # 通道注意力
        out = self.sa(out) * out    # 空间注意力

        out += residual             # 残差连接

        out = self.relu(out)
        return out

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bn1   = nn.BatchNorm2d(4*self.channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4*self.channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=2, bias=False)
        self.bn0   = nn.BatchNorm2d(self.channels)

    def forward(self, x):
        skip = x
        skip = self.conv0(skip)
        skip = self.bn0(skip)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu2(x + skip)

class DNANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False):
        super(DNANet, self).__init__()
        self.relu = nn.ReLU(inplace = True)

        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        # NOTE: 上采样层的双线性插值法：factor=2表示扩展为原来的两倍长宽，在中间插入线性增加的做法
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        # NOTE: 为0.5时，直接取左上角的数字
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # TODO: 改进实验SPD-conv
        self.down0_0 = SPDConv(nb_filter[0])
        self.down0_1 = SPDConv(nb_filter[0])
        self.down0_2 = SPDConv(nb_filter[0])
        self.down0_3 = SPDConv(nb_filter[0])
        self.down1_0 = SPDConv(nb_filter[1])
        self.down1_1 = SPDConv(nb_filter[1])
        self.down1_2 = SPDConv(nb_filter[1])
        self.down2_0 = SPDConv(nb_filter[2])
        self.down2_1 = SPDConv(nb_filter[2])
        self.down3_0 = SPDConv(nb_filter[3])

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        #NOTE: _make_layer有四个参数：block, input_c, output_c, num_block。其中nb_filter是每一层的通道数，num_blocks是连续拼接的block数量，用iooo连接法
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        """将num_blocks个block顺序连接。其中每个block的设定都是input_channels和output_channels"""
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        # NOTE: 非常好的写法，先用列表装起来，然后给他拆了装Sequential里面
        return nn.Sequential(*layers)

    # def forward(self, input):
    #     x0_0 = self.conv0_0(input)
    #     x1_0 = self.conv1_0(self.pool(x0_0))
    #     x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

    #     x2_0 = self.conv2_0(self.pool(x1_0))
    #     x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
    #     x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

    #     x3_0 = self.conv3_0(self.pool(x2_0))
    #     x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
    #     x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
    #     x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

    #     x4_0 = self.conv4_0(self.pool(x3_0))
    #     x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
    #     x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
    #     x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
    #     x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

    #     Final_x0_4 = self.conv0_4_final(
    #         torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
    #                    self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

    #     if self.deep_supervision: # 如果ds，就是从4个
    #         output1 = self.final1(x0_1)
    #         output2 = self.final2(x0_2)
    #         output3 = self.final3(x0_3)
    #         output4 = self.final4(Final_x0_4)
    #         return [output1, output2, output3, output4]
    #     else:
    #         output = self.final(Final_x0_4)
    #         return output

    def forward(self, input):
        # TODO: 改进实验SPCConv
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.down0_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.down1_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down0_1(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.down2_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down1_1(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down0_2(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.down3_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down2_1(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down1_2(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down0_3(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

        if self.deep_supervision: # 如果ds，就是从4个
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)
            return output


