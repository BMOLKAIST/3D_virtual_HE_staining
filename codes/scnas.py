import abc
import torch
import torch.nn as nn
from math import ceil
import torch.nn.functional as F

from collections import OrderedDict


class UpReLUConvBN(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, momentum=0.5, affine=True
    ):
        super(UpReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            ReluConvBn(
                C_in,
                C_out,
                kernel_size,
                stride,
                padding,
                momentum=momentum,
                affine=affine,
            )
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, None, 2, "bilinear", align_corners=True)
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, momentum=0.5, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        # self.bn = nn.BatchNorm3d(C_out, momentum=momentum, affine=affine)
        self.bn = nn.InstanceNorm2d(C_out, momentum=momentum, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ReluConvBn(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation=1,
        momentum=0.5,
        affine=False,
    ):
        super(ReluConvBn, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.InstanceNorm2d(C_out, momentum=momentum, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SEBlock(nn.Module):
    def __init__(self, in_ch, r, stride):
        super().__init__()
        self.pre_x = None
        if stride > 1:
            self.pre_x = FactorizedReduce(in_ch, in_ch, affine=False)
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, max(1, in_ch // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_ch // r), in_ch),
            nn.Sigmoid(),
        )
        self.in_ch = in_ch

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        if self.pre_x:
            x = self.pre_x(x)
        x = x.mul(se_weight)
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:2]), -1).mean(2)


class ReductionCell(nn.Module):
    def __init__(self, C):
        super(ReductionCell, self).__init__()

        self.edge_s0_0 = FactorizedReduce(C, C)
        self.edge_s1_0 = ReluConvBn(C, C, 3, 2, 2, 2)

        self.edge_s0_1 = FactorizedReduce(C, C)
        self.edge_0_1 = nn.AvgPool2d(3, 1, 1)

        self.edge_0_2 = Identity()
        self.edge_1_2 = ReluConvBn(C, C, 3, 1, 1)

        self.edge_0_3 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_2_3 = nn.AvgPool2d(3, 1, 1)

    def forward(self, s0, s1):
        node_0 = self.edge_s0_0(s0) + self.edge_s1_0(s1)
        node_1 = self.edge_s0_1(s0) + self.edge_0_1(node_0)
        node_2 = self.edge_0_2(node_0) + self.edge_1_2(node_1)
        node_3 = self.edge_0_3(node_0) + self.edge_2_3(node_2)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class EncoderNormalCell(nn.Module):
    def __init__(self, C):
        super(EncoderNormalCell, self).__init__()

        self.edge_s0_0 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_s1_0 = ReluConvBn(C, C, 3, 1, 2, 2)

        self.edge_s0_1 = nn.AvgPool2d(3, 1, 1)
        self.edge_0_1 = Identity()

        self.edge_s0_2 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_0_2 = Identity()

        self.edge_s0_3 = Identity()

    def forward(self, s0, s1):
        node_0 = self.edge_s0_0(s0) + self.edge_s1_0(s1)
        node_1 = self.edge_s0_1(s0) + self.edge_0_1(node_0)
        node_2 = self.edge_s0_2(s0) + self.edge_0_2(node_0)
        node_3 = self.edge_s0_3(s0)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class ExpansionCell(nn.Module):
    def __init__(self, C):
        super(ExpansionCell, self).__init__()

        self.edge_s0_0 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_s1_0 = ReluConvBn(C, C, 3, 1, 3, 3)

        self.edge_s0_1 = Identity()
        self.edge_0_1 = Identity()

        self.edge_s1_2 = ReluConvBn(C, C, 3, 1, 2, 2)
        self.edge_1_2 = Identity()

        self.edge_s0_3 = Identity()
        self.edge_s1_3 = nn.AvgPool2d(3, 1, 1)

    def forward(self, s0, s1):
        node_0 = self.edge_s0_0(s0) + self.edge_s1_0(s1)
        node_1 = self.edge_s0_1(s0) + self.edge_0_1(node_0)
        node_2 = self.edge_s1_2(s1) + self.edge_1_2(node_1)
        node_3 = self.edge_s0_3(s0) + self.edge_s1_3(s1)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class DecoderNormalCell(nn.Module):
    def __init__(self, C):
        super(DecoderNormalCell, self).__init__()

        self.edge_s1_0 = ReluConvBn(C, C, 3, 1, 2, 2)

        self.edge_s1_1 = Identity()
        self.edge_0_1 = Identity()

        self.edge_s1_2 = ReluConvBn(C, C, 3, 1, 2, 2)
        self.edge_1_2 = Identity()

        self.edge_s0_3 = Identity()
        self.edge_0_3 = nn.AvgPool2d(3, 1, 1)

    def forward(self, s0, s1):
        node_0 = self.edge_s1_0(s1)
        node_1 = self.edge_s1_1(s1) + self.edge_0_1(node_0)
        node_2 = self.edge_s1_2(s1) + self.edge_1_2(node_1)
        node_3 = self.edge_s0_3(s0) + self.edge_0_3(node_0)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class Cell(nn.Module):
    def __init__(self, multiplier, C_prev_prev, C_prev, C, prev_resized):
        super(Cell, self).__init__()
        self.multiplier = multiplier
        self.C = C

        self.preprocess0, self.preprocess1 = self.preprocess(
            C_prev_prev, C_prev, C, prev_resized
        )
        self._ops = self.ops()
        self._seblock = SEBlock(C * self.multiplier, 6, stride=1)

    @abc.abstractmethod
    def ops(self):
        pass

    @abc.abstractmethod
    def preprocess(self, C_prev_prev, C_prev, C, prev_resized):
        pass

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        s = self._ops(s0, s1)
        o = self._seblock(s)
        return o


class CellEnc(Cell):
    def __init__(self, multiplier, C_prev_prev, C_prev, C, reduction, prev_resized):
        self.reduction = reduction
        super(CellEnc, self).__init__(multiplier, C_prev_prev, C_prev, C, prev_resized)

    def ops(self):
        if self.reduction:
            _ops = ReductionCell(self.C)
        else:
            _ops = EncoderNormalCell(self.C)
        return _ops

    def preprocess(self, C_prev_prev, C_prev, C, prev_resized):
        if prev_resized:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReluConvBn(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0, affine=False)

        return self.preprocess0, self.preprocess1


class CellDec(Cell):
    def __init__(self, multiplier, C_prev_prev, C_prev, C, increase, prev_resized):
        self.increase = increase
        super(CellDec, self).__init__(multiplier, C_prev_prev, C_prev, C, prev_resized)

    def ops(self):
        if self.increase:
            return ExpansionCell(self.C)
        else:
            return DecoderNormalCell(self.C)

    def preprocess(self, C_prev_prev, C_prev, C, prev_resized):
        if prev_resized:
            self.preprocess0 = UpReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0, affine=False)
        elif self.increase:
            self.preprocess0 = UpReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = UpReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        else:
            self.preprocess0 = ReluConvBn(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0, affine=False)
        return self.preprocess0, self.preprocess1


class ScNas(nn.Module):
    def __init__(
        self, input_channel, num_feature, num_layers, num_multiplier, num_classes
    ):
        super(ScNas, self).__init__()
        self.ch = input_channel
        self.C = num_feature
        self.num_classes = num_classes
        self._layers = num_layers
        self._step = 3
        self._multiplier = num_multiplier

        stem_feature = self.C * self._step
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            self.ch,
                            stem_feature,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("act", nn.LeakyReLU(inplace=False)),
                    ("se", SEBlock(stem_feature, 6, stride=1)),
                    (
                        "norm",
                        nn.InstanceNorm2d(stem_feature, momentum=0.5, affine=False),
                    ),
                    # ('pool', nn.MaxPool2d([2, 2], [2, 2]))
                ]
            )
        )

        resize = [1, 0] * self._layers

        C_prev_prev, C_prev, C_curr = stem_feature, stem_feature, self.C
        self.enc_cells = nn.ModuleList()
        reduction_prev = False
        for i in range(self._layers):
            if resize[i] == 1:
                C_curr *= self._multiplier
                reduction = True
            else:
                reduction = False

            cell = CellEnc(
                self._step, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )

            reduction_prev = reduction
            self.enc_cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._step * C_curr

        self.bridge_cell = CellEnc(
            self._step, C_prev_prev, C_prev, C_curr, False, False
        )

        self.dec_cells = nn.ModuleList()
        increase_prev = False
        for i in range(self._layers):
            if resize[len(resize) - 1 - i] == 1:
                C_curr //= self._multiplier
                increase = True
            else:
                increase = False

            cell = CellDec(
                self._step, C_prev_prev, C_prev, C_curr, increase, increase_prev
            )
            C_prev_prev, C_prev = C_prev, self._step * C_curr

            increase_prev = increase
            self.dec_cells.append(cell)

        self.out = nn.Conv2d(
            C_prev, self.num_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        enc_feats = []
        s0 = s1 = self.stem(x)
        enc_feats.append(s1)

        for i, cell in enumerate(self.enc_cells):
            # print('enc{}'.format(i))
            s0, s1 = s1, cell(s0, s1)
            enc_feats.append(s1)

        s0, s1 = enc_feats.pop(), self.bridge_cell(s0, s1)

        for i, cell in enumerate(self.dec_cells):
            # print('dec{}'.format(i))
            s0, s1 = s1, cell(s0, s1)

            low = enc_feats.pop()
            s1 += low

        # s1 = nn.functional.interpolate(s1, None, (2, 2), 'bilinear', align_corners=True)
        out = self.out(s1)

        return out


class ScNasDropOutRegressor(ScNas):
    def __init__(self, num_feature, num_layer, multiplier, num_class, drop_rate=0.01):
        super(ScNasDropOutRegressor, self).__init__(
            num_feature, num_layer, multiplier, num_class
        )
        self.drop_out = nn.Dropout(p=drop_rate)

    def forward(self, x):
        enc_feats = []
        # s0 = s1 = F.dropout(self.stem(x), p=self.dropout_rate, training=True)
        s0 = s1 = self.stem(x)
        enc_feats.append(s1)

        for i, cell in enumerate(self.enc_cells):
            # s0, s1 = s1, F.dropout(cell(s0, s1), p=self.dropout_rate)
            s0, s1 = s1, self.drop_out(cell(s0, s1))
            enc_feats.append(s1)

        enc_feats.pop()

        for i, cell in enumerate(self.dec_cells):
            # s0, s1 = s1, F.dropout(cell(s0, s1), p=self.dropout_rate, training=self.training)
            s0, s1 = s1, self.drop_out(cell(s0, s1))

            low = enc_feats.pop()
            s1 += low

        # s1 = nn.functional.interpolate(s1, None, (2, 2), 'bilinear', align_corners=True)
        out = self.out(s1)
        return out


class NbsScNas(ScNas):
    def decoder(self, enc_feats, s0, s1):
        for i, cell in enumerate(self.dec_cells):
            s0, s1 = s1, cell(s0, s1)

            low = enc_feats.pop()
            s1 += low

        out = self.out(s1)
        return out

    def forward(self, x, alpha):
        enc_feats = []
        s0 = s1 = self.stem(x)
        enc_feats.append(s1)

        for i, cell in enumerate(self.enc_cells):
            # print('enc{}'.format(i))
            s0, s1 = s1, cell(s0, s1)
            enc_feats.append(s1)

        s0, s1 = enc_feats.pop(), self.bridge_cell(s0, s1)

        if isinstance(alpha, int):
            res_ = torch.zeros([alpha, x.size(0), self.num_classes, *x.shape[2:]]).to(
                x.device
            )
            for i in range(alpha):
                w = torch.rand(*s1.shape[:2], 1, 1).to(x.device)
                s1 = s1 * w
                res = self.decoder(enc_feats.copy(), s0, s1)
                res_[i] += res
            return res_
        else:
            w = torch.exp(-F.interpolate(alpha[:, None], s1.size(1)))[
                :, 0, :, None, None
            ]
            s1 = s1 * w
            return self.decoder(enc_feats, s0, s1)


if __name__ == "__main__":
    from collections import namedtuple

    # Arg = namedtuple('Arg', ['num_feature', 'num_class', 'layers', 'multiplier'])
    # args = Arg(24, 5, 6, 3)
    model = ScNas(
        input_channel=2, num_feature=4, num_layer=4, multiplier=2, num_class=3
    )
    model = nn.DataParallel(model).cuda()
    # print(model)
    #
    input = torch.randn([1, 2, 128, 128]).cuda()
    output = model(input)
    print(output.min(), output.max())
    print(output.shape)
