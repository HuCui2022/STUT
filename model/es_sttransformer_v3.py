import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from graph.ntu_rgb_d import Graph as NTU_Graph
from graph.ucla import Graph as UCLA_Graph
from graph.shrec17 import Graph as SHREC_Graph
from graph.dhg14_28 import Graph as DHG_Graph
from model.utils  import conv_init,bn_init,fc_init,DropPath,comput_params,generate_G_from_H
import os
from torch.nn import Parameter
from einops import rearrange
from torchvision.ops import SqueezeExcitation

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
######  temporal model
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # n,c,t,v
        x = self.conv(x)
        x = self.bn(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)



    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)  # n,c,t,v    cat on channel dimenssion.
        out += res
        return out


'''### test MultiScale Temporal Conv
x = torch.rand(16,32,120,25).cuda()   # n, c, t, v
net = MultiScale_TemporalConv(
                 in_channels = 32,
                 out_channels = 36,  # must mulitpy with len(dilations)
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1).cuda()
out = net(x)
print(out.shape)
print(torch.isnan(out).max())
#torch.Size([16, 36, 120, 25]) , tensor(False, device='cuda:0')
'''

# classic adaptive temporal model
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # n c t v
        x = self.bn(self.conv(x))
        return x

'''#### test unit tcn
x = torch.rand(16, 32, 120, 25).cuda()
net = unit_tcn(32, 32, 5, 1).cuda()
output = net(x)
print(output.shape, torch.isnan(output).max())'''
# torch.Size([16, 32, 120, 25]) tensor(False, device='cuda:0')



class dp_spatial_att(nn.Module):
    def __init__(self,
                 dim_in, dim, num_heads,
                 attn_drop=0,
                 sp_attention = True,
                 sp_pe=True,    # spatial attention or not.
                 partition_pe = True,   # use partion encoding or not
                 soft_partion = False,   # softmax for it.
                 partion_pe_learnabel = True,   # partion encoding is learnable or not.
                 structal_enc = True,   # relative distance encoding for skeleton struction
                 att_type = 'dpgat', # gatv1, gatv2
                 sparsity = True,   # sparsity or fully connecting.
                 bias_att = True,   # attention bias
                 num_joints=25,
                 time_len = 120,
                 use_unfold = True,
                 window_size = 3,
                 window_stride = 1,   # stride == 1, make the output T' == T
                 window_dilation = 1,
                 att_bias=True,
                 att_soft = 'softmax',    # softmax or tan.
                 leakeyR = 0.1,
                 graph = None,
                 g = None,
                 ):
        super(dp_spatial_att, self).__init__()

        assert att_type in ['dpgat', 'gatv1', 'gatv2'], f"attention is {att_type}, it should be one of dpgat, gatv1, gatv2"
        self.sp_attention = sp_attention
        self.att_type = att_type
        self.partition_pe = partition_pe
        self.bias_att = bias_att
        self.structal_enc = structal_enc
        self.use_unfold = use_unfold
        self.window_size = window_size
        self.sparsity  = sparsity
        self.num_heads = num_heads
        self.att_soft = att_soft

        if self.sp_attention:
            if sp_pe:
                self.sp_pe = PositionalEncoding(dim_in, num_joints, time_len, domain = 'spatial')
            else:
                self.sp_pe = nn.Identity()

            dim_hid = dim // num_heads
            self.dim_hid = dim_hid
            assert num_heads*dim_hid == dim, 'num heads * hid dim != out dim.'
            self.q = nn.Conv2d(dim_in, dim, 1, bias=True)
            # get q shape : n, c', t, v
            if use_unfold:
                self.unfold = UnfoldTemporalWindows(window_size, window_stride, window_dilation = window_dilation)
                # stride 1, make T' == T, 这里unfold 没有改变通道， 所以还需要一个 self.v 来进行辅助，或者 outnet 来进行之后的通道变化。todo.
            # get output shape :  n,c, t', windowsize*v;
            if att_type == 'gatv1':
                self.k = self.q
            else:
                self.k = nn.Conv2d(dim_in, dim, 1, bias=True)
            self.a_left = nn.Conv2d(self.dim_hid, 1, 1)
            self.a_right = nn.Conv2d(self.dim_hid, 1, 1)
            # after unfold : n,c',t', w*v

            if partition_pe :
                if num_joints == 25:
                    self.partion_encod = HyperPartionEmbedding(g = g,
                                                               dim_in=dim_in,
                                                               dim_out=dim,
                                                               joints_num=num_joints,
                                                               softpartion=soft_partion,
                                                               learnabel = partion_pe_learnabel)
                elif num_joints == 20:
                    self.partion_encod = HyperPartionEmbedding(g = g,
                                                               dim_in=dim_in,
                                                               dim_out=dim,
                                                               joints_num=num_joints,
                                                               softpartion=soft_partion,
                                                               learnabel = partion_pe_learnabel)
                elif num_joints == 22:
                    self.partion_encod = HyperPartionEmbedding(g=g,
                                                               dim_in=dim_in,
                                                               dim_out=dim,
                                                               joints_num=num_joints,
                                                               softpartion=soft_partion,
                                                               learnabel=partion_pe_learnabel)
                # can embedding hyper edge information, shape == input x

            # set structal_enc: enc shape 25 25; params; adjx;

            # if num_joints == 25:
            #     graph = NTU_Graph()   # can get graph.A = list :3 25 x25, graph.dis_adjs->list: 3, 25 x25; graph.hops-> 25 x25, values 0 - 12; or 0-11
            # elif num_joints == 20:
            #     graph = UCLA_Graph()   #

            self.graph = graph

            if structal_enc:
                #  struc_enc shape : 25, 25, values == 0-25; can be buffer tensor;
                self.dis_matrix = torch.tensor(graph.hops).long()
                # params
                # print(self.dis_matrix.max())
                self.se_pe = nn.Parameter(torch.zeros((self.dis_matrix.max() + 1, dim_in)))  # shape： hops.max() 24 阶, but the max value maybe 12 or 11
                nn.init.trunc_normal_(self.se_pe.data)
            if bias_att:
                self.att0 = nn.Parameter(torch.stack([torch.eye(num_joints) for _ in range(num_heads)], dim=0),
                                          requires_grad=True)

                # self.outer shape : num_heads, 25,25
                self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
                nn.init.trunc_normal_(self.alpha)
                self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
                nn.init.trunc_normal_(self.beta)


            self.attn_drop = nn.Dropout(attn_drop)

            # set different params for different att_type:
            # gatv1, w shape

            self.relu = nn.LeakyReLU(leakeyR)

            self.out_net_sp = nn.Sequential(nn.Conv2d(dim_in * num_heads, dim, 1),
                                          nn.BatchNorm2d(dim),
                                          )
            if dim_in != dim:
                self.downsample_s = nn.Sequential(nn.Conv2d(dim_in, dim, 1, bias=True),
                                                nn.BatchNorm2d(dim),
                                                )
            else:
                self.downsample_s = lambda x: x

        else:
            self.out_net_sp = nn.Sequential(nn.Conv2d(dim_in * num_heads, dim, (1, 3), padding=(0,1), bias=True, stride=1),
                                          nn.BatchNorm2d(dim),
                                          )



    def forward(self, x):
        # NM,C,T,V : X SHAPE
        N,C,T,V = x.shape
        if self.sp_attention:
            # compute attention from hyper edge information

            y = self.sp_pe(x)  # spatial positional encoding, or indentity for x

            # compute attention from distance encoding.
            if self.structal_enc:
                struc_en = self.se_pe[self.dis_matrix]  # shape : 25,25, c=dim_in.
                sec = torch.einsum('nctu, uvc -> nctv', x, struc_en)  # n, c_in, t,v
                y = y + sec


            # compute attention  q, k, if use unfold, attention shape : window*v, v;
            # if not use unfold,  attention shape : v, v;
            k = self.k(y)
            k = rearrange(k, 'n (h c) t v -> n h c t v', h = self.num_heads)
            if self.use_unfold:
                uy = self.unfold(y)  # shape : n,c,t, v' ;   v' = window_size * v;  t' is the new num of clips.
                q = self.q(uy)  # shape: n c t v'  # v' 为局部相邻的多个帧的local spatial temporal, v' = windowsize * v
                q = rearrange(q, 'n (h c) t v -> n h c t v', h = self.num_heads)
            else:
                q = self.q(y)
                q = rearrange(q, 'n (h c) t v -> n h c t v', h=self.num_heads)

            # compute attention from  qk
            if self.att_type == 'dpgat':
                attention = torch.einsum('n h c t u, n h c t v -> n h u v', q, k) / (self.dim_hid * T)
                #  u = windeowsize * v
            else:
                # when att_type  == gatv1, self.k = self.q;  k == q;
                # when att_type === gatv2, self.k != self.q; k != q; but you can make it same to reduce params.
                q0 = q
                k = rearrange(k, 'n h c t v -> (n h) c t v')
                q = rearrange(q, 'n h c t v -> (n h) c t v')

                if self.att_type == 'gatv1':
                    q = self.a_left(q).squeeze(1)  # (nh) 1, t v
                    q = rearrange(q, '(n h) t v -> n h t v', h = self.num_heads)
                    k = self.a_right(k).squeeze(1)  # (nh), 1, t, v
                    k = rearrange(k, '(n h) t v -> n h t v', h = self.num_heads)

                    attention = torch.einsum('nhtu, nhtv -> nhuv',q,k)/T
                    attention = self.relu(attention)
                elif self.att_type == 'gatv2':
                    k = self.relu(k)
                    q = self.relu(q)
                    k = self.a_right(k).squeeze(1)
                    q = self.a_left(q).squeeze(1)    # nh,1,t,v -> nh, t,v;
                    k = rearrange(k, '(n h) t v -> n h t v', h = self.num_heads)
                    q = rearrange(q, '(n h) t v -> n h t v', h = self.num_heads)
                    attention = torch.einsum('nhtu, nhtv -> nhuv', q, k) / T

            if self.partition_pe:
                # use partion encoding information
                e = self.partion_encod(y)  # e,x shape : N,C_out,T,V;  e have partition information
                e = rearrange(e, 'n (h c) t v -> n h c t v', h = self.num_heads)
                if self.att_type == 'dpgat':
                    attention_e = torch.einsum('n h c t u, n h c t v -> n h u v', q, e) / (self.dim_hid * T)
                else:
                    attention_e = torch.einsum('n h c t u, n h c t v -> n h u v', q0, e) / (self.dim_hid * T)

                attention = attention + attention_e*self.beta
            # softmax and  set sparsity or fully:
            # todo : skip values set.
            if self.sparsity:
                zero_vec = -9e15 * torch.ones_like(attention[:, 0])  # n u v; only singel head; cuase const for every head.
                # print(self.graph.dis_adjs)
                # print(x.device)
                dis_graphs = torch.from_numpy(self.graph.dis_adjs).to(x.device)
                if self.use_unfold:
                    attention = torch.cat([torch.where(dis_graphs[i].repeat(self.window_size, 1) > 0, attention[:, i,:,:], zero_vec).unsqueeze(1) for i in range(self.num_heads)], dim=1)
                else:
                    attention = torch.cat([torch.where(dis_graphs[i] > 0, attention[:, i,:,:], zero_vec).unsqueeze(1) for i in range(self.num_heads)], dim=1)

                # repeat shape : v v -> w*v,v;  attention[:, i, :,:] the i-th head of the attention.
                # unsqueeze(1), add a new  head dim for every attenion.
                # attention shape : n h u v; u = windowsize * v

            # softmax :
            if self.att_soft == 'softmax':
                attention = F.softmax(attention, dim=-2)
            else:
                attention = F.tanh(attention, dim=-2)

            if self.bias_att:
                if self.use_unfold:
                    attention = attention*self.alpha + self.att0.repeat(1, self.window_size, 1)
                    # self.att0 shape : h v v -> h windowsize*v v,
                else:
                    attention = attention*self.alpha + self.att0
                    # self.att0 shape : h v v  for all sample.


            # attention shape : n h u v
            # unfold shape : n c_in, t, w*v
            # not unfold shape : n c_in, t, v
            attention = self.attn_drop(attention)   # used for hook, and get the attention scors map.

            if self.use_unfold:
                zz = torch.einsum('nctu, nhuv -> nhctv', uy, attention)
            else:
                zz = torch.einsum('nctu, nhuv -> nhctv', y, attention)

            # zz shape : n h c_in, t, v
            zz = rearrange(zz, 'n h c t v -> n (h c) t v')
            zz = self.out_net_sp(zz)   # zz shape : n c_out, t, v

            out = self.relu(self.downsample_s(x) + zz)


        else:
            zz = self.out_net_sp(x)
            out = self.relu(self.downsample_s(x) + zz)

        return out   # out shape : n,c,t,v

## dp_spatial_att test at the end line of the file.

class dp_temporal_att(nn.Module):
    # se attention or temporal attention,
    # in temporal attention, we can decrease the temporal dimenssion by temporal stride
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel_size=3,
                 t_stride=1,
                 pool_type = 'maxpool', # avgpool
                 dilations=[1, 2, 3, 4],
                 t_type = 'mul_scale',  # 'mul_scale', 'tcn', 'none';
                 temporal_att = 'se_att', # se_att, dpatt, none.
                 squeeze_factor = 1,   # 大于1, 来减少中间的计算量。
                 num_heads = 4,
                 num_frames = 120,
                 num_joints = 25,
                 att_drop = 0.0,
                 t_pe = True
                 ):
        super(dp_temporal_att, self).__init__()
        assert  dim_out % (len(dilations)+2) == 0, 'dim_out can div 4, 6, dilations chosse option in : [1,2], or [1,2,3,4]'
        self.t_stride = t_stride
        self.temporal_att = temporal_att
        if t_type == 'mul_scale':
            self.tcn = MultiScale_TemporalConv(in_channels=dim_in,
                                               out_channels=dim_out,
                                               kernel_size=kernel_size,
                                               dilations=dilations,
                                               stride=1)
        elif t_type == 'tcn':
            self.tcn = unit_tcn(in_channels=dim_in,
                                out_channels=dim_out,
                                kernel_size=kernel_size,
                                stride=1
                                )
        else:
            self.tcn = nn.Identity()


        if temporal_att == 'se_att':
            self.att = SELayer(input_channels=num_frames,
                               squeeze_factor=squeeze_factor)
        elif temporal_att == 'none':
            self.att = nn.Identity()


        elif temporal_att == 'dpatt':
            if t_pe:
                self.t_pe = PositionalEncoding(dim_in, num_joints, num_frames, domain = 'temporal')
            else:
                self.t_pe = nn.Identity()

            self.dim_hid = dim_out//num_heads
            self.num_heads = num_heads
            assert self.dim_hid * num_heads == dim_out
            self.qk = nn.Conv2d(dim_in, num_heads* self.dim_hid * 2, 1)
            self.att0 = nn.Parameter(torch.zeros(1, num_heads, num_frames, num_frames) + torch.eye(num_frames),
                                                requires_grad=True)
            self.alphat = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
            self.outnet = nn.Sequential(
                nn.Conv2d(dim_in*num_heads, dim_out, 1),
                nn.BatchNorm2d(dim_out)
            )
            self.tan = nn.Tanh()
            self.drop_att = nn.Dropout(att_drop)
            self.relu = nn.LeakyReLU(0.2)
        if t_stride == 2:
            if pool_type == 'maxpool':
                self.temporl_pool = nn.MaxPool2d((2,1), (2,1))
            elif pool_type == 'avgpool':
                self.temporl_pool = nn.AvgPool2d((2,1), (2,1))

    def forward(self, x):
        N, C, T, V = x.shape
        if self.temporal_att == 'se_att' :
            x = rearrange(x, 'n c t v -> n t c v')
            x = self.att(x)
            out = rearrange(x, 'n t c v -> n c t v')

        elif self.temporal_att == 'none':
            out = self.att(x)

        elif self.temporal_att == 'dpatt':
            # do the gpat attention
            y = self.t_pe(x)
            qk = self.qk(y)   # n 2*h*c t v
            qk = rearrange(qk, 'n (h c) t v -> n h c t v', h = 2*self.num_heads)
            q, k = torch.chunk(qk, 2, dim=1)    # q, k shape : n h c t v
            atten_t = torch.einsum('nhcsv, nhctv -> nhst', q, k)/(self.dim_hid * V)
            atten_t = self.tan(atten_t) * self.alphat + self.att0
            atten_t = self.drop_att(atten_t)
            z = torch.einsum('ncsv, nhst -> nhctv', x, atten_t)
            z = rearrange(z, 'n h c t v -> n (h c) t v')
            z = self.outnet(z)

            out = self.relu(z + x)

        out = self.tcn(out)
        if self.t_stride == 2:
            out = self.temporl_pool(out)
        return out

# test dp_temporal att :
# after SElyar

# ffmlp net
class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # n,c,t,v = x.shape
        # x = self.fc1(x.transpose(1,2)).transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x.transpose(1,2)).transpose(1,2)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# test MLP
'''x = torch.rand(16, 32, 120, 25).cuda()
net = mlp(32).cuda()
output = net(x)
print(output.shape, torch.isnan(output).max())'''
#torch.Size([16, 32, 120, 25]) tensor(False, device='cuda:0')


class SELayer(SqueezeExcitation):

    def __init__(self, input_channels, squeeze_factor=1, bias=True):
        squeeze_channels = input_channels // squeeze_factor
        super().__init__(input_channels, squeeze_channels)

        if not bias:
            self.fc1.register_parameter('bias', None)
            self.fc2.register_parameter('bias', None)
### test SELayer
'''x = torch.rand(16, 25, 120, 32).cuda()
net = SELayer(25, squeeze_factor=1).cuda()
output = net(x)
print(output.shape, torch.isnan(output).max())'''
# torch.Size([16, 25, 120, 32]) tensor(False, device='cuda:0')

### test dp_temporal att
'''x = torch.rand(16, 32, 120, 25).cuda()
net = dp_temporal_att(dim_in = 32,
                 dim_out = 32,
                 kernel_size=3,
                 t_stride=2,
                 pool_type = 'maxpool', # avgpool
                 dilations=[1, 2],
                 t_type = 'mul_scale',  # 'mul_scale', 'tcn', 'none';
                 temporal_att = 'dpatt', # se_att, dpatt, none.
                 num_heads = 4,
                 num_frames = 120,
                 att_drop = 0.0,
                 num_joints = 25,
                 t_pe = True
).cuda()
output = net(x)
print(output.shape, torch.isnan(output).max())

# torch.Size([16, 32, 60, 25]) tensor(False, device='cuda:0')'''

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.conv = nn.Conv2d(in_ft, out_ft, 1)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # x shape : n c t v
        x = self.conv(x)
        x = torch.matmul(x, G)
        return x

class HyperPartionEmbedding(nn.Module):
    ''' encode the partition can not be learned'''
    def __init__(self, g, dim_in, dim_out, joints_num=25, softpartion=False, bias = False, learnabel = True):
        super().__init__()
        # dim_in == dim_out
         # g   #[0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1]
        # g shape : 25 x 25
        self.g = g

        self.proj = HGNN_conv(dim_in, dim_out)
    def forward(self, x):
        # x shape : n, c, t, v
        g = self.g.to(x.device, non_blocking = True)
        out = self.proj(x, g)
        return out

# test HyeprPartionEmbedding
'''x = torch.rand(12,16, 120, 25).cuda()
net = HyperPartionEmbedding(partion=[0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1],
                            dim_in=16, dim_out=16, joints_num=25, learnabel=False).cuda()

out = net(x)
print(out.shape)
print(out)'''


class Structual_Encoding_re_distance():
    ''' use relational distance for structual encoding.'''
    def __init__(self, A, num_joints, dim):
        # A shape : 3, 25, 25; e, in_degree, out_degree
        assert A[0].shape[-1] == num_joints
        A = A.sum(0)  # shape: 25, 25, adjmatrix
        A[A!=0] = 1
        dis_adj = [None for _ in range(num_joints)]
        dis_adj[0] = np.eye(num_joints)
        dis_adj[1] = A
        self.hops = 0 * dis_adj[0]    # values ==0 , shape : 25 25
        for i in range(2, num_joints):
            dis_adj[i] = dis_adj[i-1] @ A.transpose(0, 1)
            dis_adj[i][dis_adj[i] != 0] = 1  # 得到第 i 阶的矩阵

        for i in range(num_joints - 1, 0, -1):
            if np.any(dis_adj[i] - dis_adj[i - 1]):
                dis_adj[i] = dis_adj[i] - dis_adj[i - 1]  # 去掉重复的部分，也就是只在 i 阶出现，但是不在 i-1阶出现的值。
                self.hops += i * dis_adj[i]
            else:
                continue  # self.hops : 表示不同整数的矩阵。表示点和点之间的相对距离。

        self.hops = torch.tensor(self.hops).long()  # shape : 25,25,25 -> 25,25 将不同相对距离矩阵进行合并。
        # self.hops shape : 25 25 values is differnte distance

'''# test :
graph = NTU_Graph()
sencod = Structual_Encoding_re_distance(graph.A, 25, 32)
print(sencod.hops.shape)   # 25,25'''


# classic postioanl encoding

class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x

'''# test
x  = torch.rand(12, 16, 120, 25).cuda()
poencoding = PositionalEncoding(channel=16, joint_num=25, time_len=120, domain="spatial").cuda()

output = poencoding(x)
print(output.shape)'''



# 时间局部切块
class UnfoldTemporalWindows(nn.Module):
    ''' unfold temporal
        在时间维度进行切割，减少attention 的计算量。
        input : n c t v
        output : n c*kernal1*kernal2, blocks
    '''
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)  #(N, C*Window_Size, (T-Window_Size+1)*(V-1+1))
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)   # -1 : window numbers; self.window_size : local frames;
        return x

'''# test upfolds
x = torch.rand(4, 16, 120, 25).cuda()
net = UnfoldTemporalWindows(5, 1, 2).cuda()  #可以设定为连续的局部特征, stride 为 1， 即可。
out = net(x)
print(out.shape)   # torch.Size([4, 16, 40, 75])'''

### input_projection
class Input_Projection(nn.Module):
    ''' with motion informations '''
    def __init__(self, num_channel, in_channels,num_person, num_point,  use_motion = True, g = None):
        super().__init__()

        self.use_motion = use_motion
        if use_motion:
            self.input_map = nn.Sequential(
                nn.Conv2d(num_channel, in_channels // 2, 1),
                nn.BatchNorm2d(in_channels // 2),
                nn.LeakyReLU(0.1),
            )
            self.diff_map1 = nn.Sequential(
                nn.Conv2d(num_channel, in_channels // 8, 1),
                nn.BatchNorm2d(in_channels // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map2 = nn.Sequential(
                nn.Conv2d(num_channel, in_channels // 8, 1),
                nn.BatchNorm2d(in_channels // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map3 = nn.Sequential(
                nn.Conv2d(num_channel, in_channels // 8, 1),
                nn.BatchNorm2d(in_channels // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map4 = nn.Sequential(
                nn.Conv2d(num_channel, in_channels // 8, 1),
                nn.BatchNorm2d(in_channels // 8),
                nn.LeakyReLU(0.1),
            )
        else:
            self.input_map = nn.Sequential(
                nn.Conv2d(num_channel, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.1),
            )

        self.data_bn = nn.BatchNorm1d(num_person * num_channel * num_point)
        bn_init(self.data_bn, 1)
        # self.partion_encod = HyperPartionEmbedding(g=g,
        #                                            dim_in=num_channel,
        #                                            dim_out=num_channel,
        #                                            joints_num=num_point
        #                                            )
    def forward(self, x):
        # input N,C,T,V,M;  output shape: NM,C',T,V;
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)  # nm, c, t, v
        # x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        # x = self.partion_encod(x) + x
        if self.use_motion:
            #x = self.input_map(x)
            dif1 = x[:, :, 1:] - x[:, :, 0:-1]   # distance == 1
            dif1 = torch.cat([dif1.new(N*M, C, 1, V).zero_(), dif1], dim=-2)   # NM,C,T,V
            dif2 = x[:, :, 2:] - x[:, :, 0:-2]  # distance = 2
            dif2 = torch.cat([dif2.new(N*M, C, 2, V).zero_(), dif2], dim=-2)
            dif3 = x[:, :, :-1] - x[:, :, 1:]   # distance == -1
            dif3 = torch.cat([dif3, dif3.new(N*M, C, 1, V).zero_()], dim=-2)
            dif4 = x[:, :, :-2] - x[:, :, 2:]   # distance == -2
            dif4 = torch.cat([dif4, dif4.new(N*M, C, 2, V).zero_()], dim=-2)
            x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2), self.diff_map3(dif3), self.diff_map4(dif4)), dim = 1)
        else:
            x = self.input_map(x)
        return x

# test input projection :
'''x = torch.rand(16, 3, 120, 25, 2).cuda()
net = Input_Projection(num_channel=3, in_channels=64,num_person = 2, num_point = 25,  use_motion=False).cuda()
output = net(x)
print(output.shape)'''



class es_vit_layer(nn.Module):
    ''' dp_spatial_att + dp_temporal_att '''
    def __init__(self, dim_in, dim, num_heads,
                 attn_drop=0,
                 sp_attention = True,
                 sp_pe=True,    # spatial attention or not.
                 partition_pe = True,   # use partion encoding or not
                 soft_partion = True,   # softmax for it.
                 partion_pe_learnabel = True,   # partion encoding is learnable or not.
                 structal_enc = True,   # relative distance encoding for skeleton struction
                 att_type = 'dpgat', # gatv1, gatv2
                 sparsity = True,   # sparsity or fully connecting.
                 num_joints=25,
                 time_len = 120,
                 use_unfold = True,
                 window_size = 3,
                 window_dilation = 1,
                 att_soft = 'softmax',    # softmax or tan.
                 leakeyR = 0.1,
                 kernel_size=3,
                 t_stride=1,
                 pool_type='maxpool',  # avgpool
                 dilations=[1, 2],
                 t_type='mul_scale',  # 'mul_scale', 'tcn', 'none';
                 temporal_att='se_att',  # se_att, dpatt, none.
                 squeeze_factor = 1,
                 t_pe = True,
                 att_drop_t=0.0,
                 graph = None,
                 g= None,
                 ):
        super(es_vit_layer, self).__init__()
        self.dp_satt = dp_spatial_att(dim_in = dim_in, dim = dim, num_heads = num_heads,
                 attn_drop=attn_drop,
                 sp_attention = sp_attention,
                 sp_pe= sp_pe,    # spatial attention or not.
                 partition_pe = partition_pe,   # use partion encoding or not
                 soft_partion = soft_partion,   # softmax for it.
                 partion_pe_learnabel = partion_pe_learnabel,   # partion encoding is learnable or not.
                 structal_enc = structal_enc,   # relative distance encoding for skeleton struction
                 att_type = att_type, # gatv1, gatv2
                 bias_att=True,
                 sparsity = sparsity,   # sparsity or fully connecting.
                 num_joints=num_joints,
                 time_len = time_len,
                 use_unfold = use_unfold,
                 window_size = window_size,
                 window_dilation = window_dilation,
                 att_soft = att_soft,    # softmax or tan.
                 leakeyR = leakeyR,
                 graph= graph,
                 g = g,
                                      )

        self.dp_tatt = dp_temporal_att(
            dim_in = dim,
            dim_out = dim,
            kernel_size=kernel_size,
            t_stride=t_stride,
            pool_type= pool_type,  # avgpool, defalut : maxpool
            dilations= dilations,  # default [1,2]
            t_type= t_type,  # 'mul_scale', 'tcn', 'none';
            temporal_att= temporal_att,  # se_att, dpatt, none.
            squeeze_factor = squeeze_factor,   # 控制squeeze fator 的计算量。
            num_heads= num_heads,
            num_frames= time_len,
            num_joints=num_joints,
            att_drop= att_drop_t,
            t_pe = t_pe
        )

    def forward(self, x):
        # n c t v = x.shape
        y = self.dp_satt(x)
        y = self.dp_tatt(y)
        return y

### test es_vit layer :
'''x = torch.rand(16, 32, 120, 25).cuda()
net = es_vit_layer(dim_in =32, dim = 32, num_heads = 4,
                 attn_drop=0,
                 sp_attention = True,
                 sp_pe=True,    # spatial attention or not.
                 partition_pe = True,   # use partion encoding or not
                 soft_partion = True,   # softmax for it.
                 partion_pe_learnabel = True,   # partion encoding is learnable or not.
                 structal_enc = True,   # relative distance encoding for skeleton struction
                 att_type = 'gatv2', # gatv1, gatv2, dpgat
                 sparsity = True,   # sparsity or fully connecting.
                 num_joints=25,
                 time_len = 120,
                 use_unfold = True,
                 window_size = 3,
                 window_dilation = 1,
                 att_soft = 'softmax',    # softmax or tan.
                 leakeyR = 0.1,
                 kernel_size=3,  # temporal att params
                 t_stride=1,
                 pool_type='maxpool',  # avgpool, maxpool
                 dilations=[1, 2],
                 t_type='mul_scale',  # 'mul_scale', 'tcn', 'none';
                 temporal_att='se_att',  # se_att, dpatt, none.
                 squeeze_factor= 4,
                 t_pe = True,
                 att_drop_t=0.0,).cuda()
output = net(x)
print(output.shape, torch.isnan(output).max(), comput_params(net))'''
#  dpgat + dpatt : 75832;   dpgat + se : 40964;  dpgat + none : 11924;  dpgat + se(4) : 19274;
#  gatv2 + se(4) : 19274;



class es_vit(nn.Module):
    def __init__(self, dim_ins = [64, 64, 128, 128, 256, 256], dims = [64, 128, 128, 256, 256,256], num_heads = 4,
                 attn_drop=0,
                 sp_attention = True,
                 sp_pe=True,    # spatial attention or not.
                 partition_pe = True,   # use partion encoding or not
                 soft_partion = False,   # softmax for it.
                 partion_pe_learnabel = False,   # partion encoding is learnable or not.
                 structal_enc = True,   # relative distance encoding for skeleton struction
                 att_type = 'dpgat', # gatv1, gatv2, dpgat, gatv1 > gatv2 > dpgat
                 sparsity = True,   # sparsity or fully connecting.
                 num_joints=25,
                 time_len = 50,
                 num_class = 60,
                 use_unfold = True,   # 速度下降
                 window_size = 3,
                 window_dilation = 1,
                 att_soft = 'softmax',    # softmax or tan.
                 leakeyR = 0.1,
                 kernel_size=3,
                 t_stride=1,
                 pool_type='maxpool',  # avgpool
                 dilations=[1, 2],
                 t_type='mul_scale',  # 'mul_scale', 'tcn', 'none';  速度下降
                 temporal_att='se_att',  # se_att, dpatt, none.
                 squeeze_factor = 4,
                 t_pe = True,
                 att_drop_t=0.0,
                 use_motion = True,
                 partion = [0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1],  # partion for ntu datasets
                 ):
        # todo : stocastica drop rate for different layers
        super(es_vit, self).__init__()

        if num_joints == 25:
            num_person = 2
            graph = NTU_Graph()  # can get graph.A = list :3 25 x25, graph.dis_adjs->list: 3, 25 x25; graph.hops-> 25 x25, values 0 - 12; or 0-11
        elif num_joints == 20:
            num_person = 1
            graph = UCLA_Graph()  # todo: 将graph 本地化。
        elif num_joints == 22:
            graph = DHG_Graph()   # DHG_GRAPH == SHREC GRAPH
            num_person = 1
        self.graph = graph
        p = np.array(F.one_hot(torch.tensor(partion)).float())
        self.g = torch.from_numpy(generate_G_from_H(p)).float()  # numpy arry shape : V V
        self.input_map = Input_Projection(3, dim_ins[0], num_person = num_person, num_point = num_joints, use_motion=use_motion, g=self.g)   # input shape : n c t v m -> nm, c,t,v;
        self.st_vit_layers = nn.ModuleList()
        for i in range(len(dim_ins)):
            self.st_vit_layers.append(
                es_vit_layer(
                    dim_in = dim_ins[i],
                    dim = dims[i],
                    num_heads = num_heads,
                    attn_drop= attn_drop * (i/len(dim_ins)),
                    sp_attention= sp_attention,
                    sp_pe= sp_pe,  # spatial attention or not.
                    partition_pe= partition_pe,  # use partion encoding or not
                    soft_partion= soft_partion,  # softmax for it.
                    partion_pe_learnabel= partion_pe_learnabel,  # partion encoding is learnable or not.
                    structal_enc= structal_enc,  # relative distance encoding for skeleton struction
                    att_type= att_type,  # gatv1, gatv2
                    sparsity= sparsity,  # sparsity or fully connecting.
                    num_joints= num_joints,
                    time_len= time_len,   #
                    use_unfold=use_unfold,
                    window_size=window_size,
                    window_dilation=window_dilation,
                    att_soft= att_soft,  # softmax or tan.
                    leakeyR= leakeyR,
                    kernel_size= kernel_size,
                    t_stride= t_stride,   #
                    pool_type= pool_type,  # avgpool
                    dilations= dilations,
                    t_type= t_type,  # 'mul_scale', 'tcn', 'none';
                    temporal_att= temporal_att,  # se_att, dpatt, none.
                    squeeze_factor= squeeze_factor,
                    t_pe=t_pe,
                    att_drop_t= att_drop_t * (i/len(dim_ins)),
                    graph=self.graph,
                    g = self.g,
                             )
            )

        self.fc = nn.Linear(dims[-1], num_class)


    def forward(self, x):
        # n, c,t,v,m
        N,C,T,V,M = x.shape

        x = self.input_map(x)

        for i, layer in enumerate(self.st_vit_layers):
            x = layer(x)   # n,c,t,v

        x = rearrange(x, '(n m) c t v -> n m c (t v)', m = M)
        x = x.mean(-1).mean(1)
        x = self.fc(x)
        return x

### test es_vit
'''x = torch.rand(16, 3, 50, 25, 2).cuda()  # ntu datasets
net = es_vit().cuda()
output = net(x)
print(output.shape, torch.isnan(output).max(), comput_params(net))'''
# torch.Size([16, 60]) tensor(False, device='cuda:0') 1941012


# test spatial attention :
'''x = torch.rand(10, 16, 120, 25).cuda()
# todo : num_heads set
net = dp_spatial_att(dim_in=16, dim=32, num_heads=4,
                 attn_drop=0,
                 sp_attention = True,
                 sp_pe=True,    # spatial attention or not.
                 partition_pe = True,   # use partion encoding or not
                 soft_partion = True,   # softmax for it.
                 partion_pe_learnabel = True,   # partion encoding is learnable or not.
                 structal_enc = True,   # relative distance encoding for skeleton struction
                 att_type = 'dpgat', # gatv1, gatv2, dpgat
                 sparsity = True,   # sparsity or fully connecting.
                 bias_att = True,   # attention bias
                 num_joints=25,
                 time_len = 120,
                 use_unfold = False,
                 window_size = 3,
                 window_stride = 1,   # stride == 1, make the output T' == T
                 window_dilation = 1,
                 att_bias=True,
                 att_soft = 'softmax',    # softmax or tan.#
                ).cuda()

output = net(x)
print(output.shape)
print(torch.isnan(output).max())
'''
# output :
# torch.Size([10, 32, 120, 25])
# tensor(False, device='cuda:0')