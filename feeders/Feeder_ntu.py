import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

NTU_60_XSUB_PATH = '/content/drive/MyDrive/Datasets/NTU-60/NTU60_XSub.npz'
NTU_60_XVIEW_PATH = '/content/drive/MyDrive/Datasets/NTU-60/NTU60_XView.npz'


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False):
        """

        data_path:
        label_path:
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        two new modality : bone, and motion we can use.
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug  # added debug mod to test training process or testing process.
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval  # list like [0.95] for test or [0.5, 1] for train
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # input data shape : N,T,M*25*3
        # label :  one hot
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            # np.where() get the index of label -> int.
            self.label = np.where(npz_data['y_train'] > 0)[1]

            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        # data shape : nt, mvc
        N, T, _ = self.data.shape  # data shape : N,T,M*25*3 需要进行 resshape

        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N, C, T, 25, M

        if self.debug:
            self.data = self.data[:1000]
            self.label = self.label[:1000]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        # mean map for n,t,m
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data shape : N,C,T,25,M
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # C, T, 25, M
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = random_rot(data_numpy)
        if self.bone:
            # 骨骼数据
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                         (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                         (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                         (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            # 运动数据
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index
        # c,t,v,m


# data format :
# /content/drive/MyDrive/Datasets/NTU-60/NTU60/ntu60_xsub/train
# /content/drive/MyDrive/Datasets/NTU-60/NTU60/ntu60_xsub/val
#  300, 150
# - train
#   - train_1.npz
#   - train_2.npz
#   - train_3.npz

# - val
#   - val_1.npz
#   - val_2.npz
#   - val_3.npz

import os


class Feeder2(Dataset):
    def __init__(self, data_path, split="train", p_interval=1, random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, debug=False, use_mmap=True,
                 bone=False, vel=False):
        """

        data_path: trian or val folder
        label_path : no need hear
        split: training set or test set
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move:
        random_rot: rotate skeleton around xyz axis
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        use_mmap: If true, use mmap mode to load data, which can save the running memory
        two new modality : bone, and motion we can use.
        bone: use bone modality or not
        vel: use motion modality or not
        only_label: only load label for ensemble score compute
        """

        self.debug = debug  # added debug mod to test training process or testing process.
        self.split = split
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.use_mmap = use_mmap
        self.p_interval = p_interval  # list like [0.95] for test or [0.5, 1] for train
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data shape : N,C,T,25,M
        # N, T, _ = self.data.shape  # data shape : N,T,M*25*3

        file_name = os.path.join(self.data_path, self.split + "_" + str(index) + ".npz")
        data_np = np.load(file_name, mmap_mode='r')  # only read mode
        # data numpy : data, label
        data = data_np["data"]
        T, _ = data.shape  # T, M*25*3
        data = data.reshape((T, 2, 25, 3)).transpose(3, 0, 2, 1)  # C,T,V,M
        label = data_np["label"]

        data_numpy = np.array(data)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = random_rot(data_numpy)
        if self.bone:
            # 骨骼数据
            ntu_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                         (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                         (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                         (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12))
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            # 运动数据
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index
        # c,t,v,m


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',
                         align_corners=False).squeeze()  # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V * M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot
