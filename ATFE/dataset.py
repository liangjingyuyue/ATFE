import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    # 进入Dataset后，先执行__init__函数，对于train_nloader,其 is_normal=True;对于train_aloader，其 is_normal=False;
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop_1.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop_1.list'
        elif self.dataset == 'ped2':
            if test_mode:
                self.rgb_list_file = 'list/ped2-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/ped2-i3d-train-10crop.list'
        elif self.dataset == 'avenue':
            if test_mode:
                self.rgb_list_file = 'list/avenue-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/avenue-i3d-train-10crop.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'
            # if test_mode:
            #     self.rgb_list_file = 'list/UCF/Abuse_test.list'
            # else:
            #     self.rgb_list_file = 'list/UCF/Abuse.list'

        self.tranform = transform
        self.test_mode = test_mode
        self.feature_path = args.feature_path
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal: # [63,238]为正常视频
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63] # [0,62]为异常视频
                    print('abnormal list for shanghai tech')
                    print(self.list)
            elif self.dataset == 'ped2':
                if self.is_normal: # [63,238]为正常视频
                    self.list = self.list[6:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:6] # [0,62]为异常视频
                    print('abnormal list for shanghai tech')
                    print(self.list)
            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)
                # if self.is_normal:
                #     self.list = self.list[48:]
                #     print('normal list for ucf')
                #     print(self.list)
                # else:
                #     self.list = self.list[:48]
                #     print('abnormal list for ucf')
                #     print(self.list)

    # 当执行train(loadern_iter, loadera_iter, model, ..., )下的 ninput, nlabel = next(nloader) 时，才执行 __getitem__ 函数
    def __getitem__(self, index):
        # 在这里 ncrops = 10，T则不定
        label = self.get_label()  # get video level label 0/1

        features = np.load(self.feature_path + self.list[index].split('\n')[0], allow_pickle=True) #ped2 时使用这个
        # features = np.load(self.feature_path + self.list[index].split('_i3d.npy')[0] + '.npy', allow_pickle=True) # shanghaitech swin 时使用这个
        # features = np.load(self.list[index].strip('\n'), allow_pickle=True) # shanghaitech i3d时使用这个
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, T, F],
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments：有的T>32，就只取32个segment，有的T<32，则补充齐32个segment
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32) # (10,32,F)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
