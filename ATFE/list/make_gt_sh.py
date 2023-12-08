import numpy as np
import os
import glob
import numpy as np
from scipy.io import loadmat
from os import walk

'''
FOR UCF CRIME
'''
root_path = "/home/ubuntu/PycharmProjects/DataSet/SH_Test_ten_crop_i3d/"
dirs = os.listdir(root_path)
rgb_list_file ='/home/ubuntu/PycharmProjects/myProject_testTopK/list/shanghai-i3d-test-10crop_1.list'
temporal_root = '/home/ubuntu/PycharmProjects/DataSet/UCF_and_Shanghai/ShanghaiTech/testing/test_frame_mask/'
# mat_name_list = os.listdir(temporal_root)
gt_files = os.listdir(temporal_root)
file_list = list(open(rgb_list_file))
num_frame = 0
gt = []
index = 0
total = 0
abnormal_count =0
for  file in file_list:

    features = np.load(file.strip('\n'), allow_pickle=True)

    # features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    # features = np.squeeze(features, axis=1)

    num_frame = features.shape[0] * 16

    count = 0
    if index > 43:
        print('normal video' + str(file))
        for i in range(0, num_frame):
            gt.append(0)
            count += 1

    else:
        print('abnormal video' + str(file))
        gt_file = file.split('_i3d.npy')[0] + '.npy'
        gt_file = gt_file.split('/')[-1]
        if not os.path.isfile(os.path.join(temporal_root, gt_file)):
            print('no such file')
            exit(1)
        abnormal_count += 1
        ground_annotation = np.load(os.path.join(temporal_root, gt_file))
        ground_annotation = list(ground_annotation)
        if len(ground_annotation) < num_frame:
            last_frame_label = ground_annotation[-1]
            for i in range(len(ground_annotation), num_frame):
                ground_annotation.append(last_frame_label)

        if len(ground_annotation)!= num_frame:
            print("wrong frame number")
            exit(1)
        count += len(ground_annotation)
        gt.extend(ground_annotation)

    index = index + 1
    total += count

print(abnormal_count)






