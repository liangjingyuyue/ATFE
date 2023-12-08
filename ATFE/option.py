import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feat-extractor', default='swin', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--rgb-list', default='list/avenue-i3d-train-10crop.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='list/avenue-i3d-test-10crop.list', help='list of test rgb features ')
# /home/ubuntu/DataSet/Avenue/avenue_swin/
# /home/ubuntu/PycharmProjects/DataSet/ped2/
# /home/ubuntu/DataSet/Avenue/Swin_feature/bs10/
# /home/ubuntu/PycharmProjects/DataSet/Ped2/videoswin/bs30_ped2/
parser.add_argument('--feature_path', default='/home/ubuntu/PycharmProjects/DataSet/Ped2/i3d/bs20_ped2/', help='list of test swin features ') # /home/ubuntu/DataSet/Avenue/avenue_swin/
# parser.add_argument('--feature_path', default='/PythonProject/Download/features/', help='list of test swin features ') # VideoSwin : /PythonProject/Download/features/
parser.add_argument('--gt', default='list/gt-sh.npy', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
# parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=2, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
# avenue shanghai ped2 ucf
parser.add_argument('--dataset', default='ped2', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=15000, help='maximum iteration to train (default: 15000)')
