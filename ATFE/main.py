from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model_test.new_model3 import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import datetime

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)
# python -m visdom.server
if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False) # 注意test_loader的 batch_size = 1

   
    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    # 评估模型的参数量
    print_model_parm_nums(model)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''   # put your own path here
    # auc = test(test_loader, model, args, viz, device)
    sumauc = 0.0
    cnt = 0

    starttime = datetime.datetime.now()
    print(starttime)
    minfa = 100
    maxauc = 0
    sumfa = 0
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)
        # 执行到这里，跳转到train.py
        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)
        if step % 5 == 0 and step > 200: 
            cnt += 1
            auc,fa = test(test_loader, model, args, viz, device)

            if fa==0.0:
                minfa = minfa
            else:
                minfa = min(minfa,fa)
            if auc==1.0:
                maxauc = maxauc
            else:
                maxauc = max(maxauc,auc)
            sumauc += maxauc
            sumfa += minfa
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(maxauc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
       torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
endtime = datetime.datetime.now()
