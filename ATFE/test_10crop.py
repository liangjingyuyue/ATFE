import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve,confusion_matrix,roc_auc_score
import numpy as np
def scorebinary(scores=None, threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold

def test(dataloader, model, args, viz, device):
    mifa = 100
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device) 
        for i, input in enumerate(dataloader):
            input = input.to(device) 
            input = input.permute(0, 2, 1, 3) 
            # score_abnormal : (1,1); feat_select_abn:(10,3,2048); logits (1,28,1); scores_nor_bottom:(10,3,2048); feat_magnitudes:(1,28)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1) 
            logits = torch.mean(logits, 0) 
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai': # /home/ubuntu/PycharmProjects/DataSet/shanghaitech/list/gt-sh-test.npy or list/gt-sh.npy
            # gt = np.load('/home/ubuntu/PycharmProjects/DataSet/shanghaitech/list/gt-sh-test.npy')
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ped2':
            gt = np.load('list/gt-ped2-ver1.npy') # (2070,)
        elif args.dataset == 'avenue':
            gt = np.load('list/gt-avenue.npy') # (18188,)
        else:
            gt = np.load('list/gt-ucf.npy')
        pred = list(pred.cpu().detach().numpy()) 
        pred = np.repeat(np.array(pred), 16)
        '''
            fpr : ndarray of shape (>2,)
            tpr : ndarray of shape (>2,)
            thresholds : ndarray of shape = (n_thresholds,)
        '''
        fpr, tpr, threshold = roc_curve(list(gt), pred) # fpr={ndarray:(856,)} tpr={ndarray:(856,)}, threshold={ndarray:(856,)}
        pred1 = scorebinary(pred, threshold=0.8)
        tn, fp, fn, tp = confusion_matrix(gt, pred1).ravel()
        all_ano_false_alarm = fp / (fp + tn)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('all_ano_false_alarm = ',all_ano_false_alarm, ', auc :' , str(rec_auc),)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc, all_ano_false_alarm

