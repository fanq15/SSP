from dataset.fewshot import FewShot
from model.SSP_matching import SSP_MatchingNet
from util.utils import count_params, set_seed, mIOU

import argparse
import os
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Mining Latent Classes for Few-shot Segmentation')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal', 'coco'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--refine', dest='refine', action='store_true', default=False)

    # few-shot training arguments
    parser.add_argument('--fold',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3],
                        help='validation fold')
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')

    args = parser.parse_args()
    return args


def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    num_classes = 21 if args.dataset == 'pascal' else 81
    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_q, mask_q = img_q.cuda(), mask_q.cuda()
        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()
        cls = cls[0].item()

        with torch.no_grad():
            pred = model(img_s_list, mask_s_list, img_q, None)[0]
            pred = torch.argmax(pred, dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    save_path = 'outdir/models/%s/fold_%i' % (args.dataset, args.fold)
    os.makedirs(save_path, exist_ok=True)

    testset = FewShot(args.dataset, args.data_root, None, 'val',
                      args.fold, args.shot, 1000 if args.dataset == 'pascal' else 4000)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model = SSP_MatchingNet(args.backbone, args.refine)
    checkpoint_path = glob.glob('outdir/models/' + args.dataset + '/fold_' + str(args.fold) + '/*.pth')[0]
    #checkpoint_path = glob.glob('outdir_coco_50_5/models/' + args.dataset + '/fold_' + str(args.fold) + '/*.pth')[0]

    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    #print(model)
    print('\nParams: %.1fM' % count_params(model))

    best_model = DataParallel(model).cuda()

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')


if __name__ == '__main__':
    main()

