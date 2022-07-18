r""" Hypercorrelation Squeeze testing code """
import argparse

import torch.nn.functional as F
import torch.nn as nn
import torch

from hsnet_test.common.logger import Logger, AverageMeter
from hsnet_test.common.vis import Visualizer
from hsnet_test.common.evaluation import Evaluator
from hsnet_test.common import utils
from hsnet_test.data.dataset import FSSDataset
from model.SSP_matching import SSP_MatchingNet
import glob

def test(model, dataloader, nshot):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)
        img_s_list = batch['support_imgs']
        mask_s_list = batch['support_masks']
        img_q = batch['query_img']
        
        img_s_list = [i.unsqueeze(0) for i in img_s_list.squeeze(0)]
        mask_s_list = [i.unsqueeze(0) for i in mask_s_list.squeeze(0)]

        #print(img_q.shape, mask_s_list.shape, img_s_list.shape)
        pred_mask = model(img_s_list, mask_s_list, img_q, None)[0] #model.module.predict_mask_nshot(batch, nshot=nshot)
        pred_mask = torch.argmax(pred_mask, dim=1)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=50)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = SSP_MatchingNet(args.backbone) #HypercorrSqueezeNetwork(args.backbone, args.use_original_imgsize)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    checkpoint_path = glob.glob(args.load + '/*')
    print(checkpoint_path)
    assert len(checkpoint_path) == 1
    checkpoint_path = checkpoint_path[0]
    model.module.load_state_dict(torch.load(checkpoint_path))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
