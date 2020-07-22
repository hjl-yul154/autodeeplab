import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.metrics import Evaluator
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    save_dir = os.path.join('./data/', args.save_path)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model_fname = os.path.join(save_dir,
                               'deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp))
    record_name = os.path.join(save_dir, 'training_record.txt')
    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes, val_loader = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(dataset_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))
    start_epoch = 0

    evaluator=Evaluator(num_classes)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        print('Training epoch {}'.format(epoch))
        model.train()
        for i, sample in enumerate(dataset_loader):
            cur_iter = epoch * len(dataset_loader) + i
            scheduler(optimizer, cur_iter)
            inputs = sample['image'].cuda()
            target = sample['label'].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % 200 == 0:
                print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                    epoch + 1, i + 1, len(dataset_loader), scheduler.get_lr(optimizer), loss=losses))
        if epoch < args.epochs:
            if (epoch+1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fname % (epoch + 1))
        else:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_fname % (epoch + 1))

        line0 = 'epoch: {0}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
            epoch, loss=losses)
        with open(record_name, 'a') as f:
            f.write(line0)
            if line0[-1] != '\n':
                f.write('\n')

        if epoch%3!=0 and epoch <args.epochs-20:
            continue

        print('Validate epoch {}'.format(epoch))
        model.eval()
        evaluator.reset()
        test_loss=0.0
        for i,sample in enumerate(val_loader):
            inputs = sample['image'].cuda()
            target = sample['label'].cuda()
            with torch.no_grad():
                outputs = model(inputs)
            # loss = criterion(outputs, target)
            # test_loss+=loss.item()
            pred=outputs.data.cpu().numpy()
            target=target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.add_batch(target,pred)
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print("epoch: {}\t Acc:{:.3f}, Acc_class:{:.3f}, mIoU:{:.3f}, fwIoU: {:.3f}".format(epoch,Acc, Acc_class, mIoU, FWIoU))

        line1='epoch: {}\t''mIoU: {:.3f}'.format(epoch,mIoU)
        with open(record_name, 'a') as f:
            f.write(line1)
            if line1[-1] != '\n':
                f.write('\n')


if __name__ == "__main__":
    main()
