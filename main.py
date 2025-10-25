import logging
logging.getLogger().setLevel(logging.WARNING)
import sys
from utils.model_builder import build_model
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms
from utils.metric_tool import ConfuseMatrixMeter
from utils.utils import BCEDiceLoss, init_seed, adjust_learning_rate
import os, time
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, '')


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):
        loss = 0.0
        img, target, name = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]
        start_time = time.time()
        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        output = model(pre_img_var, post_img_var)
        p4, p3, p2, p1 = output
        l1 = BCEDiceLoss(p1, target_var)
        l2 = BCEDiceLoss(p2, target_var)
        l3 = BCEDiceLoss(p3, target_var)
        l4 = BCEDiceLoss(p4, target_var)
        loss = l1 + l2 + l3 + l4
        pred = torch.where(p1 > 0.5, torch.ones_like(p1), torch.zeros_like(p1)).long()

        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
                  end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    model.train()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    for iter, batched_inputs in enumerate(train_loader):
        loss = 0.0
        img, target, name = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]
        start_time = time.time()
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)
        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()
        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()
        output = model(pre_img_var, post_img_var)
        p4,p3,p2,p1 = output
        l1 = BCEDiceLoss(p1, target_var)
        l2 = BCEDiceLoss(p2, target_var)
        l3 = BCEDiceLoss(p3, target_var)
        l4 = BCEDiceLoss(p4, target_var)
        loss = l1 + l2 + l3 + l4
        pred = torch.where(p1 > 0.5, torch.ones_like(p1), torch.zeros_like(p1)).long()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.data.item(),
                res_time),
                  end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr


def trainVal(args):
    torch.backends.cudnn.benchmark = True
    init_seed(args.seed)
    args.savedir = args.savedir + args.model_name + '_' + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/'
    
    if args.file_root == 'xxx':
        args.file_root = 'xxxx'
    elif args.file_root == 'xxxx':
        args.file_root = 'xxx'

    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model = build_model(args.model_name)
    if args.onGPU:
        model = model.cuda()

    mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset(file_root=args.file_root, mode="train", transform=trainDataset_main)
    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_data = myDataLoader.Dataset(file_root=args.file_root, mode="test", transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    max_batches = len(trainLoader)

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    if args.resume is not None:
        args.resume = args.savedir + 'checkpoint.pth.tar'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * len(trainLoader)
            # args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa (val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)', 'OA (val)'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    for epoch in range(start_epoch, args.max_epochs):
        lossTr, score_tr, lr = \
            train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        if epoch == 0:
            continue

        lossVal, score_val = val(args, testLoader, model)
        torch.cuda.empty_cache()
        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, score_val['Kappa'], score_val['IoU'],
                                                                       score_val['F1'], score_val['recall'],
                                                                       score_val['precision'], score_val['OA']))
        logger.flush()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        model_file_name = args.savedir + 'best_model.pth'
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)

        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f" \
              % (epoch, lossTr, lossVal, score_tr['F1'], score_val['F1']))
        torch.cuda.empty_cache()

    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, testLoader, model)
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f\t OA (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision'], score_test['OA']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['Kappa'], score_test['IoU'],
                                                                   score_test['F1'], score_test['recall'],
                                                                   score_test['precision'], score_test['OA']))
    logger.flush()
    logger.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default="AWCANet", help='')
    parser.add_argument('--file_root', default="Data", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--seed', default=16, help='initialization seed number')
    parser.add_argument('--savedir', default='./results/', help='Directory to save the results')
    parser.add_argument('--resume', default=True, help='Use this checkpoint to continue training | '
                                                       './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--gpu_id', default=4, type=int, help='GPU id number')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    trainVal(parse_args())


