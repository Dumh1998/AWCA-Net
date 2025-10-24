import sys
from main import parse_args
from PIL import Image
import numpy as np
from utils.model_builder import build_model

sys.path.insert(0, '')

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset.dataset as myDataLoader
import dataset.Transforms as myTransforms
from utils.metric_tool import ConfuseMatrixMeter
from utils.utils import BCEDiceLoss

import os, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def visilize(args, model, loader, device, model_name, dataset_name,savedir):
    ressavepath = savedir + '/img'


    # 如果目录不存在，创建目录
    print('load in'+ressavepath)
    if not os.path.exists(ressavepath):
        os.makedirs(ressavepath)

    model.eval()
    for iter, batched_inputs in enumerate(loader):
        img, target, name = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            post_img = post_img.cuda()

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)
        p4, p3, p2, p1 = output
        pred = torch.where(p1 > 0.5, torch.ones_like(p1), torch.zeros_like(p1)).long()

        k1 = pred.cpu().detach().numpy()
        for i in range(len(k1)):
            img = k1[i]
            saven = name[i].split('/')[-1]
            resimg1 = np.squeeze(img * 255.0)
            image = Image.fromarray(np.uint8(resimg1))
            savepath = os.path.join(ressavepath, saven)
            image.save(savepath)


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, target, name = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)
        p4, p3, p2, p1 = output
        l1 = BCEDiceLoss(p1, target_var)
        l2 = BCEDiceLoss(p2, target_var)
        l3 = BCEDiceLoss(p3, target_var)
        l4 = BCEDiceLoss(p4, target_var)
        loss = l1 + l2 + l3 + l4
        pred = torch.where(p1 > 0.5, torch.ones_like(p1), torch.zeros_like(p1)).long()

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores

def ValidateSegmentation(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    model = build_model(args.model_name)

    args.savedir = args.savedir + args.model_name + '_' + args.file_root + '_iter_' + str(
        args.max_steps) + '_lr_' + str(args.lr) + '/'
    if args.file_root == 'xxx':
        args.file_root = 'xxxx'
    elif args.file_root == 'xxxx':
        args.file_root = 'xxx'

    if args.onGPU:
        model = model.cuda()

    mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset(file_root=args.file_root, mode="test", transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=8, num_workers=args.num_workers, pin_memory=True)

    if args.onGPU:
        cudnn.benchmark = True

    # load the models
    model_file_name = args.savedir + 'best_model.pth'
    print(model_file_name)
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, testLoader, model)
    print(
        "\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f\t  OA (te) = %.4f" \
        % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision'],
           score_test['OA']))

    visilize(args, model, testLoader, device, args.model_name, args.file_root, args.savedir)


if __name__ == '__main__':
    ValidateSegmentation(parse_args())
