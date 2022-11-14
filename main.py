from args import get_args
import torch
from torchvision import transforms
import numpy as np
import os
import sys
import json
import time
import pickle
from tqdm import tqdm
from torch.autograd import Variable

from polyvore_outfits import TripletImageLoader
from torchvision.models import resnet18
from tpye_specific_network import TypeSpecificNet
from tripletnet import Tripletnet

args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA status:",torch.cuda.is_available())
if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.manual_seed(1000)
    print("已设置GPU随机种子")
else:
    args.cuda = False
    torch.manual_seed(args.seed)
    print("已设置CPU随机种子")

def test(test_loader, tnet):
    # switch to evaluation mode

    tnet.eval()
    embeddings = []
    pbar = tqdm(total=len(test_loader), ncols=150)
    for batch_idx, images in enumerate(test_loader):
        if args.cuda:
            images = images.cuda()
        images = Variable(images)
        with torch.no_grad():
            embedding = tnet.embeddingnet(images).data
        embeddings.append(embedding)
        # embeddings.append(tnet.embeddingnet(images)[0].cpu())
        pbar.set_description(
            "batch_idx:{:d}".format(
                batch_idx)
        )
        if batch_idx%100==0:
            print("batch_idx: ",batch_idx)
        pbar.update(1)


    embeddings = torch.cat(embeddings)
    metric = tnet.metric_branch
    auc = test_loader.dataset.test_compatibility(embeddings, metric)
    acc = test_loader.dataset.test_fitb(embeddings, metric)
    total = auc + acc
    print('\n{} set: Compat AUC: {:.2f} FITB: {:.1f}\n'.format(
        test_loader.dataset.split,
        round(auc, 2), round(acc * 100, 1)))

    return total

class AverageMeter(object):
    """
    用来记录每次的结果, 并辅助计算平均指标的值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainData():
    def __init__(self, images, text, has_text, conditions=None):
        has_text = has_text.float()
        if args.cuda:
            images, text, has_text = images.cuda(), text.cuda(), has_text.cuda()
        images, text, has_text = Variable(images), Variable(text), Variable(has_text)

        if conditions is not None and not args.use_fc:
            if args.cuda:
                conditions = conditions.cuda()

            conditions = Variable(conditions)

        self.images = images
        self.text = text
        self.has_text = has_text
        self.conditions = conditions

    def __len__(self):
        return self.images.size(0)

def train(train_loader, tnet, criterion, optimizer, epoch):
    print("epoch: ", epoch)
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    pbar = tqdm(total=len(train_loader),ncols=150)
    for batch_idx, (img1, desc1, has_text1, img2, desc2, has_text2, img3, desc3, has_text3, condition) in enumerate(train_loader):

        # TrainData是一个封装类, 包括了图片,文本,类别, 可以用来抽象地表示A,P,N三个角色
        anchor = TrainData(img1, desc1, has_text1, condition)
        close = TrainData(img2, desc2, has_text2)
        far = TrainData(img3, desc3, has_text3)

        # compute output
        acc, loss_triplet, loss_mask, loss_embed, loss_vse, loss_sim_t, loss_sim_i = tnet(anchor, far, close)

        # print("acc:",acc," loss_triplet:",loss_triplet," loss_mask:",loss_mask, " loss_embed:", loss_embed, " loss_vse:",loss_vse, " loss_sim_t:",loss_sim_t," loss_sim_i:",loss_sim_i)

        # encorages similar text inputs (sim_t) and image inputs (sim_i) to
        # embed close to each other, images operate on the general embedding
        loss_sim = args.sim_t_loss * loss_sim_t + args.sim_i_loss * loss_sim_i

        # cross-modal similarity regularizer on the general embedding
        loss_vse_w = args.vse_loss * loss_vse

        # sparsity and l2 regularizer
        loss_reg = args.embed_loss * loss_embed + args.mask_loss * loss_mask

        loss = loss_triplet + loss_reg
        if args.vse_loss > 0:
            loss += loss_vse_w
        if args.sim_t_loss > 0 or args.sim_i_loss > 0:
            loss += loss_sim

        num_items = len(anchor)
        # measure accuracy and record loss
        losses.update(loss_triplet.item(), num_items)
        accs.update(acc.item(), num_items)
        emb_norms.update(loss_embed.item())
        mask_norms.update(loss_mask.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()

        if loss == loss:
            loss.backward()
            optimizer.step()
        # pbar.set_description(
        #     "epoch {}, step {}".format(
        #         epoch, batch_idx)
        # )
        pbar.set_description(
            "Epoch:{:d} Loss:{:.4f}({:.4f}) Acc:{:.2f}%({:.2f}%) Emb_Norm:{:.2f}({:.2f})".format(
                epoch, losses.val, losses.avg, 100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg)
        )
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{}]\t'
        #           'Loss: {:.4f} ({:.4f}) \t'
        #           'Acc: {:.2f}% ({:.2f}%) \t'
        #           'Emb_Norm: {:.2f} ({:.2f})'.format(
        #         epoch, batch_idx * num_items, len(train_loader.dataset),
        #         losses.val, losses.avg,
        #                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
        pbar.update(1)


def main():
    print("This is Type-aware reimplemented by xyy and yyc.")

    #对图像的一个变换,每个通道指定均值和标准差
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # 指定数据集位置
    fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
    # 读取元数据
    meta_data = json.load(open(fn, 'r'))
    # 指定文本嵌入维度
    text_feature_dim = 6000 # xyy: 可以加到arg里去, 不要写在这
    # 配置数据集加载参数
    # kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    kwargs = {}

    print("setting test_loader...")
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader(args, 'test', meta_data,
                           transform=transforms.Compose([
                               transforms.Scale(112),
                               transforms.CenterCrop(112),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)



    criterion = torch.nn.MarginRankingLoss(margin=args.margin)

    if args.resume:
        # model = resnet18(pretrained=True)
        # csn_model = TypeSpecificNet(model, len(test_loader.dataset.typespaces))  # 之后会被传入tnet作为求图像embedding的网络
        print("")
        print("load TNET by resuming from the model checkpoint: ", args.checkpoint_name)
        tnet = torch.load('./model_saved/' + args.checkpoint_name)
        print("finsih resuming model checkpoints")
    else:
        print("load TNET by instantiating a new model and train from scratch")
        model = resnet18(pretrained=True)
        csn_model = TypeSpecificNet(model, len(test_loader.dataset.typespaces))  # 之后会被传入tnet作为求图像embedding的网络
        tnet = Tripletnet(csn_model, text_feature_dim, criterion)
        print("finish loading the model")

    if args.cuda:
        tnet.cuda()

    if not args.test:
        print("setting train_loader...")
        train_loader = torch.utils.data.DataLoader(
            TripletImageLoader(args, 'train', meta_data,
                               text_dim=text_feature_dim,
                               transform=transforms.Compose([
                                   transforms.Scale(112),
                                   transforms.CenterCrop(112),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   normalize,
                               ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        print('finish setting train_loader')

    print("setting val_loader...")
    val_loader = torch.utils.data.DataLoader(
        TripletImageLoader(args, 'valid', meta_data,
                           transform=transforms.Compose([
                               transforms.Scale(112),
                               transforms.CenterCrop(112),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    best_acc = 0
    cudnn.benchmark = True
    if args.test:
        print("start testing...")
        test_acc = test(test_loader, tnet)
        sys.exit()
    else:
        parameters = filter(lambda p: p.requires_grad, tnet.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
        print('  + Number of params: {}'.format(n_parameters))

        for epoch in range(args.start_epoch, args.epochs + 1):
            # update learning rate
            lr = args.lr * ((1 - 0.015) ** epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # train for one epoch
            train(train_loader, tnet, criterion, optimizer, epoch)
            # evaluate on validation set
            # acc = test(val_loader, tnet)
            # acc = test(test_loader, tnet)

            # remember best acc and save checkpoint
            # is_best = acc > best_acc
            # best_acc = max(acc, best_acc)
            # 保存模型
            torch.save(tnet,
                       './model_saved/' + "model_" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + "_epoch_{}".format(epoch)+".pt")


if args.print2file:
    # print the log to file
    import logging, sys, time
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    logging.basicConfig(filename='./logger/Type_aware '+current_time+".txt", level=logging.DEBUG)
    logger = logging.getLogger()
    sys.stderr.write = logger.error
    sys.stdout.write = logger.info

if __name__ == '__main__':
    main()