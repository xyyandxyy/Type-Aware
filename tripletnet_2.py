import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import get_args
args = get_args()

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out),
                         nn.BatchNorm1d(f_out,eps=0.001,momentum=0.01),
                         nn.ReLU(inplace=True))


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        '''
        feat_dim: 原来文字embedding的维度, 预训练是6000维
        embedding_dim: 重新投影
        '''
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim) # 一层全连接
        self.fc2 = nn.Linear(embedding_dim, embedding_dim) # 第二层全连接, 没有BN和ReLU


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        # L2 normalize each feature vector
        # norm = torch.norm(x, ord=2, dim=1) + 1e-10
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        # print("norm.shape:",norm.shape)
        # debug: x:(256,64), norm:(256,)
        norm = norm.unsqueeze(dim=1) # xyy: 这里可以的话对比debug一下源代码
        x = x / norm.expand_as(x)
        return x


def accuracy(pos_samples, neg_samples):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
    """
    is_cuda = pos_samples.is_cuda
    margin = 0
    pred = (pos_samples - neg_samples - margin).cpu().data
    acc = (pred > 0).sum() * 1.0 / pos_samples.size()[0]
    acc = torch.from_numpy(np.array([acc], np.float32))
    if is_cuda:
        acc = acc.cuda()

    return Variable(acc)

def selective_margin_loss(pos_samples, neg_samples, margin, has_sample):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
        margin: minimum desired margin between pos and neg samples
        has_sample: Indicates if the sample should be used to calcuate the loss
    """
    margin_diff = torch.clamp((pos_samples - neg_samples) + margin, min=0, max=1e6)
    num_sample = max(torch.sum(has_sample), 1)
    return torch.sum(margin_diff * has_sample) / num_sample


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, text_dim, criterion):
        '''
        embeddingnet: 求图像embedding的网络, 这里是csn_model(ResNet-18
        text_dim: 文字的embedding维度, 6000
        criterion: 损失函数, torch.nn.MarginRankingLoss(margin = args.margin)
        '''
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet  # 求图像embedding的网络, 这里是csn_model(ResNet-18
        self.text_branch = EmbedBranch(text_dim, args.dim_embed)  # 求文字embedding的网络
        self.metric_branch = None
        if args.learned_metric:
            self.metric_branch = nn.Linear(args.dim_embed, 1, bias=False)

            # initilize as having an even weighting across all dimensions
            weight = torch.zeros(1,args.dim_embed)/float(args.dim_embed) # xyy: 这里除法是在干什么?
            self.metric_branch.weight = nn.Parameter(weight)

        self.criterion = criterion
        self.margin = args.margin

    def image_forward(self, x, y, z):
        """ x: Anchor data
            y: Distant (negative) data
            z: Close (positive) data
        """
        # conditions only available on the anchor sample
        c = x.conditions
        '''
        特定条件的嵌入, 特定嵌入结果l2 norm, 通用嵌入结果l2 norm, 通用嵌入得到的结果
        '''
        embedded_x, masknorm_norm_x, embed_norm_x, general_x = self.embeddingnet(x.images, c)
        embedded_y, masknorm_norm_y, embed_norm_y, general_y = self.embeddingnet(y.images, c)
        embedded_z, masknorm_norm_z, embed_norm_z, general_z = self.embeddingnet(z.images, c)
        mask_norm = (masknorm_norm_x + masknorm_norm_y + masknorm_norm_z) / 3
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        loss_embed = embed_norm / np.sqrt(len(x))
        loss_mask = mask_norm / len(x)

        if self.metric_branch is None:
            dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
            dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        else:
            dist_a = self.metric_branch(embedded_x*embedded_y)
            dist_b = self.metric_branch(embedded_x*embedded_z)

        target = torch.FloatTensor(dist_a.size()).fill_(1)  # 全填充1, 为后面方便使用损失函数
        if dist_a.is_cuda:
            target = target.cuda()
        target = Variable(target)

        # type specific triplet loss
        loss_triplet = self.criterion(dist_a, dist_b, target)
        acc = accuracy(dist_a, dist_b) # xyy: 这里可以改用skilearn的

        # calculate image similarity loss on the general embedding
        disti_p = F.pairwise_distance(general_y, general_z, 2)
        disti_n1 = F.pairwise_distance(general_y, general_x, 2)
        disti_n2 = F.pairwise_distance(general_z, general_x, 2)
        loss_sim_i1 = self.criterion(disti_p, disti_n1, target)
        loss_sim_i2 = self.criterion(disti_p, disti_n2, target)
        loss_sim_i = (loss_sim_i1 + loss_sim_i2) / 2.

        return acc, loss_triplet, loss_sim_i, loss_mask, loss_embed, general_x, general_y, general_z

    def text_forward(self, x, y, z):
        """ x: Anchor data
            y: Distant (negative) data
            z: Close (positive) data
        """
        desc_x = self.text_branch(x.text)
        desc_y = self.text_branch(y.text)
        desc_z = self.text_branch(z.text)
        distd_p = F.pairwise_distance(desc_y, desc_z, 2)
        distd_n1 = F.pairwise_distance(desc_x, desc_y, 2)
        distd_n2 = F.pairwise_distance(desc_x, desc_z, 2)
        has_text = x.has_text * y.has_text * z.has_text
        loss_sim_t1 = selective_margin_loss(distd_p, distd_n1, self.margin, has_text)
        loss_sim_t2 = selective_margin_loss(distd_p, distd_n2, self.margin, has_text)
        loss_sim_t = (loss_sim_t1 + loss_sim_t2) / 2.
        return loss_sim_t, desc_x, desc_y, desc_z

    def calc_vse_loss(self, desc_x, general_x, general_y, general_z, has_text):
        """ Both y and z are assumed to be negatives because they are not from the same
            item as x

            desc_x: Anchor language embedding
            general_x: Anchor visual embedding
            general_y: Visual embedding from another item from input triplet
            general_z: Visual embedding from another item from input triplet
            has_text: Binary indicator of whether x had a text description
        """
        distd1_p = F.pairwise_distance(general_x, desc_x, 2) #Anchor的图像embedding和Anchor的文字embedding的欧氏距离
        distd1_n1 = F.pairwise_distance(general_y, desc_x, 2) #Negative的图像embedding和Anchor的文字embedding的欧氏距离
        distd1_n2 = F.pairwise_distance(general_z, desc_x, 2) #Positive的图像embedding和Anchor的文字embedding的欧氏距离
        loss_vse_1 = selective_margin_loss(distd1_p, distd1_n1, self.margin, has_text)
        loss_vse_2 = selective_margin_loss(distd1_p, distd1_n2, self.margin, has_text)
        return (loss_vse_1 + loss_vse_2) / 2.

    def forward(self, x, y, z):
        """ x: Anchor data
            y: Distant (negative) data
            z: Close (positive) data
        """
        acc, loss_triplet, loss_sim_i, loss_mask, loss_embed, general_x, general_y, general_z = self.image_forward(x, y, z)
        loss_sim_t, desc_x, desc_y, desc_z = self.text_forward(x, y, z)
        loss_vse_x = self.calc_vse_loss(desc_x, general_x, general_y, general_z, x.has_text)
        loss_vse_y = self.calc_vse_loss(desc_y, general_y, general_x, general_z, y.has_text)
        loss_vse_z = self.calc_vse_loss(desc_z, general_z, general_x, general_y, z.has_text)
        loss_vse = (loss_vse_x + loss_vse_y + loss_vse_z) / 3.
        return acc, loss_triplet, loss_mask, loss_embed, loss_vse, loss_sim_t, loss_sim_i
