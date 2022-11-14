import torch

import numpy as np
from args import get_args
from torch.autograd import Variable
args = get_args()

class TypeSpecificNet(torch.nn.Module):
    def __init__(self, embeddingnet, n_conditions):
        '''
            embeddingnet: 用来将输入投影成embedding的网络, 这里是ResNet18
            n_conditions: 类别对组合的数量
        '''
        super(TypeSpecificNet, self).__init__()

        if args.rand_typespaces:
            n_conditions = int(np.ceil(n_conditions / float(args.num_rand_embed)))

        # 参数是否学习
        self.learnedmask  = args.learned
        self.embeddingnet = embeddingnet
        self.resnet_linear = torch.nn.Linear(1000,args.dim_embed)
        # 确定参数的初始化方式
        prein = args.prein

        # 当fc_masks的值为True时, 使用全连接作为映射
        self.fc_masks = args.use_fc

        # 如果为真, 我们使用l2正则化处理type specific embeddings
        self.l2_norm = args.l2_embed

        if self.fc_masks:
            masks = []
            # 每种情况定义一种MLP
            self.masks = torch.nn.ModuleList()
            for i in range(n_conditions):
                masks.append(torch.nn.Linear(args.dim_embed, args.dim_embed))
        else:
            # create the mask
            if self.learnedmask:
                # 更新参数
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, args.dim_embed)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7)  # 0.1, 0.005
            else:
                # 不更新
                # define masks
                self.masks = torch.nn.Embedding(n_conditions, args.dim_embed)
                # initialize masks
                mask_array = np.zeros([n_conditions, args.dim_embed])
                mask_len = int(args.dim_embed / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c=None):
        '''
        x: 输入的图像数据
        c: 标记, 如果为None, 则拼接上general embedinng
            如果不为NOne, 则返回类别通用的嵌入
        '''
        embedded_x = self.embeddingnet(x)
        embedded_x = self.resnet_linear(embedded_x)
        if c is None:
            # used during testing, wants all type specific embeddings returned for an image
            if self.fc_masks:
                masked_embedding = []
                for mask in self.masks:
                    masked_embedding.append(mask(embedded_x).unsqueeze(1))

                masked_embedding = torch.cat(masked_embedding, 1)
                embedded_x = embedded_x.unsqueeze(1)
            else:
                masks = Variable(self.masks.weight.data)
                # masks = torch.Tensor(self.masks.weight.data.cpu()) # xyy:这里原来是Variable, 不知道可不可以这样改
                # print(".cpu() OK!")
                masks = masks.unsqueeze(0).repeat(embedded_x.size(0), 1, 1)
                embedded_x = embedded_x.unsqueeze(1)
                masked_embedding = embedded_x.expand_as(masks) * masks

            if self.l2_norm:
                norm = torch.norm(masked_embedding, p=2, dim=2) + 1e-10
                masked_embedding = masked_embedding / norm.expand_as(masked_embedding)

            return torch.cat((masked_embedding, embedded_x), 1)
        else:
            if self.fc_masks:
                mask_norm = 0.
                masked_embedding = []
                for embed, condition in zip(embedded_x, c):
                    mask = self.masks[condition]
                    masked_embedding.append(mask(embed.unsqueeze(0)))
                    mask_norm += mask.weight.norm(1)

                masked_embedding = torch.cat(masked_embedding)
            else:
                self.mask = self.masks(c)
                if self.learnedmask:
                    self.mask = torch.nn.functional.relu(self.mask)

                masked_embedding = embedded_x * self.mask
                mask_norm = self.mask.norm(1)

            embed_norm = embedded_x.norm(2)
            if self.l2_norm:
                norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
                masked_embedding = masked_embedding / norm.expand_as(masked_embedding)

            return masked_embedding, mask_norm, embed_norm, embedded_x
