from args import get_args
import torch
import numpy as np
import os
import json
import pickle
from PIL import Image
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

args = get_args()


def load_typespaces(rootdir, rand_typespaces, num_rand_embed):
    '''
        生成类别对, 对每个类别-类别的pair进行从0开始的编号
    '''
    # 'data/polyvore_outfits/disjoint/typespaces.p'
    typespace_fn = os.path.join(rootdir, 'typespaces.p')

    # 读取"类别对类别"的pair的list
    typespaces = pickle.load(open(typespace_fn, 'rb'))

    ts = {}
    for index, t in enumerate(typespaces):
        ts[t] = index # 从0开始给类别对编个号

    typespaces = ts
    return typespaces

def load_compatibility_questions(fn, im2index, id2im):
    """
        返回一个list, 包含了一大堆分数预测问题, 以编号形式表达
    """
    with open(fn, 'r') as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        # 搭配分数预测任务:222049137_1 222049137_2 222049137_3 222049137_4 222049137_5 222049137_6
        compat_question = []
        for index, im_id in enumerate(data[1:]): #从第一个之后开始读, 第一个是0/1, 即分数
            im = id2im[im_id]
            compat_question.append((im2index[im], im))
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions

def load_fitb_questions(fn, im2index, id2im):
    """
        返回一个list, 包含了一大堆填空题问题, 以编号形式表达
    """
    data = json.load(open(fn, 'r'))
    questions = []
    for item in data:
        # FITB任务: "question": ["222049137_1", "222049137_2", "222049137_3", "222049137_4", "222049137_5"]
        question = item['question'] # 加载每个问题
        question_idx = []
        ground_truth = None # 记录真实outfit_id
        for index, im_id in enumerate(question):
            ground_truth = im_id.split('_')[0]
            im = id2im[im_id]
            question_idx.append((im2index[im], im)) # 问题原文的items

        answer = item['answers'] # 加载问题的选项
        answers_idx = []
        is_correct = np.zeros(len(answer), np.bool)
        for index, im_id in enumerate(answer):
            set_id = im_id.split('_')[0] # 获取选项的outfit
            im = id2im[im_id]
            answers_idx.append((im2index[im], im)) # 选项的items
            is_correct[index] = set_id == ground_truth # 设置每个选项的正确性
        questions.append((question_idx, answers_idx, is_correct))

    return questions

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, meta_data, text_dim=None, transform=None, loader=default_image_loader):
        '''
            split: 选择是用哪一个数据的子集, 即训练集还是测试机
            meta_data: 读取"polyvore_item_metadata.json"内的数据, 元数据里有item的描述, 类别
            text_dim: 文本embedding维度, 6000
            transform: 用来给图片做变化
            loader: # 读取图片的模组
        '''
        # 拼接数据路径
        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        # 设置图片路径
        self.impath = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        # 判断是否在训练
        self.is_train = split == 'train'
        # 读取对应split的数据, 如test.json
        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r')) # outfit数据, <outfit_id, [item_ids]>

        # get list of images and make a mapping used to quickly organize the data
        # 读取图片列表, 并且重新映射
        im2type = {} # 由item名字查询其类别的字典
        category2ims = {} # 由类别查询其下所有item的字典, 二维, <类别, outfit名>
        imnames = set() # 记录所有item名字的集合
        id2im = {} # 由outfit名和底下item的次序来查询item名字的字典
        for outfit in outfit_data:
            # 获取outfit名字
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id'] # 获取item的名字
                category = meta_data[im]['semantic_category'] # 查询语义层面的类别, 如top
                im2type[im] = category # 记录这个item的类别, 保存为字典

                if category not in category2ims:
                    category2ims[category] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(im) # 把item归到<类别, outfit名>字典底下
                id2im['%s_%i' % (outfit_id, item['index'])] = im # 由outfit名和底下item的次序来记录item名字
                imnames.add(im) # 记录所有item名字的集合

        imnames = list(imnames)
        im2index = {} # 将item从0开始编号, 可以由item编号查询item的名字
        for index, im in enumerate(imnames):
            im2index[im] = index # 更新编号

        self.data = outfit_data # outfit数据, <outfit_id, [item_ids]>
        self.imnames = imnames # 记录所有item名字的集合
        self.im2type = im2type # 由item名字查询其类别的字典
        self.typespaces = load_typespaces(rootdir, args.rand_typespaces, args.num_rand_embed)
        self.transform = transform # 用来给图片做变化
        self.loader = loader # 读取图片的模组
        self.split = split # 选择是用哪一个数据的子集, 即训练集还是测试集
        self.category2ims = category2ims

        if self.is_train:
            self.text_feat_dim = text_dim # 文本嵌入维度, 6000
            self.desc2vecs = {}  # 文本转向量字典
            # 读取文本描述的embedding,6000dim
            featfile = os.path.join(rootdir, 'train_hglmm_pca6000.txt')

            with open(featfile, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    vec = line.split(',')
                    label = ','.join(vec[:-self.text_feat_dim]) # 提取纯文本信息
                    # vec: (6000,)
                    vec = np.array([float(x) for x in vec[-self.text_feat_dim:]], np.float32)
                    assert(len(vec) == text_dim)
                    self.desc2vecs[label] = vec # 文本转向量字典

            self.im2desc = {} # 用图片的名字查找文本向量, 注意是向量
            for im in imnames:
                desc = meta_data[im]['title']
                if not desc:
                    desc = meta_data[im]['url_name']

                desc = desc.replace('\n', '').encode('ascii', 'ignore').strip().lower()

                # sometimes descriptions didn't map to any known words so they were
                # removed, so only add those which have a valid feature representation
                if desc and desc in self.desc2vecs:
                    self.im2desc[im] = desc

            # At train time we pull the list of outfits and enumerate the pairwise
            # comparisons between them to train with.  Negatives are pulled by the
            # __get_item__ functi47on
            pos_pairs = []
            max_items = 0
            for outfit in outfit_data:
                items = outfit['items']
                item_id=[]
                for item in items:
                    item_id.append(item['item_id'])
                cnt = len(items)
                max_items = max(cnt, max_items)
                outfit_id = outfit['set_id']
                for j in range(cnt-1):
                    for k in range(j+1, cnt):
                        # 记录组合三元组 <outfit_id, item的名字, 另一个item的名字>
                        pos_pairs.append((outfit_id, item_id[j], item_id[k]))

            self.pos_pairs = pos_pairs
            self.max_items = max_items # 记录单outfit的最大item数目
        else: # 如果不在训练状态
            # pull the two task's questions for test and val splits
            fn = os.path.join(rootdir, 'fill_in_blank_%s.json' % split)
            # 加载填空问题
            self.fitb_questions = load_fitb_questions(fn, im2index, id2im)
            fn = os.path.join(rootdir, 'compatibility_%s.txt' % split)
            # 加载分数预测问题
            self.compatibility_questions = load_compatibility_questions(fn, im2index, id2im)

    def load_train_item(self, image_id):
        '''
        传入图片的名字, 返回图片和文本
        '''
        imfn = os.path.join(self.impath, '%s.jpg' % image_id)
        img = self.loader(imfn)
        if self.transform is not None:
            img = self.transform(img)
        if image_id in self.im2desc:
            text = self.im2desc[image_id]
            text_features = self.desc2vecs[text] # 获取文本embedding
            has_text = 1
        else:
            text_features = np.zeros(self.text_feat_dim, np.float32)
            has_text = 0.
        has_text = np.float32(has_text) # 0/1是否有文本
        item_type = self.im2type[image_id] # 获取item的类型
        return img, text_features, has_text, item_type

    def sample_negative(self, item_id, item_type):
        '''
            给定item名字和item的类型, 去找同类, 但又不能是自己
        '''
        item_out = item_id
        candidate_sets = self.category2ims[item_type].keys()
        attempts = 0
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(list(candidate_sets))
            items = self.category2ims[item_type][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1
        return item_out

    def get_typespace(self, anchor, pair):
        '''
         给定两个item类型, 找到他们两个类型对的编号
        '''
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)

        return self.typespaces[query]

    def test_compatibility(self, embeds, metric):
        """ Returns the area under a roc curve for the compatibility
            task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        scores = []
        labels = np.zeros(len(self.compatibility_questions), np.int32)
        for index, (outfit, label) in enumerate(self.compatibility_questions):
            labels[index] = label
            n_items = len(outfit)
            outfit_score = 0.0
            num_comparisons = 0.0
            for i in range(n_items - 1):
                item1, img1 = outfit[i]
                type1 = self.im2type[img1]
                for j in range(i + 1, n_items):
                    item2, img2 = outfit[j]
                    type2 = self.im2type[img2]
                    condition = self.get_typespace(type1, type2)
                    embed1 = embeds[item1][condition].unsqueeze(0)
                    embed2 = embeds[item2][condition].unsqueeze(0)
                    if metric is None:
                        outfit_score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)
                    else:
                        outfit_score += metric(Variable(embed1 * embed2)).data

                    num_comparisons += 1.

            outfit_score /= num_comparisons
            scores.append(outfit_score)

        scores = torch.cat(scores).squeeze().cpu().numpy()
        # scores = np.load('feats.npy')
        # print(scores)
        # assert(False)
        # np.save('feats.npy', scores)
        auc = roc_auc_score(labels, 1 - scores)
        return auc

    def test_fitb(self, embeds, metric):
        """ Returns the accuracy of the fill in the blank task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        correct = 0.
        n_questions = 0.
        for q_index, (questions, answers, is_correct) in enumerate(self.fitb_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)
            for index, (answer, img1) in enumerate(answers):
                type1 = self.im2type[img1]
                score = 0.0
                for question, img2 in questions:
                    type2 = self.im2type[img2]
                    condition = self.get_typespace(type1, type2)
                    embed1 = embeds[question][condition].unsqueeze(0)
                    embed2 = embeds[answer][condition].unsqueeze(0)
                    if metric is None:
                        score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)
                    else:
                        score += metric(Variable(embed1 * embed2)).data

                answer_score[index] = score.squeeze().cpu().numpy()

            correct += is_correct[np.argmin(answer_score)]
            n_questions += 1

        # scores are based on distances so need to convert them so higher is better
        acc = correct / n_questions
        return acc

    def __getitem__(self, index):
        if self.is_train:
            outfit_id, anchor_im, pos_im = self.pos_pairs[index]
            # 获取anchor
            img1, desc1, has_text1, anchor_type = self.load_train_item(anchor_im)
            # 获取positive
            img2, desc2, has_text2, item_type = self.load_train_item(pos_im)
            # 获取negative, 先采样获得负例item的名字, 再获取它的数据
            neg_im = self.sample_negative(pos_im, item_type)
            if neg_im is None:
                print(123)
            img3, desc3, has_text3, _ = self.load_train_item(neg_im)
            # 获得类型对编号
            condition = self.get_typespace(anchor_type, item_type)
            return img1, desc1, has_text1, img2, desc2, has_text2, img3, desc3, has_text3, condition
        else:
            # 如果不是读训练集, 而是读测试集
            anchor = self.imnames[index]
            img1 = self.loader(os.path.join(self.impath, '%s.jpg' % anchor))
            if self.transform is not None:
                img1 = self.transform(img1)

            return img1

    def __len__(self):
        if self.is_train:
            return len(self.pos_pairs)
        return len(self.imnames)