import argparse
from argparse import Namespace
def get_args():
    parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of start epoch (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='Type_Specific_Fashion_Compatibility', type=str,
                        help='name of experiment')
    parser.add_argument('--polyvore_split', default='disjoint', type=str,
                        help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
    parser.add_argument('--datadir', default='/mnt/xujunhao/xyy_siamese_metrix/data', type=str,
                        help='directory of the polyvore outfits dataset (default: data)')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                        help='To only run inference on test set')
    parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                        help='how many dimensions in embedding (default: 64)')
    parser.add_argument('--use_fc', action='store_true', default=False,
                        help='Use a fully connected layer to learn type specific embeddings.')
    parser.add_argument('--learned', dest='learned', action='store_true', default=False,
                        help='To learn masks from random initialization')
    parser.add_argument('--prein', dest='prein', action='store_true', default=False,
                        help='To initialize masks to be disjoint')
    parser.add_argument('--rand_typespaces', action='store_true', default=False,
                        help='randomly assigns comparisons to type-specific embeddings where #comparisons < #embeddings')
    parser.add_argument('--num_rand_embed', type=int, default=4, metavar='N',
                        help='number of random embeddings when rand_typespaces=True')
    parser.add_argument('--l2_embed', dest='l2_embed', action='store_true', default=False,
                        help='L2 normalize the output of the type specific embeddings')
    parser.add_argument('--learned_metric', dest='learned_metric', action='store_true', default=False,
                        help='Learn a distance metric rather than euclidean distance')
    parser.add_argument('--margin', type=float, default=0.3, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--embed_loss', type=float, default=5e-4, metavar='M',
                        help='parameter for loss for embedding norm')
    parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                        help='parameter for loss for mask norm')
    parser.add_argument('--vse_loss', type=float, default=5e-3, metavar='M',
                        help='parameter for loss for the visual-semantic embedding')
    parser.add_argument('--sim_t_loss', type=float, default=5e-5, metavar='M',
                        help='parameter for loss for text-text similarity')
    parser.add_argument('--sim_i_loss', type=float, default=5e-5, metavar='M',
                        help='parameter for loss for image-image similarity')

    args = parser.parse_args()
    return args
