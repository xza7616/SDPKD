import os
from PIL import Image
import numpy as np
# from scipy.io import loadmat
# from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from udw.toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale
# from model.toolbox.utils import color_map
from torch import nn
from torch.autograd import Variable as V
import torch as t
class WE3Ds(data.Dataset):

    def __init__(self, cfg, random_state=3, mode='train',):
        assert mode in ['train', 'test']

        #将图片转为张量并归一化
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # self.dp_to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # 将深度图转为张量并归一化
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])#################depth_channel=3输入
        # 将边缘图转为张量并归一化
        self.ed_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 从jiso配置文件中加载数据
        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))
        self.resize = Resize(crop_size)
        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        self.mode = mode
        self.class_weight = np.array([49.1442,  1.4527, 43.0769, 50.4314, 50.4816, 46.6292, 48.1845, 50.2010,
                                      50.4194, 43.3021, 45.0498, 41.8012, 44.4411, 48.1478, 43.8489, 44.0530,
                                      50.4527, 49.9137, 44.7399])
        # self.class_weight = np.array([ 2.9315, 27.0240, 25.5761, 13.9443, 43.0573,  3.5077,  8.9109,  8.2166])
        # self.class_weight = np.array([0.2670, 8.4434, 2.8865, 3.3980, 1.2270, 0.3348, 0.7888, 0.8439])
        # self.class_weight = np.array([2.1623, 25.5761, 43.0573, 3.5077, 8.9109])
        # train_test_split返回切分的数据集train/test(随机划分数据集)
        # self.train_ids, self.test_ids = train_test_split(np.arange(1449), train_size=795, random_state=random_state)

        # train_test_split返回切分的数据集train/test(正常划分数据集)
        # split_filepath = '/media/hjk/HardDisk/XZA/toolbox/datasets/nyudv2_splits.mat'
        # splits = loadmat(split_filepath)
        # self.train_ids = splits['trainNdxs'][:, 0] - 1
        # self.test_ids = splits['testNdxs'][:, 0] - 1
        with open('/media/yuride/date/XZA/WE3DS/train.txt','r') as f:
            self.train_ids = [line.strip() for line in f.readlines()]
        with open('/media/yuride/date/XZA/WE3DS/test.txt','r') as f:
            self.test_ids = [line.strip() for line in f.readlines()]


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, index):
        # key=self.train_ids[index][0]

        if self.mode == 'train':
            image_index = self.train_ids[index]
            gate_gt = torch.zeros(1)
            # gate_gt[0] = key

        else:
            image_index = self.test_ids[index]

        # color_map = {
        #     (0, 0, 0): 0,  # 黑色BW -> 类别 0
        #     (0, 0, 255): 1,  # 蓝色HD -> 类别 1
        #     (0, 255, 0): 2,  # 绿色PF -> 类别 1
        #     (0, 255, 255): 3,  # 天空色WR -> 类别 3
        #     (255, 0, 0): 4,  # 红色RO -> 类别 4
        #     (255, 0, 255): 5,  # 粉色RI -> 类别 5
        #     (255, 255, 0): 6,  # 黄色FV -> 类别 6
        #     (255, 255, 255): 7  # 白色SR -> 类别 7
        # }

        # def color_to_label(mask, color_map):
        #     h, w, _ = mask.shape
        #     label = np.zeros((h, w), dtype=np.int64)
        #     for color, idx in color_map.items():
        #         label[(mask[:, :, 0] == color[0]) &
        #               (mask[:, :, 1] == color[1]) &
        #               (mask[:, :, 2] == color[2])] = idx
        #     return label

        image_path = f'images/img_{image_index}.png'
        depth_path = f'depth_refined/img_{image_index}.png'
        label_path = f'annotations/segmentation/SegmentationLabel/img_{image_index}.png'
        # edge_path = f'edge/{image_index}.png'


        # label_pathcxk = f'all_data/Label/{image_index}.png'
        # label_path = '/home/yangenquan/PycharmProjects/NYUv2/all_data/label/75.png'
        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB')  # 1 channel -> 3
        # depth = Image.open(os.path.join(self.root, depth_path))  # 1 channel -> 1
        label = Image.open(os.path.join(self.root, label_path))    # 1 channel 0~37
        # label = Image.open(os.path.join(self.root, label_path)).convert('RGB')  # 1-> 3 channel 0~37
        # edge = Image.open(os.path.join(self.root, edge_path))
        # labelcxk = Image.open(os.path.join(self.root, label_pathcxk))


        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            # 'edge': edge,
            # 'name' : image_index
            # 'labelcxk':labelcxk,
        }

        if self.mode == 'train':  # 只对训练集增强
            sample = self.aug(sample)
        else:
            sample = self.resize(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        # sample['label'] = color_to_label(sample['label'], color_map)
        # sample['edge'] = self.ed_to_tensor(sample['edge'])
        # sample['labelcxk'] = torch.from_numpy(np.asarray(sample['labelcxk'], dtype=np.int64)).long()
        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        # sample['name'] = image_index
        return sample

    """ for train loader """

    @property
    def cmap(self):
        return [(255, 255, 255),(0,0,0), (0,128,0), (128,128,0),(0,0,128),(128,0,128),(0,128,128),(128,128,128),(64,0,0),
                (192,0,0), (64,128,0), (192,128,0), (64,0,128),(192,0,128),(64,128,128),(192,128,128),(0,64,0),(128,64,0),(0,192,0)]




# if __name__ == '__main__':
#     import json
#
#     path = '../../configs/nyuv2.json'
#     with open(path, 'r') as fp:
#         cfg = json.load(fp)
#
#     dataset = NYUv2(cfg, mode='train')
#     from toolbox.utils import class_to_RGB
#     import matplotlib.pyplot as plt
#
#     for i in range(len(dataset)):
#         sample = dataset[i]
#
#         image = sample['image']
#         depth = sample['depth']
#         label = sample['label']
#
#         image = image.numpy()
#         image = image.transpose((1, 2, 0))
#         image *= np.asarray([0.229, 0.224, 0.225])
#         image += np.asarray([0.485, 0.456, 0.406])
#
#         depth = depth.numpy()
#         depth = depth.transpose((1, 2, 0))
#         depth *= np.asarray([0.226, 0.226, 0.226])
#         depth += np.asarray([0.449, 0.449, 0.449])
#
#         label = label.numpy()
#         label = class_to_RGB(label, N=41, cmap=dataset.cmap)
#
#         plt.subplot('131')  #行，列，那一幅图，如一共1*3图，该行的第一幅图
#         plt.imshow(image)
#         plt.subplot('132')
#         plt.imshow(depth)
#         plt.subplot('133')
#         plt.imshow(label)
#
#         plt.show()
if __name__ == '__main__':
    import json

    path = '/home/yangenquan/PycharmProjects/第一论文模型/(60.1)mymodel8/configs/nyuv2.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    dataset = SUIM(cfg, mode='test')
    print(len(dataset))
    from toolbox.utils import class_to_RGB
    from PIL import Image
    import matplotlib.pyplot as plt

    # label = '/home/yangenquan/PycharmProjects/NYUv2/all_data/label/166.png'
    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        depth = sample['depth']
        label = sample['label']
        name = sample['name']

        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        depth = depth.numpy()
        depth = depth.transpose((1, 2, 0))
        depth *= np.asarray([0.226, 0.226, 0.226])
        depth += np.asarray([0.449, 0.449, 0.449])
        # print(set(list(label)))
        label = label.numpy()
        # print(image)

        label = class_to_RGB(label, N=8, cmap=dataset.cmap)



        # print(dataset.cmap)
        # plt.subplot('131')  #行，列，那一幅图，如一共1*3图，该行的第一幅图
        # plt.imshow(image)
        # plt.subplot('132')
        # plt.imshow(depth)
        # plt.subplot('133')
        # plt.imshow(label)

        # plt.show()
        label = Image.fromarray(label)

        label.save(f'/home/yangenquan/PycharmProjects/NYUv2/all_data/change/label_color/{name}.png')
        # break
