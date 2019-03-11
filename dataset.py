import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.jpg')]
    return [
        (os.path.join(root, 'image', img_name + '.jpg'), os.path.join(root, 'mask', img_name + '.png'))
        for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if img.size != target.size:
            print(self.imgs[index])
        return img, target

    def __len__(self):
        return len(self.imgs)


# def make_dataset(root):
#     img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.jpg')]
#     return [
#         (os.path.join(root, 'image', img_name + '.jpg'), os.path.join(root, 'mask', img_name + '.png'),
#          os.path.join(root, 'edge', img_name + '.png'))
#         for img_name in img_list]
#
#
# class ImageFolder(data.Dataset):
#     def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None):
#         self.root = root
#         self.imgs = make_dataset(root)
#         self.joint_transform = joint_transform
#         self.img_transform = img_transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         img_path, gt_path, edge_path = self.imgs[index]
#         img = Image.open(img_path).convert('RGB')
#         target = Image.open(gt_path)
#         edge = Image.open(edge_path)
#         if self.joint_transform is not None:
#             img, target, edge = self.joint_transform(img, target, edge)
#         if self.img_transform is not None:
#             img = self.img_transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#             edge = self.target_transform(edge)
#
#         return img, target, edge
#
#     def __len__(self):
#         return len(self.imgs)
