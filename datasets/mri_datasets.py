import nibabel as nib
import numpy as np
import os
import os.path as osp
import pandas as pd
import skimage.io as io
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms


class MRIDataset(data.Dataset):
    def __init__(self,
                 root=os.path.join(os.getcwd(), 'data/'),
                 train_mode=True,
                 test_patients=None,
                 train_patients=None,
                 augmentations=None,
                 use_labels=True,
                 im_types=['T1.'],
                 proper_shape=np.array([160, 192, 160]),
                 file_type = 'mha'):

        self.root = root
        self.augmentations = augmentations
        self.use_labels = use_labels
        self.train_mode = train_mode
        self.test_patients = test_patients
        self.train_patients = train_patients
        self.file_type = file_type
        self.im_types = im_types
        self.proper_shape = proper_shape

        self.all_brains = None
        self.all_masks = None
        self.all_items_cropped = None
        self.prepare_dataset()

    def prepare_dataset(self):
        return NotImplementedError

    def __getitem__(self, index):
        pat = self.test_patients
        if self.train_mode:
            pat = self.train_patients

        if self.use_labels:
            X, trg = self.all_items_cropped[pat][index]
        else:
            X = self.all_items_cropped[pat][index]
        X = self.scale(np.array(X), (0, 1))

        if self.augmentations is not None and self.train_mode:
            if self.use_labels:
                X, trg = self.augmentations(X, trg)
                return X, torch.LongTensor(trg)
            else:
                X, _ = self.augmentations(X)
                return X
        else:
            X = torch.from_numpy(X.astype(np.float32))
            return X

    def __len__(self):
        if self.train_mode:
            return len(self.train_patients)
        return len(self.test_patients)

    def get_brain(self, path):
        result = []
        for im in path:
            if self.file_type == 'mha':
                full_head = self.read_mha(im)
            elif self.file_type == 'nii':
                full_head = self.read_nii(im)
            else:
                print('unknown file type')
            result.append(full_head)
        return result

    def read_nii(self, file_path):
        return np.array(nib.load(file_path).get_data())

    def read_mha(self, file_path):
        return io.imread(file_path, plugin='simpleitk')

    def cropp_brain(self, img, segmentation=None, verbose=True):
        curr_img = img[0]
        mask = curr_img > 0
        coords = np.argwhere(mask)
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1
        for i in range(len(img)):
            img[i] = img[i][x0:x1, y0:y1, z0:z1]

        if segmentation is not None:
            mini_segmentation = segmentation[0][x0:x1, y0:y1, z0:z1]

        if np.any(np.array(img[0].shape) > self.proper_shape) and verbose:
            print("Brain does not fit in the proper shape {}".format(
                np.array(img[0].shape)))

        diff = []
        for i in range(3):
            if img[0].shape[i] > self.proper_shape[i]:
                diff.append(img[0].shape[0] - self.proper_shape[0])
            else:
                diff.append(0)
        for i in range(len(img)):
            img[i] = img[i][diff[0] // 2 + diff[0] % 2: img[i].shape[0] - diff[0] // 2,
                     diff[1] // 2 + diff[1] % 2: img[i].shape[1] - diff[1] // 2,
                     diff[2] // 2 + diff[2] % 2: img[i].shape[2] - diff[2] // 2]
        if segmentation is not None:
            mini_segmentation = mini_segmentation[
                                diff[0] // 2 + diff[0] % 2: mini_segmentation.shape[0] -
                                                            diff[0] // 2,
                                diff[1] // 2 + diff[1] % 2: mini_segmentation.shape[1] -
                                                            diff[1] // 2,
                                diff[2] // 2 + diff[2] % 2: mini_segmentation.shape[2] -
                                                            diff[2] // 2]
        for i in range(len(img)):
            img[i] = self.pad_to_size(img[i], self.proper_shape.tolist())
        assert np.all(
            img[0].shape == self.proper_shape), "Did not manage to achieve proper shape"
        if segmentation is not None:
            final_segmentation = self.pad_to_size(mini_segmentation,
                                                  self.proper_shape.tolist())
            return img, final_segmentation
        return img

    def pad_to_size(self, img, size=(512, 512, 512)):
        h, w, c = img.shape
        h_res = ((size[0] - h) // 2, (size[0] - h) // 2 + (size[0] - h) % 2)
        w_res = ((size[1] - w) // 2, (size[1] - w) // 2 + (size[1] - w) % 2)
        c_res = ((size[2] - c) // 2, (size[2] - c) // 2 + (size[2] - c) % 2)
        return np.pad(img, pad_width=[h_res, w_res, c_res], mode='constant')

    def scale(self, x, out_range=(0, 1), axis=None):
        domain = np.min(x, axis), np.max(x, axis)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


class BRATS18(MRIDataset):
    def __init__(self,
                 root=os.path.join(os.getcwd(), 'data/BRATS18/'),
                 train_mode=True,
                 test_patients=None,
                 train_patients=None,
                 augmentations=None,
                 use_labels=True,
                 im_types=['t1.'],
                 proper_shape=np.array([160, 192, 160])):

        super(BRATS18, self).__init__(
            root=root,
            train_mode=train_mode,
            test_patients=test_patients,
            train_patients=train_patients,
            augmentations=augmentations,
            use_labels=use_labels,
            im_types=im_types,
            proper_shape=proper_shape,
            file_type='nii'
        )

    def prepare_dataset(self):
        self.all_patients = [osp.join(path, d) for path, dirs, files in os.walk(root) \
                             for d in dirs if (
                                         (path[-3:] == 'HGG') | (path[-3:] == 'LGG') & (
                                             '.ipy' not in d))]
        if self.test_patients is None:
            self.test_patients = np.random.randint(0, len(self.all_patients), 30)
        if self.train_patients is None:
            self.train_patients = [x for x in range(len(self.all_patients)) if
                                   x not in self.test_patients]

        all_brains = [[osp.join(p, file) for file in os.listdir(p)
                       if self.check_file(file, file)] for p in self.all_patients]
        self.all_brains = list(map(self.get_brain, all_brains))
        if self.use_labels:
            all_masks = [[osp.join(p, file) for file in os.listdir(p) \
                          if self.check_file(file, file, ['seg'])] for p in
                         self.all_patients]
            self.all_masks = np.array(list(map(self.get_brain, all_masks)))
            for i in range(len(self.all_masks)):
                self.all_masks[i][self.all_masks[i] > 0] = 1
            self.all_items_cropped = np.array(
                list(map(self.cropp_brain, self.all_brains, self.all_masks)))
        else:
            self.all_items_cropped = np.array(
                list(map(self.cropp_brain, self.all_brains)))
