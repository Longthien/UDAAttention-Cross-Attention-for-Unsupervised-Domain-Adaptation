import json
import os.path as osp

import mmcv
import numpy as np
import torch

from . import CityscapesDataset
from .builder import DATASETS
from .uda_dataset import get_rcs_class_probs
from mmseg.utils import get_root_logger
from mmcv.utils import print_log

@DATASETS.register_module()
class SSDADataset(object):
    """Dataset containing:
    - source: labeled source domain
    - target_unlabeled: unlabeled target domain
    - target_labeled: labeled target domain (small)
    """

    def __init__(self, source, target_unlabeled, target_labeled, cfg):
        self.source = source
        self.target_unlabeled = target_unlabeled
        self.target_labeled = target_labeled

        # Keep ignore_index, CLASSES, PALETTE aligned
        self.ignore_index = target_unlabeled.ignore_index
        self.CLASSES = target_unlabeled.CLASSES
        self.PALETTE = target_unlabeled.PALETTE

        # print(target_labeled.ignore_index)
        # print(target_unlabeled.ignore_index)
        # print(source.ignore_index)
        assert target_unlabeled.ignore_index == source.ignore_index
        assert target_unlabeled.CLASSES == source.CLASSES
        assert target_unlabeled.PALETTE == source.PALETTE
        assert target_labeled.ignore_index == source.ignore_index
        assert target_labeled.CLASSES == source.CLASSES
        assert target_labeled.PALETTE == source.PALETTE

        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                osp.join(cfg['source']['data_root'], 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0

            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for _ in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]

        i_unlabeled = np.random.choice(range(len(self.target_unlabeled)))
        t_unlabeled = self.target_unlabeled[i_unlabeled]

        i_labeled = np.random.choice(range(len(self.target_labeled)))
        t_labeled = self.target_labeled[i_labeled]

        return {
            **s1,
            'target_unlabeled_img_metas': t_unlabeled['img_metas'],
            'target_unlabeled_img': t_unlabeled['img'],
            # 'target_unlabeled_gt_semantic_seg': t_unlabeled['gt_semantic_seg'],

            'target_labeled_img_metas': t_labeled['img_metas'],
            'target_labeled_img': t_labeled['img'],
            'target_labeled_gt_semantic_seg': t_labeled['gt_semantic_seg'],
        }

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s_source = self.source[idx // len(self.target_unlabeled)]
            t_unlabeled = self.target_unlabeled[idx % len(self.target_unlabeled)]
            t_labeled = self.target_labeled[idx % len(self.target_labeled)]

            return {
                **s_source,
                'target_unlabeled_img_metas': t_unlabeled['img_metas'],
                'target_unlabeled_img': t_unlabeled['img'],
                # 'target_unlabeled_gt_semantic_seg': t_unlabeled['gt_semantic_seg'],

                'target_labeled_img_metas': t_labeled['img_metas'],
                'target_labeled_img': t_labeled['img'],
                'target_labeled_gt_semantic_seg': t_labeled['gt_semantic_seg'],
            }

    def __len__(self):
        return len(self.source) * len(self.target_unlabeled)
