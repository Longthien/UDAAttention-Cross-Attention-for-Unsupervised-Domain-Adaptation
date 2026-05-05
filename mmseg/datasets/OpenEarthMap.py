import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

from mmseg.utils import get_root_logger
from mmcv.utils import print_log

@DATASETS.register_module()
class OpenEarthMapDataset(CustomDataset):
    """OpenEarthMap dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    CLASSES = (
        'Bareland',
        'Grass',
        'Pavement',
        'Road',
        'Tree',
        'Water',
        'Cropland',
        'Buiding',
    )

    PALETTE = [
        [128, 0, 0],     # barren / bareland
        [0, 255, 36],    # grass
        [148, 148, 148], # pavement
        [255, 255, 255], # road
        [34, 97, 38],    # forest / tree
        [0, 69, 255],    # water
        [75, 181, 73],   # agricultural / cropland
        [222, 31, 7],    # building
    ]

    def __init__(self, **kwargs):
        super(OpenEarthMapDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=True,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        assert split, "OpenEarthMap only support for the split file"
        img_infos = []
        with open(split) as f:
            for line in f:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name.replace('images', 'labels') + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos

    def results2img(self, results, imgfile_prefix, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 6.
            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self, results, imgfile_prefix, indices=None):
        """Format the results into dir (standard format for LoveDA evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, indices)

        return result_files
def get_len_ann(infos):
    len_ann = []
    for info in infos:
        with_labels=info["ann"].get('with_labels', False)
        # print_log(ann, logger=get_root_logger())
        if not with_labels:
            continue
        len_ann.append(with_labels)
    return len(len_ann)
@DATASETS.register_module()
class SMOpenEarthMapDataset(OpenEarthMapDataset):
    def __init__(self, **kwargs):
        super(SMOpenEarthMapDataset, self).__init__(**kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        If split is specificed, file with suffix in the splits will be loaded with labels
        otherwise images will be loaded without labels.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        assert split, "OpenEarthMap only support for the split file"
        split_full = split.replace(split.split('/')[-1].split('.')[0], 'train')
        # img_names = []
        # with open(split_full) as f:
        #     for line in f:
        #         img_name = line.strip() + img_suffix
        #         img_names.append(img_name)

        with open(split) as f:
            split=[(line.strip()) for line in f]
        with open(split_full) as f:
            split_full=[(line.strip() + img_suffix) for line in f]
        # print_log(split, logger=get_root_logger())
        img_infos = []

        for img in split_full:
            img_info = dict(filename=img)
            filename = img.split('/')[-1]
            if len(filename.split('_'))==2:
                city = filename.split("_")[0]
            else:
                city = '_'.join(filename.split('_')[:-1])
            filedir = f"{city}/images/{filename}".split('.')[0]
            if filedir in split:
                # print_log('load image with labels', logger=get_root_logger())
                img_info['ann'] = dict(
                    seg_map=img.replace('images', 'labels'),
                    with_labels=True
                    )
                # print_log(f"split in {img_info['ann']} ", logger=get_root_logger())                 
            else:
                img_info['ann'] = dict(
                    seg_map=img.replace('images', 'labels').replace(img_suffix, seg_map_suffix),
                    with_labels=False)
                # print_log(f"split not in {img_info['ann']} ", logger=get_root_logger()) 

            img_infos.append(img_info)
                    
        print_log(
            f"Loaded {len(img_infos)} images from {img_dir}, {get_len_ann(img_infos)} images with labels",
            logger=get_root_logger())
        # print_log(
        #     img_infos, logger=get_root_logger()
        # )
        return img_infos
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        img_info['with_labels'] = ann_info.get('with_labels', False)
        # print_log(img_info, logger=get_root_logger())

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        img_info['with_labels'] = ann_info.get('with_labels', False)
        # print_log(img_info, logger=get_root_logger())

        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)