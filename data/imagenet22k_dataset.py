import json
import os
import warnings

import numpy as np
import torch.utils.data as data
from PIL import Image

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class IN22KDATASET(data.Dataset):
    def __init__(self, root, ann_file="", transform=None, target_transform=None):
        super(IN22KDATASET, self).__init__()

        self.data_path = root
        self.ann_path = os.path.join(self.data_path, ann_file)
        self.transform = transform
        self.target_transform = target_transform
        # id & label: https://github.com/google-research/big_transfer/issues/7
        # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        self.database = json.load(open(self.ann_path))

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.database[index]

        # images
        images = self._load_image(self.data_path + "/" + idb[0]).convert("RGB")
        if self.transform is not None:
            images = self.transform(images)

        # target
        target = int(idb[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target

    def __len__(self):
        return len(self.database)
