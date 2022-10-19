# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import io
import os
import zipfile

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return ".zip@" in img_or_path


class ZipReader(object):
    """A class to read zipped files"""

    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, "r")
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index("@")
        assert pos_at != -1, (
            "character '@' is not found from the given path '%s'" % path
        )

        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1 :]
        folder_path = str.strip(folder_path, "/")
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, "/")
            if (
                file_foler_name.startswith(folder_path)
                and len(os.path.splitext(file_foler_name)[-1]) == 0
                and file_foler_name != folder_path
            ):
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1 :])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        if extension is None:
            extension = [".*"]
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, "/")
            if (
                file_foler_name.startswith(folder_path)
                and str.lower(os.path.splitext(file_foler_name)[-1]) in extension
            ):
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1 :])

        return file_lists

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

    @staticmethod
    def imread(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        try:
            im = Image.open(io.BytesIO(data))
        except:
            print("ERROR IMG LOADED: ", path_img)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im
