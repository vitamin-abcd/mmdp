# coding=utf-8

import os
import config as cfg
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pickle


class VocData():
    def __init__(self):
        self.train_pic_list_file = os.path.join(cfg.dataset_dir, 'ImageSets', 'Main', 'train.txt')
        self.image_size = cfg.image_size
        self.cell_num = cfg.cell_num
        self.clz = cfg.clz
        self.cache_dir = cfg.cache_dir
        self.batch_size = cfg.batch_size
        self.epoch = 0
        self.bootstrap()

    def bootstrap(self):
        """

        :return:
        """

        # Get picture name of training set
        with open(self.train_pic_list_file) as f:
            self.image_list = [x.strip() for x in f.readlines()]

        labels = []
        for image_name in self.image_list:
            label = self.get_label(image_name)
            image = self.image_read(image_name)
            labels.append({'name': image_name, 'label': label, 'image': image})

        cache_file = os.path.join(
            self.cache_dir, 'pascal_train_data.pkl')

        # Store label , image data to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(labels, f)

        np.random.shuffle(labels)
        self.labels = labels
        return labels

    def get_next_batch(self):
        """

        :return:
        """
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_num, self.cell_num, 25))
        count = 0
        while count < self.batch_size:
            images[count, :, :, :] = self.labels[self.cursor]['image']
            labels[count, :, :, :] = self.labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.labels):
                np.random.shuffle(self.labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def get_label(self, file_name):
        """

        :return:
        """
        image_file = os.path.join(cfg.dataset_dir, 'JPEGImages', file_name + '.jpg')
        image = cv2.imread(image_file)

        # Calculate the scaling factor between the target image size and the original image size
        width_scale = 1.0 * self.image_size / image.shape[0]
        height_scale = 1.0 * self.image_size / image.shape[1]

        annotation_file = os.path.join(cfg.dataset_dir, 'Annotations', file_name + '.xml')
        label = np.zeros((self.cell_num, self.cell_num, 25))
        dom_tree = ET.parse(annotation_file)

        objs = dom_tree.findall('object')
        for obj in objs:
            bounding_box = obj.find('bndbox')

            x_min = max(min((float(bounding_box.find('xmin').text) - 1) * width_scale, self.image_size - 1), 0)
            y_min = max(min((float(bounding_box.find('ymin').text) - 1) * height_scale, self.image_size - 1), 0)
            x_max = max(min((float(bounding_box.find('xmax').text) - 1) * width_scale, self.image_size - 1), 0)
            y_max = max(min((float(bounding_box.find('ymax').text) - 1) * height_scale, self.image_size - 1), 0)

            clz_index = dict(zip(self.clz, range(len(self.clz))))[obj.find('name').text.lower().strip()]

            # Calculate the center point of x-dim , the center point of y-dim , width , height
            boxes = [(x_max + x_min) / 2.0, (y_max + y_min) / 2.0, x_max - x_min, y_max - y_min]
            # Calculate the target object belong to which grid cell
            x_ind = int(boxes[0] * self.cell_num / self.image_size)
            y_ind = int(boxes[1] * self.cell_num / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + clz_index] = 1

        return label

    def image_read(self, file_name):
        """

        :param file_name:
        :return:
        """
        imname = os.path.join(cfg.dataset_dir, 'JPEGImages', file_name + '.jpg')
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Scale image to 0~1 ; it may speed up
        image = (image / 255.0) * 2.0 - 1.0
        return image
