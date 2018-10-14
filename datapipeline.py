import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.contrib.data import Iterator
from tensorflow import convert_to_tensor
from tensorflow.python.framework import dtypes
from tensorflow.contrib.data import Dataset

class DataPipeline:
    def __init__(self, json_file, mode, batch_size, shuffle):
        self.JSON_FILE = json_file
        self.IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

        self.read_json()
        print 'json read'
        
        self.data_size = len(self.img_paths)
    
        if shuffle:
            self._shuffle_lists()
        
        print 'Converting to tensor..'
        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths)
        self.gt_attr1 = convert_to_tensor(self.gt_attr1)
        self.gt_attr2 = convert_to_tensor(self.gt_attr2)
        self.gt_attr3 = convert_to_tensor(self.gt_attr3)
        
        # create dataset
        print 'creating dataset'
        data = Dataset.from_tensor_slices((self.img_paths, self.gt_attr1, self.gt_attr2, self.gt_attr3))

        print 'created..'
        
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
            data.prefetch(100*batch_size)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)
            data.prefetch(100*batch_size)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        print 'data mapped'
        # shuffle the first `buffer_size` elements of the dataset
        #if shuffle:
            #data = data.shuffle(buffer_size=buffer_size)

        print 'creating with batches'
        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data
    
    def read_json(self):
        self.PP_DICT = json.load(open(self.JSON_FILE))
        self.img_paths = []
        self.gt_attr1 = []
        self.gt_attr2 = []
        self.gt_attr3 = []
        for img in self.PP_DICT:
            self.img_paths.append(img)
            attr_info = self.PP_DICT[img]
            for key in attr_info:
                if key == '1':
                    self.gt_attr1.append(attr_info[key])
                elif key == '2':
                    self.gt_attr2.append(attr_info[key])
                else:
                    self.gt_attr3.append(attr_info[key])
    
    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        gt_attr1 = self.gt_attr1
        gt_attr2 = self.gt_attr2
        gt_attr3 = self.gt_attr3
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.gt_attr1 = []
        self.gt_attr2 = []
        self.gt_attr3 = []
        
        for i in permutation:
            self.img_paths.append(path[i])
            self.gt_attr1.append(gt_attr1[i])
            self.gt_attr2.append(gt_attr2[i])
            self.gt_attr3.append(gt_attr3[i])
    
    def _parse_function_train(self,filename, gt_attr1, gt_attr2, gt_attr3):
        """Input parser for samples of the training set."""
        
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, self.IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_centered, gt_attr1, gt_attr2, gt_attr3

    def _parse_function_inference(self, filename, gt_attr1, gt_attr2, gt_attr3):
        
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        
        img_centered = tf.subtract(img_resized, self.IMAGENET_MEAN)
        
        img_bgr = img_centered[:, :, ::-1]
        
        return img_centered, gt_attr1, gt_attr2, gt_attr3



