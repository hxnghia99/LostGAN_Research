import os, json, glob
from itertools import compress

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL

import pycocotools.mask as mask_utils

#Constant Parameters
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]



class FireDataset(Dataset):
    def __init__(self, image_dir, classname_file, image_size=(64, 64),
                 normalize_images=True, max_samples=None, min_object_size=0.02,
                 min_objects_per_image=1, max_objects_per_image=8, left_right_flip=False):
        """
        A PyTorch Dataset for loading self-built fire dataset
    
        Inputs:
        - image_dir:                Path to a directory where images are held
        - image_size:               Size (H, W) at which to load images. Default (64, 64).
        - normalize_image:          If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples:              If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - min_object_size:          Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image:    Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image:    Ignore images which have more than this many
          object annotations.
        """
        super(Dataset, self).__init__()

        self.max_samples =              max_samples
        self.max_objects_per_image =    max_objects_per_image
        self.normalize_images =         normalize_images
        self.left_right_flip =          left_right_flip
        
        self.set_image_size(image_size)

        with open(classname_file, 'r') as f:
            class_names = f.read().splitlines()

        vocal = dict()
        for idx, name in enumerate(class_names):
            vocal[name] = idx
        self.vocal = vocal

        image_files = glob.glob(os.path.join(image_dir,"*.jpg"))
        annotation_files = [(os.path.join(x.split("\\fire_images\\")[0], "annotations", x.split("\\fire_images\\")[1])).split(".jpg")[0]+".json" for x in image_files]

        #Filter out objects that have size less than min_object_size and images with higher number of max_num_objects_per_image
        filtered_annotation_flag = [True] * len(annotation_files)
        annotation_datas = []
        for idx, annotation_file in enumerate(annotation_files):
            with open(annotation_file, 'r') as jsonf:
                annotation_data = json.load(jsonf)
            
            img_h, img_w = annotation_data["image_size"]
            objects = annotation_data["objects"]
            new_objects = []
            for object in objects:
                _, _, w, h = object['bbox']
                if w*h/(img_w*img_h) > min_object_size:
                    new_objects.append(object)
            annotation_data['objects'] = new_objects

            #0 objects or >8 objects --> remove image
            if len(new_objects)==0 or len(new_objects)>max_objects_per_image:
                filtered_annotation_flag[idx] = False
            else:    
                annotation_datas.append(annotation_data)
        
        self.annotation_datas = annotation_datas
        self.image_files = list(compress(image_files,filtered_annotation_flag))


    #Setup the input resolution: Resize --> ToTensor --> Normalize (optional)
    def set_image_size(self, image_size):
        print('The program called set_image_size() : ', image_size)
        transform = [T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()]
        if self.normalize_images:
            transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = T.Compose(transform)       #used in __getitem__()
        self.image_size = image_size


    #Length of dataset: number of images
    def __len__(self):
        if self.max_samples is None:
            if self.left_right_flip:
                return len(self.image_files)*2
            return len(self.image_files)
        return min(len(self.image_files), self.max_samples)

    #Function for DataLoader
    def __getitem__(self, index):
        """
        Get the image and bbox information. We assume that the image will have 
        height H, width W, C channels; there will be O object annotations, 
        each of which will have both a bounding box with class.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,) --> class_id
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (xm, ym, w, h) format, in a [0, 1] coordinate system
        """
        flip = False
        if index >= len(self.image_files):
            index = index - len(self.image_files)
            flip = True
        
        image_file      = self.image_files[index]

        #Read image
        with open(image_file, 'rb') as f:
            with PIL.Image.open(f).convert("RGB") as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                

        #Read annotation
        annotation_data = self.annotation_datas[index]
        objects = annotation_data['objects']

        classes, boxes = [], []

        
        for object_data in objects:
            classes.append(object_data['class_id'])
            xm, ym, w, h = object_data['bbox']
            xm = xm / WW
            ym = ym / HH
            w = (w) / WW
            h = (h) / HH
            if flip:
                xm = 1 - (xm + w)
            boxes.append(np.array([xm, ym, w, h]))

        # used_boxes = np.array(boxes)
        # used_boxes[:][0::2] = used_boxes[:][0::2] * WW
        # used_boxes[:][1::2] = used_boxes[:][1::2] * HH
        # used_boxes = np.array(used_boxes, dtype=np.int32)
        # used_image = image.copy()
        # for i, box in enumerate(used_boxes):
        #     w, h = box[2:4]
        #     noise = np.array(np.random.rand(h,w) * 255, dtype=np.uint8)
        #     noise = np.expand_dims(noise, axis=2)


        # If less then 8 objects, add 0 class_id and unused bbox --> then add the background as 8th object
        for idx in range(len(objects), self.max_objects_per_image):
            if idx+1 == self.max_objects_per_image:
                classes.append(self.vocal['background'])
                boxes.append(np.array([0.0, 0.0, 1.0, 1.0]))
            else:    
                classes.append(self.vocal['_None_'])
                boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))
            # classes.append(self.vocal['_None_'])
            # boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))

        classes = torch.LongTensor(classes)
        boxes = np.vstack(boxes)
        image = self.transform(image.convert('RGB'))
        # print(image.shape, classes.shape, boxes.shape)

        return image, classes, boxes

