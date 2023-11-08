import os, json
from collections import defaultdict

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



class CocoSceneGraphDataset(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None, fire_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 normalize_images=True, max_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=2, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.
    
        Inputs:
        - image_dir:                Path to a directory where images are held
        - instances_json:           Path to a JSON file giving COCO annotations
        - stuff_json: (optional)    Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size:               Size (H, W) at which to load images. Default (64, 64).
        - mask_size:                Size M for object segmentation masks; default 16.
        - normalize_image:          If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples:              If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships:    If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size:          Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image:    Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image:    Ignore images which have more than this many
          object annotations.
        - include_other:            If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist:       None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist:          None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir =                image_dir
        self.mask_size =                mask_size
        self.max_samples =              max_samples
        self.max_objects_per_image =    max_objects_per_image
        self.normalize_images =         normalize_images
        self.include_relationships =    include_relationships
        self.left_right_flip =          left_right_flip
        
        self.set_image_size(image_size)

        #read instance json
        with open(instances_json, 'r') as f:
            instances_data = json.load(f)
        #read stuff json
        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        #FIRE
        fire_data = None
        if fire_json is not None and fire_json != '':
            with open(fire_json, 'r') as f:
                fire_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        
        for image_data in instances_data['images']:
            #read data
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            #save data
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        #FIRE
        if fire_data:
            for image_data in fire_data['images']:
                image_id = image_data['id']
                filename = image_data['file_name']
                width = image_data['width']
                height = image_data['height']
                self.image_ids.append(image_id)
                self.image_id_to_filename[image_id] = filename
                self.image_id_to_size[image_id] = (width, height)

        #convert obj_name into index
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}

        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        #FIRE
        all_fire_categories = []
        if fire_data:
            for category_data in fire_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_fire_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        #Assign category_list
        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        #FIRE
        # category_whitelist = set(instance_whitelist) | set(stuff_whitelist) 
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)  | set(all_fire_categories)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            # box_area = object_data['area'] / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                # box_area = object_data['area'] / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)
        
        #FIRE
        if fire_data:
            image_ids_with_fire = set()
            for object_data in fire_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_fire.add(image_id)          #additional
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                # box_area = object_data['area'] / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)

        #FIRE
        if stuff_data and fire_data:
            image_ids_with_stuff = image_ids_with_stuff | image_ids_with_fire

        if stuff_data:
            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx


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
                return len(self.image_ids)*2
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

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
        if index >= len(self.image_ids):
            index = index - len(self.image_ids)
            flip = True
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        objs, boxes = [], []
        for object_data in self.image_id_to_objects[image_id]:
            objs.append(object_data['category_id'])
            xm, ym, w, h = object_data['bbox']
            xm = xm / WW
            ym = ym / HH
            w = (w) / WW
            h = (h) / HH
            if flip:
                xm = 1 - (xm + w)
            boxes.append(np.array([xm, ym, w, h]))

        # If less then 8 objects, add 0 class_id and unused bbox
        for _ in range(len(objs), self.max_objects_per_image):
            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))

        objs = torch.LongTensor(objs)
        boxes = np.vstack(boxes)
    
        return image, objs, boxes

