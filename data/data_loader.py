import os, json, glob, random
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
    def __init__(self, fire_image_dir, non_fire_image_dir, classname_file, image_size=(64, 64),
                 normalize_images=True, max_samples=None, min_object_size=0.02, max_object_size = 0.8,
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
        
        #Tranformation: Resize, ToTensor, Normalize
        self.set_image_size(image_size)

        with open(classname_file, 'r') as f:
            class_names = f.read().splitlines()

        vocal = dict()
        for idx, name in enumerate(class_names):
            vocal[name] = idx
        self.vocal = vocal

        fire_image_files = glob.glob(os.path.join(fire_image_dir,"*.jpg"))
        non_fire_image_files = glob.glob(os.path.join(non_fire_image_dir,"*.jpg")) + glob.glob(os.path.join(non_fire_image_dir,"*.png"))
        
        fire_annotation_files = [(os.path.join(x.split("\\fire_images\\")[0], "annotations", x.split("\\fire_images\\")[1])).split(".jpg")[0]+".json" for x in fire_image_files]


        #Filter out objects that have size less than min_object_size and images with higher number of max_num_objects_per_image
        filtered_annotation_flag = [True] * len(fire_annotation_files)
        annotation_datas = []
        for idx, annotation_file in enumerate(fire_annotation_files):
            with open(annotation_file, 'r') as jsonf:
                annotation_data = json.load(jsonf)
            
            #filtered out objects from min-max size
            img_h, img_w = annotation_data["image_size"]
            objects = annotation_data["objects"]
            new_objects = []
            for object in objects:
                _, _, w, h = object['bbox']
                if w*h/(img_w*img_h) > min_object_size and w*h/(img_w*img_h) < max_object_size:
                    new_objects.append(object)
            annotation_data['objects'] = new_objects

            #0 objects or >8 objects --> remove image
            if len(new_objects)==0 or len(new_objects)>max_objects_per_image:
                filtered_annotation_flag[idx] = False
            else:    
                annotation_datas.append(annotation_data)
        
        self.fire_annotation_datas = annotation_datas
        self.fire_image_files = list(compress(fire_image_files,filtered_annotation_flag))
        self.non_fire_image_files = non_fire_image_files
        self.len_fire = len(self.fire_image_files)
        self.len_non_fire = len(self.non_fire_image_files)

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
        len_max = max(self.len_fire, self.len_non_fire)
        if self.max_samples is None:
            if self.left_right_flip:
                return len_max*2
            return len_max
        return min(len_max, self.max_samples)

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
        if index >= max(self.len_fire, self.len_non_fire):
            index = index - max(self.len_fire, self.len_non_fire)
            flip = True
        
        fire_image_file     = self.fire_image_files[index % self.len_fire]
        non_fire_image_file = self.non_fire_image_files[random.randint(0, self.len_non_fire-1)]    

        #Read image
        with open(fire_image_file, 'rb') as f:
            with PIL.Image.open(f).convert("RGB") as fire_image:
                if flip:
                    fire_image = PIL.ImageOps.mirror(fire_image)
                WF, HF = fire_image.size
        with open(non_fire_image_file, 'rb') as f:
            with PIL.Image.open(f).convert("RGB") as non_fire_image:
                if random.random() > 0.5:
                    non_fire_image = PIL.ImageOps.mirror(non_fire_image)
                WNF, HNF = non_fire_image.size
                non_fire_crop = non_fire_image.copy()

        #Read annotations: 2 classes [fire, smoke]
        fire_annotation_data = self.fire_annotation_datas[index % self.len_fire]
        objects = fire_annotation_data['objects']
        for object_data in objects:
            xm, ym, w, h = object_data['bbox']
            if flip:
                object_data['bbox'] = [WF - (xm + w), ym, w, h]

        # #TESTING
        # fire_box = draw_bbox(fire_image.copy(), objects)
        # fire_box.show()

        #Add noise to non_fire_img at the corresponding positions of fire
        objects_for_non = objects.copy()
        for object_data in objects_for_non:
            xm, ym, w, h = object_data['bbox']
            xm = int(xm * WNF / WF)
            ym = int(ym * HNF / HF)
            w = int((w) * WNF / WF)
            h = int((h) * HNF / HF)
            object_data['bbox'] = xm, ym, w, h
            #create noise
            noise = np.random.randint(0, 2, (h, w, 1), dtype=np.uint8) * 255
            noise = np.repeat(noise, repeats=3, axis=2)
            noise = PIL.Image.fromarray(noise, 'RGB')
            #Add noise to image
            non_fire_crop.paste(noise, (xm, ym))

        # #TESTING
        # non_fire_box = draw_bbox(non_fire_image.copy(), objects)
        # non_fire_box.show()

        #Assign values to boxes, label for training, testing
        classes, boxes = [], []
        for object_data in objects:
            classes.append(object_data['class_id'])
            xm, ym, w, h = object_data['bbox']
            xm = xm / WF
            ym = ym / HF
            w = (w) / WF
            h = (h) / HF
            boxes.append(np.array([xm, ym, w, h]))

        # If less then 8 objects, add 0 class_id and unused bbox --> then add the background as 8th object
        for idx in range(len(objects), self.max_objects_per_image):
            # if idx+1 == self.max_objects_per_image:
            #     classes.append(self.vocal['background'])
            #     boxes.append(np.array([0.0, 0.0, 1.0, 1.0]))
            # else:    
            #     classes.append(self.vocal['_None_'])
            #     boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))
            classes.append(self.vocal['_None_'])
            boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))

        classes = torch.LongTensor(classes)
        boxes = np.vstack(boxes)
        list_images = [self.transform(x) for x in [fire_image, non_fire_image, non_fire_crop]] #The list [fire_image, non_fire_image, non_fire_crop]

        return list_images, classes, boxes


def draw_bbox(image, bboxes):
    #Initlization
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype('arial.ttf', 16)
    
    for bbox in bboxes:
        class_name = 'fire' if bbox['class_id'] == 1 else 'smoke'
        xmin, ymin, w, h = bbox['bbox']
        xmax, ymax = xmin + w, ymin + h
        #draw bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255,0,0) if class_name=='fire' else (0,0,255), width=3)
        #draw class_name
        text_x = xmin
        text_y = ymin - 20
        draw.text((text_x, text_y), class_name, fill=(255,0,0) if class_name=='fire' else (0,0,255), font=font)
    
    return image