import os, json, glob, random, copy
from itertools import compress

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
import cv2
from utils.util import draw_layout, create_continuous_map


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
                 min_objects_per_image=1, max_objects_per_image=2, left_right_flip=False, get_first_fire_smoke=False,
                 use_noised_input=False, weight_map_type='extreme', test=False):
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
        self.use_noised_input =         use_noised_input
        self.weight_map_type =          weight_map_type
        self.testing_phase =            test
        #Tranformation: Resize, ToTensor, Normalize
        self.set_image_size(image_size)

        with open(classname_file, 'r') as f:
            self.class_names = f.read().splitlines()

        vocal = dict()
        for idx, name in enumerate(self.class_names):
            vocal[name] = idx
        self.vocal = vocal

        fire_image_files = glob.glob(os.path.join(fire_image_dir,"*.jpg"))
        fire_image_files = [x.replace("\\", '/') for x in fire_image_files]
        non_fire_image_files = glob.glob(os.path.join(non_fire_image_dir,"*.jpg")) + glob.glob(os.path.join(non_fire_image_dir,"*.png"))
        non_fire_image_files = [x.replace("\\", '/') for x in non_fire_image_files]
        mode = 'train' if 'train' in fire_image_files[0] else 'val'
        fire_annotation_files = [(x.split(mode+"_images_A")[0] + "annotations" + x.split(mode+"_images_A")[1]).split(".jpg")[0]+".json" for x in fire_image_files]


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
            #keep only the first 'fire' and the the first 'smoke'
            first_obj = False
            secnd_obj = False
            for object in objects:
                _, _, w, h = object['bbox']
                if w*h/(img_w*img_h) > min_object_size and w*h/(img_w*img_h) < max_object_size: #check criterias
                    if get_first_fire_smoke: #check whether get first fire or not
                        if not first_obj and object['class_name']=='fire':
                            first_obj = True
                            new_objects.append(object)
                        elif not secnd_obj and object['class_name']=='smoke':
                            secnd_obj = True
                            new_objects.append(object)
                    else:
                        new_objects.append(object)
            annotation_data['objects'] = new_objects
            #0 objects or >max objects --> remove image
            if len(new_objects)<min_objects_per_image or len(new_objects)>max_objects_per_image:
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
        transform = [T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()]    #toTensor(): Normalize (0, 1)
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
        non_fire_image_file = self.non_fire_image_files[random.randint(0, self.len_non_fire-1) if not self.testing_phase else index % self.len_non_fire]    

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
        fire_annotation_data = copy.deepcopy(self.fire_annotation_datas[index % self.len_fire])
        objects = fire_annotation_data['objects']
        for object_data in objects:
            xm, ym, w, h = object_data['bbox']
            if flip:
                object_data['bbox'] = [WF - (xm + w), ym, w, h]

        # #TESTING
        # fire_box = draw_bbox(fire_image.copy(), objects)
        # fire_box = fire_box.resize((256, 256))
        # fire_box.show()

        #Add noise to non_fire_img at the corresponding positions of fire
        objects_for_non = copy.deepcopy(objects)
        for object_data in objects_for_non:
            xm, ym, w, h = object_data['bbox']
            xm = int(xm * WNF / WF)
            ym = int(ym * HNF / HF)
            w = int((w) * WNF / WF)
            h = int((h) * HNF / HF)
            object_data['bbox'] = xm, ym, w, h
            #create noise
            if self.use_noised_input:
                noise = np.random.randint(0, 2, (h, w, 1), dtype=np.uint8) * 255
                noise = np.repeat(noise, repeats=3, axis=2)
                noise = PIL.Image.fromarray(noise, 'RGB')
            else:
                noise = np.zeros((h, w, 3), dtype=np.uint8)
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

        #make weight for background / fire_region
        if self.weight_map_type == 'extreme':
            weight_map = np.ones((self.max_objects_per_image, self.image_size[0], self.image_size[1]))
            weight_map_64 = np.ones((self.max_objects_per_image, int(self.image_size[0]/2), int(self.image_size[1]/2)))
            for i, box in enumerate(boxes):
                xm, ym, w, h = box
                xmin = np.ceil(xm*self.image_size[0]).astype(np.int32)
                ymin = np.ceil(ym*self.image_size[0]).astype(np.int32)
                xmax = np.floor((xm+w)*self.image_size[0]).astype(np.int32)
                ymax = np.floor((ym+h)*self.image_size[0]).astype(np.int32)
                weight_map[i,ymin:ymax,xmin:xmax] = 0

                xmin = np.ceil(xm*self.image_size[0]/2).astype(np.int32)
                ymin = np.ceil(ym*self.image_size[0]/2).astype(np.int32)
                xmax = np.floor((xm+w)*self.image_size[0]/2).astype(np.int32)
                ymax = np.floor((ym+h)*self.image_size[0]/2).astype(np.int32)
                weight_map_64[i,ymin:ymax,xmin:xmax] = 0

        elif self.weight_map_type == 'continuous':
            weight_map = np.ones((self.max_objects_per_image, self.image_size[0], self.image_size[1]))
            for box in boxes:
                xm, ym, w, h = [round(x*self.image_size[0]) for x in box]
                continuous_map_2d = create_continuous_map(h, w)
                continuous_map_3d = np.repeat(np.expand_dims(continuous_map_2d, axis=0), repeats=3, axis=0)
                weight_map[i,ym:ym+h,xm:xm+w] = continuous_map_3d
        else:
            raise NotImplemented("Not implement the weight map type as ['extreme', 'continuous'] ...")

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

        # #TEST
        # test_weight(list_images[0], weight_map)

        # test_img = test_img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        # test_img = np.array(test_img*255, np.uint8)
        # test_img = draw_layout(classes.unsqueeze(0), torch.FloatTensor(boxes).unsqueeze(0), (480,480), self.class_names, input_img=test_img)
        # cv2.imshow("test img", cv2.cvtColor(test_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # # if index in [2671, 2495, 1062, 2186, 2345, 2538] or index in (np.array([611, 2671, 688, 2495, 1062, 2186, 2345, 2538])+3236):
        # # print(index, flip)
        # if cv2.waitKey() == ord('s'):
        #     print(fire_image_file)

        return list_images, classes, boxes, weight_map, weight_map_64

def test_weight(image, weight):
    image = np.round((np.array(image)*0.5+0.5)*255).astype(np.uint8)
    image = image * weight
    image = PIL.Image.fromarray(np.array(image, np.uint8).transpose(1,2,0))
    image = image.resize((256,256))
    image.show()


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
