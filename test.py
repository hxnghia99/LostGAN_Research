import os
import argparse
from collections import OrderedDict
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cocostuff_loader import CocoSceneGraphDataset
from data.data_loader import FireDataset

from model.resnet_generator import ResnetGenerator128

import colorsys


def truncted_random(num_o=8, thres=1.0):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = 0.5#np.random.normal()
    return z


def main(args):
    #Configuration setup
    dataset_path =      os.path.join("./datasets", args.dataset)
    
    img_size = (args.img_size, args.img_size)
    num_o = 8
    
    
    if args.dataset == 'coco':
        train_img_dir =     os.path.join(dataset_path, "val2017")
        instances_json =    os.path.join(dataset_path, "annotations/instances_val2017.json")
        stuff_json =        os.path.join(dataset_path, "annotations/stuff_val2017.json")
        num_classes = 184

        train_data = CocoSceneGraphDataset(image_dir=train_img_dir,
                                       instances_json=instances_json,
                                       stuff_json=stuff_json,
                                       stuff_only=True, image_size=img_size, left_right_flip=False)

        with open("./datasets/coco/labels.txt", "r") as f:
            class_names = f.read().split("\n")[0:-1]
            class_names = [x.split(": ")[1] for x in class_names]

    elif args.dataset == 'fire':
        train_img_dir   = os.path.join(dataset_path, "images")
        classname_file  = os.path.join(dataset_path, "class_names.txt")
        num_classes = 4
        train_data = FireDataset(image_dir=train_img_dir, classname_file=classname_file,
                                image_size=img_size, left_right_flip=True)

        with open("./datasets/fire/class_names.txt", "r") as f:
            class_names = f.read().splitlines()


    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, drop_last=True, shuffle=False, num_workers=0)#num_workers=args.num_workers)


    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError("Not found model on provided path: {}".format(args.model_path))
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres=2.0
    for idx, data in enumerate(dataloader):
        real_images, label, bbox = data

        layout_img = draw_layout(label, bbox, [256,256], class_names)

        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
        fake_images = netG.forward(z_img=z_im, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
        fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        fake_images = np.array(fake_images*255, np.uint8)
        cv2.imshow("Generated", cv2.resize(fake_images, (256, 256)))
        cv2.imshow("Layout", cv2.resize(layout_img, (256, 256)))
        if cv2.waitKey() == 'q':
            pass
        
        # cv2.imwrite("{save_path}/sample_{idx}.jpg".format(save_path=args.sample_path, idx=idx),
        #             fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

        cv2.destroyAllWindows()

def draw_layout(label, bbox, size, class_names):
    temp_img = np.zeros([size[0],size[1],3])
    bbox = (bbox[0]*size[0]).numpy().astype(np.int32)
    label = label[0]
    num_classes = 184

    rectangle_hsv_tuples     = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    label_hsv_tuples         = [(1.0 * x / num_classes, 1., 1.) for x in range(int(num_classes/2), num_classes)] 
    label_hsv_tuples        += [(1.0 * x / num_classes, 1., 1.) for x in range(0, int(num_classes/2))]                                
    rand_rectangle_colors    = list(map(lambda x: colorsys.hsv_to_rgb(*x), rectangle_hsv_tuples))
    rand_rectangle_colors    = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_rectangle_colors))
    rand_text_colors         = list(map(lambda x: colorsys.hsv_to_rgb(*x), label_hsv_tuples))
    rand_text_colors         = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_text_colors))
    
   
    
    for i in range(len(bbox)):
        bbox_color = rand_rectangle_colors[label[i]]
        label_color = rand_text_colors[label[i]]
        x,y,width,height = bbox[i]
        class_name = class_names[label[i]]
        cv2.rectangle(temp_img, (x, y), (x + width, y + height), bbox_color, 1)  # (0, 255, 0) is the color (green), 2 is the thickness
        cv2.putText(temp_img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)

    return temp_img
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fire',
                        help='training dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='test input resolution')
    parser.add_argument('--model_path', type=str, default="./outputs/model/G_85.pth",
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
