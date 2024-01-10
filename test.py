import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import cv2

import torch
from data.cocostuff_loader import CocoSceneGraphDataset
from data.data_loader import FireDataset

from model.resnet_generator import ResnetGenerator128
from model.rcnn_discriminator import CombineDiscriminator128

from utils.util import draw_layout, IS_compute_np, truncted_random




def normalize_minmax(img, targe_range):
    target_min, target_max = targe_range
    current_min, current_max = img.min(), img.max()
    return ((img-current_min)/(current_max-current_min))*target_max + target_min

def main(args):
    #Common
    args.mode = 'train'
    args.G_path = "./outputs/model_test/HXNGHIA3/G_200.pth"
    args.D_path = "./outputs/model_test/HXNGHIA3/D_200.pth"
    img_size = (args.img_size, args.img_size)

    #Special: Test
    use_noised_input = False
    max_num_obj = 2                 #if max_obj=2, get only first fire and smoke
    get_first_fire_smoke = True if max_num_obj==2 else False
    z_obj_random_dim = 32
    z_obj_cls_dim = 32
    normalized = False
    z_obj_random_thres=2.0

    #Training Initilization
    save_results = False
    dataset_path =      os.path.join("./datasets", args.dataset)
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

    elif 'fire' in args.dataset:
        val_fire_img_dir   = os.path.join(dataset_path, args.mode+"_images_A")
        val_non_fire_img_dir   = os.path.join(dataset_path, args.mode+"_images_B")
        classname_file  = os.path.join(dataset_path, "class_names.txt")
        num_classes = 3
        train_data = FireDataset(fire_image_dir=val_fire_img_dir, non_fire_image_dir=val_non_fire_img_dir,
                                classname_file=classname_file,
                                image_size=img_size, left_right_flip=False,
                                use_noised_input=use_noised_input,
                                max_objects_per_image=max_num_obj,
                                get_first_fire_smoke=get_first_fire_smoke,
                                normalize_images=normalized)

        with open(os.path.join(dataset_path, "class_names.txt"), "r") as f:
            class_names = f.read().splitlines()


    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, drop_last=True, shuffle=False, num_workers=0)#num_workers=args.num_workers)


    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3, z_obj_random_dim=z_obj_random_dim, z_obj_class_dim=z_obj_cls_dim, normalized_data=normalized).cuda()
    netD = CombineDiscriminator128(num_classes=num_classes).cuda()

    if not os.path.isfile(args.G_path):
        raise FileNotFoundError("Not found model on provided path: {}".format(args.G_path))
    
    state_dict = torch.load(args.G_path)
    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    state_dict = torch.load(args.D_path)
    model_dict = netD.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netD.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()
    netD.cuda()
    netD.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    if save_results:
        id_img = 0
    for idx, data in enumerate(dataloader):
        [fire_images, non_fire_images, non_fire_crops], label, bbox, weight_map = data
        label, bbox, weight_map = label.long().cuda().unsqueeze(-1), bbox.float(), weight_map.float().cuda()      #keep bbox in cpu --> make input of netG,netD in gpu
        fire_images = fire_images.cuda()
        non_fire_images = non_fire_images.cuda()
        non_fire_crops = non_fire_crops.cuda()

        if normalized:
            z_obj = torch.from_numpy(truncted_random(z_obj_dim=z_obj_random_dim, num_o=max_num_obj, thres=z_obj_random_thres, test=True)).float().cuda()
        else:
            z_obj = torch.rand(fire_images.size(0), max_num_obj, z_obj_random_dim).cuda()
        fake_images, stage_masks = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
        if not normalized:
            fake_images = fake_images*0.5+0.5
        g_out_fake, _ = netD(fake_images, bbox.cuda(), label)
        g_out_real, _ = netD(fire_images, bbox.cuda(), label)

        # fake_fire_crops = fake_images * weight_map
        
        if not normalized:
            fake_images = (fake_images-0.5)*2
        fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        fake_images = np.array(fake_images*255, np.uint8)
        g_out_fake  = g_out_fake[0,0].cpu().detach()
        g_out_fake = -1 if g_out_fake<-1 else 1 if g_out_fake>1 else torch.round(g_out_fake,decimals=2)
        fake_images = draw_layout(label, bbox, [256,256], class_names, input_img=fake_images, D_class_score=g_out_fake)

        if not normalized:
            fire_images = (fire_images-0.5)*2
        fire_images = fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        fire_images = np.array(fire_images*255, np.uint8)
        g_out_real  = g_out_real[0,0].cpu().detach()
        g_out_real = -1 if g_out_real<-1 else 1 if g_out_real>1 else torch.round(g_out_real,decimals=2)
        fire_images = draw_layout(label, bbox, [256,256], class_names, input_img=fire_images, D_class_score=g_out_real)

        if not normalized:
            non_fire_images = (non_fire_images-0.5)*2
        non_fire_images = non_fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        non_fire_images = np.array(non_fire_images*255, np.uint8)
        non_fire_images = draw_layout(label, bbox, [256,256], class_names, input_img=non_fire_images)


        none_mask   = normalize_minmax(stage_masks[0][0:1].cpu().detach().numpy().transpose(1,2,0), [0, 255])
        none_mask   = draw_layout(label, bbox, [256,256], class_names, input_img=none_mask)
        fire_mask   = normalize_minmax(stage_masks[0][1:2].cpu().detach().numpy().transpose(1,2,0), [0, 255])
        fire_mask   = draw_layout(label, bbox, [256,256], class_names, input_img=fire_mask)
        smoke_mask  = normalize_minmax(stage_masks[0][2:3].cpu().detach().numpy().transpose(1,2,0), [0, 255])
        smoke_mask  = draw_layout(label, bbox, [256,256], class_names, input_img=smoke_mask)
        # bkg_mask    = normalize_minmax(stage_masks[0][3:4].cpu().detach().numpy().transpose(1,2,0), [0, 255])
        # bkg_mask    = draw_layout(label, bbox, [256,256], class_names, input_img=bkg_mask)

        cv2.imshow("None mask", cv2.resize(none_mask.astype(np.uint8), (256, 256)))
        cv2.imshow("Fire mask", cv2.resize(fire_mask.astype(np.uint8), (256, 256)))
        cv2.imshow("Smoke mask", cv2.resize(smoke_mask.astype(np.uint8), (256, 256)))
        # cv2.imshow("Background mask", cv2.resize(bkg_mask.astype(np.uint8), (256, 256)))

        # non_fire_crops = non_fire_crops[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        # non_fire_crops = np.array(non_fire_crops*255, np.uint8)
        # non_fire_crops = draw_layout(label, bbox, [256,256], class_names, input_img=non_fire_crops)

        # fake_fire_crops = fake_fire_crops[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        # weight_map = weight_map[0].cpu().detach().numpy().transpose(1, 2, 0)
        # fake_fire_crops = np.array(fake_fire_crops*255, np.uint8) * weight_map
        # fake_fire_crops = draw_layout(label, bbox, [256,256], class_names, input_img=fake_fire_crops)

        
        cv2.imshow("Non-fire", cv2.resize(cv2.cvtColor(non_fire_images.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
        cv2.imshow("Fake-fire", cv2.resize(cv2.cvtColor(fake_images.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
        cv2.imshow("Fire", cv2.resize(cv2.cvtColor(fire_images.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))

        # cv2.imshow("Non-fire-cropped", cv2.resize(cv2.cvtColor(non_fire_crops.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
        # cv2.imshow("Fake-fire-cropped", cv2.resize(cv2.cvtColor(fake_fire_crops.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
        if cv2.waitKey() == 'q':
            pass
        
        # from pytorch_gan_metrics import get_inception_score
        # import torchvision.transforms as T
        # toten = T.ToTensor()
        # a = toten(non_fire_images[25:25+256, 25:25+256].astype(np.uint8))
        # b = toten(fake_images[25:25+256, 25:25+256].astype(np.uint8))
        # c = toten(fire_images[25:25+256, 25:25+256].astype(np.uint8))
        # d = torch.concat([a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)], axis=0)
        # get_inception_score(d)
        
        if save_results:
            if np.mean(fake_images)>40:
                id_img+=1
            
                if id_img > 300:
                    continue
                elif id_img<10:
                    name = '00'+str(id_img)+'_'
                elif id_img<100:
                    name = '0'+str(id_img)+'_'
                else:
                    name = str(id_img)+'_'
                cv2.imwrite("./image_results/01_Fire_LostGAN/"+name+"input.png", cv2.cvtColor(np.array(non_fire_images, dtype=np.uint8), cv2.COLOR_RGB2BGR ))
                cv2.imwrite("./image_results/01_Fire_LostGAN/"+name+"generated_image.png", cv2.resize(cv2.cvtColor(fake_images, cv2.COLOR_RGB2BGR), (256, 256)))
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str,   default="train",             help="processing phase: train, val")
    parser.add_argument('--dataset',        type=str,   default='fire3',              help='training dataset')
    parser.add_argument('--img_size',       type=int,   default=128,                help='test input resolution')
    parser.add_argument('--G_path',     type=str,   default="./outputs/model_test/HXNGHIA3/G_200.pth",
                                                                                   help='which epoch to load')
    parser.add_argument('--sample_path',    type=str,   default='samples',          help='path to save generated images')
    args = parser.parse_args()
    main(args)
