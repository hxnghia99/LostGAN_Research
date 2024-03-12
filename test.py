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

from utils.util import draw_layout, IS_compute_np, truncted_random, normalize_minmax, combine_images




def main(args):
    #Common
    args.mode = 'train'
    args.G_path = "./outputs/model_test/041_FireGAN_baseline/G_200.pth"
    args.D_path = "./outputs/model_test/041_FireGAN_baseline/D_200.pth"
    img_size = (args.img_size, args.img_size)

    #Special: Test
    max_num_obj = 2                 #if max_obj=2, get only first fire and smoke
    get_first_fire_smoke = True if max_num_obj==2 else False    
    use_bkg_cls = False             #bboxes do not cover whole image --> True: add 1 bkg_cls + bkg_noise_embedding_input as random
    if use_bkg_cls: max_num_obj = 3
    use_enc_feat_as_bkg_cls_noise = False           #transform encoded features using FC to bkg_cls noise input
    use_random_input_noise_w_enc_feat = False       #Later: use random input noise concatenating with enc_feat
    
    z_obj_random_dim = 128
    z_obj_cls_dim = 128
    normalized = True
    z_obj_random_thres=2.0
    args.seg_mask_thresh = 0.5

    phase_testing = True

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
        if use_bkg_cls: num_classes+=1

        train_data = FireDataset(fire_image_dir=val_fire_img_dir, non_fire_image_dir=val_non_fire_img_dir,
                                classname_file=classname_file,
                                image_size=img_size,
                                max_objects_per_image=max_num_obj,
                                get_first_fire_smoke=get_first_fire_smoke,
                                normalize_images=normalized,
                                test=phase_testing)

        with open(os.path.join(dataset_path, "class_names.txt"), "r") as f:
            class_names = f.read().splitlines()


    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, drop_last=True, shuffle=False, num_workers=0)#num_workers=args.num_workers)


    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3, z_obj_random_dim=z_obj_random_dim, z_obj_class_dim=z_obj_cls_dim,
                              enc_feat_as_bkg_noise=use_enc_feat_as_bkg_cls_noise, random_input_noise=use_random_input_noise_w_enc_feat, test=phase_testing).cuda()
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
        [fire_images, non_fire_images, _], label, bbox, weight_map = data
        label, bbox, weight_map = label.long().cuda().unsqueeze(-1), bbox.float(), weight_map.float().cuda()      #keep bbox in cpu --> make input of netG,netD in gpu
        fire_images = fire_images.cuda()
        non_fire_images = non_fire_images.cuda()

        if normalized:
            z_obj = torch.from_numpy(truncted_random(z_obj_dim=z_obj_random_dim, num_o=max_num_obj, thres=z_obj_random_thres, test=phase_testing)).float().cuda()
        else:
            z_obj = torch.rand(fire_images.size(0), max_num_obj, z_obj_random_dim).cuda()
        fake_images, stage_mask128, [bbox_mask64, stage_mask16, stage_mask32, stage_mask64] = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
        
        if not normalized:
            fake_images = fake_images*0.5+0.5
        g_out_fake, _ = netD(fake_images, bbox.cuda(), label)
        g_out_real, _ = netD(fire_images, bbox.cuda(), label)

        # fake_fire_crops = fake_images * weight_map
        
        #1) fake-fire
        if not normalized:
            fake_images = (fake_images-0.5)*2
        fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        fake_images = np.array(fake_images*255, np.uint8)
        g_out_fake  = g_out_fake[0,0].cpu().detach()
        g_out_fake = -1 if g_out_fake<-1 else 1 if g_out_fake>1 else torch.round(g_out_fake,decimals=2)
        fake_images = draw_layout(label, bbox, [256,256], class_names, fake_images, g_out_fake, topleft_name='Fake-fire image')
        #2) real-fire
        if not normalized:
            fire_images = (fire_images-0.5)*2
        fire_images = fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        fire_images = np.array(fire_images*255, np.uint8)
        g_out_real  = g_out_real[0,0].cpu().detach()
        g_out_real = -1 if g_out_real<-1 else 1 if g_out_real>1 else torch.round(g_out_real,decimals=2)
        fire_images = draw_layout(label, bbox, [256,256], class_names, fire_images, g_out_real, topleft_name='Real-fire image')
        #3) non-fire
        if not normalized:
            non_fire_images = (non_fire_images-0.5)*2
        non_fire_images = non_fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        non_fire_images = np.array(non_fire_images*255, np.uint8)
        non_fire_images = draw_layout(label, bbox, [256,256], class_names, non_fire_images, topleft_name='Non-fire image')
        
        #Segmentation mask
        bbox_mask64 = bbox_mask64[0].cpu().detach().numpy()
        stage_mask16 = stage_mask16[0].cpu().detach().numpy()
        stage_mask32 = stage_mask32[0].cpu().detach().numpy()
        stage_mask64 = stage_mask64[0].cpu().detach().numpy()
        stage_mask128 = stage_mask128[0].cpu().detach().numpy()   #shape [3 objs, 128, 128]
        #4) soft-mask
        embed_mask64 = normalize_minmax(np.clip(np.sum(bbox_mask64[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        test_embed_mask64 = draw_layout(label, bbox, [64,64], class_names, input_img=embed_mask64, topleft_name='Test Embed mask 64x64')
        cv2.imshow("Test embed 64x64", cv2.resize(cv2.cvtColor(test_embed_mask64.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
        embed_mask64 = draw_layout(label, bbox, [256,256], class_names, input_img=embed_mask64, topleft_name='Embed mask 64x64')
        
        soft_mask16 = normalize_minmax(np.clip(np.sum(stage_mask16[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        test_soft_mask16 = draw_layout(label, bbox, [16,16], class_names, input_img=soft_mask16, topleft_name='Test Soft 16x16')
        cv2.imshow("Test soft 16x16", cv2.resize(cv2.cvtColor(test_soft_mask16.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
        soft_mask16 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask16, topleft_name='Soft mask 16x16')

        soft_mask32 = normalize_minmax(np.clip(np.sum(stage_mask32[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask32 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask32, topleft_name='Soft mask 32x32')

        soft_mask64 = normalize_minmax(np.clip(np.sum(stage_mask64[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask64 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask64, topleft_name='Soft mask 64x64')

        soft_mask128 = normalize_minmax(np.clip(np.sum(stage_mask128[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask128, topleft_name='Soft mask 128x128')
        
        # #5) hard-mask
        # hard_mask = np.any(stage_mask128[0:2]>args.seg_mask_thresh, axis=0).astype(np.uint8)
        # hard_mask = normalize_minmax(hard_mask, [0, 255], input_range=[0,1])
        # hard_mask = draw_layout(label, bbox, [256,256], class_names, input_img=hard_mask, topleft_name='Hard seg-mask')

        # output_images = combine_images([fire_images, non_fire_images, fake_images, soft_mask128, hard_mask], [256,256])
        
        output_images = combine_images([fire_images, non_fire_images, fake_images, embed_mask64, soft_mask16, soft_mask32, soft_mask64, soft_mask128], [256,256])

        cv2.imshow("Test generating Fire + Mask", cv2.cvtColor(output_images.astype(np.uint8), cv2.COLOR_RGB2BGR))
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
