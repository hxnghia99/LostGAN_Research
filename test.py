import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import argparse
import numpy as np
import cv2

import torch
from data.cocostuff_loader import CocoSceneGraphDataset
from data.data_loader import FireDataset

from model.resnet_generator import ResnetGenerator128
from model.rcnn_discriminator import CombineDiscriminator128

from utils.util import draw_layout, IS_compute_np, truncted_random, normalize_minmax, combine_images

import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table


from thop import profile


def main(args):
    #Common
    args.mode = 'train'
    args.G_path = "./outputs/model_test/077_FireGAN_best/G_200.pth"
    args.D_path = "./outputs/model_test/077_FireGAN_best/D_200.pth"
    img_size = (args.img_size, args.img_size)

    #Special: Test
    max_num_obj = 2                 #if max_obj=2, get only first fire and smoke
    use_bkg_cls = True             #bboxes do not cover whole image --> True: add 1 bkg_cls + bkg_noise_embedding_input as random
    bkg_bbox_cover_whole = True

    if use_bkg_cls and not bkg_bbox_cover_whole:
        max_num_obj *= 2
    elif use_bkg_cls and bkg_bbox_cover_whole:
        max_num_obj += 1
    
    use_res11 = False                               #use residual block 11
    z_obj_random_dim = 128
    z_obj_cls_dim = 128
    z_obj_random_thres=2.0
    args.seg_mask_thresh = 0.20
    use_random_input_noise_w_enc_feat = False       #Later: use random input noise concatenating with enc_feat

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
                                test=phase_testing,
                                left_right_flip=True)

        with open(os.path.join(dataset_path, "class_names.txt"), "r") as f:
            class_names = f.read().splitlines()


    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, drop_last=True, shuffle=False, num_workers=0)#num_workers=args.num_workers)


    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3, z_obj_random_dim=z_obj_random_dim, z_obj_class_dim=z_obj_cls_dim,
                              random_input_noise=use_random_input_noise_w_enc_feat, test=phase_testing, use_res11=use_res11).cuda()
    netD = CombineDiscriminator128(num_classes=num_classes).cuda()

    if not os.path.isfile(args.G_path):
        raise FileNotFoundError("Not found model on provided path: {}".format(args.G_path))
    
    state_dict = torch.load(args.G_path)
    model_dict = netG.state_dict()
    assert len(state_dict) == len(model_dict), f"The weight file are different from the G model: {len(state_dict)} != {len(model_dict)}"
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # model_dict['final.2.weight_orig'] = torch.randn(model_dict['final.2.weight_orig'].shape) 
    # model_dict['res11.conv2.weight_orig'] = torch.randn(model_dict['res11.conv2.weight_orig'].shape) 

    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    state_dict = torch.load(args.D_path)
    model_dict = netD.state_dict()
    assert len(state_dict) == len(model_dict), "The weight file are different from the D model"
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
        list_train_txt = []

    
    time_proc = []


    for idx, data in enumerate(dataloader):
        sys.stdout.write("\r Processing time for each image: {:.4f} sec".format(np.mean(time_proc)))
        
        idx += 689
        
        [fire_images, non_fire_images], label, bbox, weight_map_orig = data
        
        if label[0,1] != 2:
            continue
        
        # bbox[0,1] = torch.tensor([0.4413, 0.0159, 0.2016, 0.4858])
        # # label[0][0] = 1
        # label[0][1] = 2
        # # bbox[0,0] = torch.tensor([0.418, 0.202, 0.265, 0.465])
        # bbox[0,1] = torch.tensor([0.58, 0.101, 0.258, 0.352])

        # # bbox[0,0] = torch.tensor([0.0, 0.513, 0.499, 0.335])
        # # bbox[0,1] = torch.tensor([0.301, 0.323, 0.327, 0.180])

        # bbox[0,0] = torch.tensor([0.459, 0.511, 0.481, 0.245])
        # bbox[0,1] = torch.tensor([0.241, 0.016, 0.701, 0.885])

        # if bbox[0,0][2]*bbox[0,0][3] > 0.5 or label[0,1] != 2:
        #     continue

        # input_img = (1 - torch.all(weight_map, dim=1, keepdim=True).expand(fire_images.shape).type(torch.cuda.FloatTensor))*2-1
        
        # temp_img = ((input_img+1)/2).cpu().numpy()[0].transpose(1, 2, 0)
        # cv2.imshow("test bw image", cv2.resize(temp_img.astype(np.uint8)*255, (256,256)))
        

        fire_images, non_fire_images = fire_images.cuda(), non_fire_images.cuda()
        label, bbox = label.long().cuda().unsqueeze(-1), bbox.float()    #keep bbox in cpu --> make input of netG,netD in gpu
        weight_map_orig = weight_map_orig.float().cuda()

        weight_map_fire = torch.all(weight_map_orig[:,:2], dim=1, keepdim=True).expand(fire_images.shape).type(torch.cuda.IntTensor)

        z_obj = torch.from_numpy(truncted_random(z_obj_dim=z_obj_random_dim, num_o=max_num_obj, thres=z_obj_random_thres, test=False)).float().cuda()
 

        flops, params = profile(netG, inputs=(non_fire_images,z_obj, bbox.cuda(), label.squeeze(dim=-1)), verbose=False)
        print(f"Total GFLOPs: {params/1e6}")


        time_start = time.time_ns()
        #Forward()
        fake_images, stage_mask128, [bbox_mask64, stage_mask16, stage_mask32, stage_mask64] = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
        time_proc.append((time.time_ns() - time_start) / 1e6)


        # flops_G = FlopCountAnalysis(netG, (non_fire_images, z_obj, bbox.cuda(), label.squeeze(dim=-1)))
        # print(f"\nFLOPs for current forward pass: {flops_G.total():,} FLOPs")
        # if use_bkg_cls:
        #     z_obj[:,0,:] = 0.0  #make bkg_cls_nosie = 0.0 -> no effect of input noise
        #     z_obj[:,2,:] = 0.0

        # continue
        # z_obj = torch.zeros_like(z_obj)
        # fake_images, stage_mask128, [bbox_mask64, stage_mask16, stage_mask32, stage_mask64] = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128

        g_out_fake, _ = netD(fake_images, None, None)
        g_out_real, _ = netD(fire_images, None, None)

        # fake_fire_crops = fake_images * weight_map
        
        #1) fake-fire
        fake_images = (fake_images*0.5+0.5)[0].cpu().detach().numpy().transpose(1, 2, 0)
        fake_images = np.array(fake_images*255, np.uint8)
        saved_fake_images = fake_images.copy()
        # if save_results: 
        saved_weight_map = weight_map_fire[0].cpu().detach().numpy().transpose(1, 2, 0)
        g_out_fake  = g_out_fake[0,0].cpu().detach()
        g_out_fake = -1 if g_out_fake<-1 else 1 if g_out_fake>1 else torch.round(g_out_fake,decimals=2)
        fake_images = draw_layout(label, bbox, [256,256], class_names, fake_images, g_out_fake, topleft_name='Fake-fire image')
        

        # fire_images_2 = (non_fire_images)[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        # fire_images_2 = np.array(fire_images_2*255, np.uint8)
        # fire_images_2 = draw_layout(label, bbox, [256,256], class_names, fire_images_2, topleft_name='Real-fire')
        # cv2.imshow("Real-fire", cv2.resize(cv2.cvtColor(fire_images_2.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))

        # fire_images_2 = (non_fire_images*(1-weight_map_2))[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        # fire_images_2 = np.array(fire_images_2*255, np.uint8)
        # fire_images_2 = draw_layout(label, bbox, [256,256], class_names, fire_images_2, topleft_name='Real-fire')
        # cv2.imshow("Real-fire2", cv2.resize(cv2.cvtColor(fire_images_2.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))

        #2) real-fire
        fire_images = fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        fire_images = np.array(fire_images*255, np.uint8)
        g_out_real  = g_out_real[0,0].cpu().detach()
        g_out_real = -1 if g_out_real<-1 else 1 if g_out_real>1 else torch.round(g_out_real,decimals=2)
        fire_images = draw_layout(label, bbox, [256,256], class_names, fire_images, g_out_real, topleft_name='Real-fire image')
        #3) non-fire
        non_fire_images = non_fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
        non_fire_images = np.array(non_fire_images*255, np.uint8)
        # if save_results: saved_non_images = non_fire_images.copy()
        saved_non_images = non_fire_images.copy()
        non_fire_images_no_box = draw_layout(torch.tensor([[0]]),torch.tensor([[[-0.6, -0.6, 0.5, 0.5]]]), [256,256], class_names, saved_non_images, topleft_name='Non-fire image no box')
        non_fire_images = draw_layout(label, bbox, [256,256], class_names, non_fire_images, topleft_name='Non-fire image')
        
        #Segmentation mask
        bbox_mask64 = bbox_mask64[0].cpu().detach().numpy()
        stage_mask16 = stage_mask16[0].cpu().detach().numpy()
        stage_mask32 = stage_mask32[0].cpu().detach().numpy()
        stage_mask64 = stage_mask64[0].cpu().detach().numpy()
        stage_mask128 = stage_mask128[0].cpu().detach().numpy()   #shape [3 objs, 128, 128]

        #4) soft-mask
        embed_mask64 = normalize_minmax(np.clip(np.sum(bbox_mask64 if not use_bkg_cls else bbox_mask64[0:3:2] if (use_bkg_cls and max_num_obj==4) else bbox_mask64[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        embed_mask64 = draw_layout(label, bbox, [256,256], class_names, input_img=embed_mask64, topleft_name='Embed mask 64x64')
        
        soft_mask16 = normalize_minmax(np.clip(np.sum(stage_mask16 if not use_bkg_cls else stage_mask16[0:3:2] if (use_bkg_cls and max_num_obj==4) else stage_mask16[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask16 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask16, topleft_name='Soft mask 16x16')

        soft_mask32 = normalize_minmax(np.clip(np.sum(stage_mask32 if not use_bkg_cls else stage_mask32[0:3:2] if (use_bkg_cls and max_num_obj==4) else stage_mask32[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask32 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask32, topleft_name='Soft mask 32x32')

        soft_mask64 = normalize_minmax(np.clip(np.sum(stage_mask64 if not use_bkg_cls else stage_mask64[0:3:2] if (use_bkg_cls and max_num_obj==4) else stage_mask64[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask64 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask64, topleft_name='Soft mask 64x64')

        soft_mask128 = normalize_minmax(np.clip(np.sum(stage_mask128 if not use_bkg_cls else stage_mask128[0:3:2] if (use_bkg_cls and max_num_obj==4) else stage_mask128[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
        soft_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask128, topleft_name='Soft mask 128x128')
    
        if use_bkg_cls:
            if max_num_obj==4:
                fire_bin_mask128 = normalize_minmax(np.argmax((1-weight_map_orig.cpu().numpy()[0,0:1])*np.concatenate([stage_mask128[2:3],stage_mask128[0:1]], axis=0), axis=0), [0,255], [0,1])
                fire_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=fire_bin_mask128, topleft_name='Mask Fire 128')
                if label[0,2,0] == 2:
                    smoke_bin_mask128 = normalize_minmax(np.argmax((1-weight_map_orig.cpu().numpy()[0,1:2])*np.concatenate([stage_mask128[3:4],1-stage_mask128[2:3]], axis=0), axis=0), [0,255],[0,1])
                else:
                    smoke_bin_mask128 = np.zeros_like(fire_bin_mask128)
                smoke_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=smoke_bin_mask128, topleft_name='Mask Smoke 128')

                #5) background soft-mask
                embed_mask64_bkg = normalize_minmax(np.clip(np.sum(bbox_mask64 if not use_bkg_cls else bbox_mask64[1:4:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                embed_mask64_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=embed_mask64_bkg, topleft_name='Embed mask bkg 64x64')
                
                soft_mask16_bkg = normalize_minmax(np.clip(np.sum(stage_mask16 if not use_bkg_cls else stage_mask16[1:4:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask16_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask16_bkg, topleft_name='Soft mask bkg 16x16')

                soft_mask32_bkg = normalize_minmax(np.clip(np.sum(stage_mask32 if not use_bkg_cls else stage_mask32[1:4:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask32_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask32_bkg, topleft_name='Soft mask 32x32')

                soft_mask64_bkg = normalize_minmax(np.clip(np.sum(stage_mask64 if not use_bkg_cls else stage_mask64[1:4:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask64_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask64_bkg, topleft_name='Soft mask 64x64')

                soft_mask128_bkg = normalize_minmax(np.clip(np.sum(stage_mask128 if not use_bkg_cls else stage_mask128[1:4:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask128_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask128_bkg, topleft_name='Soft mask 128x128')

            elif max_num_obj==3:
                fire_bin_mask128 = normalize_minmax(np.argmax((1-weight_map_orig.cpu().numpy()[0,0:1])*np.concatenate([stage_mask128[2:3],stage_mask128[0:1]], axis=0), axis=0), [0,255],[0,1])
                
                # fire_bin_mask128_2 = normalize_minmax(np.argmax(np.concatenate([stage_mask128[2:3],stage_mask128[1:2]], axis=0), axis=0), [0,255],[0,1])
                # fire_bin_mask128 = fire_bin_mask128 + fire_bin_mask128_2

                total_points = int(np.sum((1-weight_map_orig.cpu().numpy()[0,0:1])))
                fire_points = np.sum(np.argmax((1-weight_map_orig.cpu().numpy()[0,0:1])*np.concatenate([stage_mask128[2:3],stage_mask128[0:1]], axis=0), axis=0))
                
                fire_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=fire_bin_mask128, topleft_name='Mask Fire 128')
                if label[0,1,0] == 2:
                    smoke_bin_mask128 = normalize_minmax(np.argmax((1-weight_map_orig.cpu().numpy()[0,1:2])*np.concatenate([stage_mask128[2:3]*0.5,stage_mask128[1:2]], axis=0), axis=0), [0,255],[0,1])
                else:
                    smoke_bin_mask128 = np.zeros_like(fire_bin_mask128)
                smoke_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=smoke_bin_mask128, topleft_name='Mask Smoke 128')
                
                #5) background soft-mask
                embed_mask64_bkg = normalize_minmax(np.clip(np.sum(bbox_mask64 if not use_bkg_cls else bbox_mask64[2:3], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                embed_mask64_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=embed_mask64_bkg, topleft_name='Embed mask bkg 64x64')
                
                soft_mask16_bkg = normalize_minmax(np.clip(np.sum(stage_mask16 if not use_bkg_cls else stage_mask16[2:3], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask16_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask16_bkg, topleft_name='Soft mask bkg 16x16')

                soft_mask32_bkg = normalize_minmax(np.clip(np.sum(stage_mask32 if not use_bkg_cls else stage_mask32[2:3], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask32_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask32_bkg, topleft_name='Soft mask 32x32')

                soft_mask64_bkg = normalize_minmax(np.clip(np.sum(stage_mask64 if not use_bkg_cls else stage_mask64[2:3], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask64_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask64_bkg, topleft_name='Soft mask 64x64')

                soft_mask128_bkg = normalize_minmax(np.clip(np.sum(stage_mask128 if not use_bkg_cls else stage_mask128[2:3], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
                soft_mask128_bkg = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask128_bkg, topleft_name='Soft mask 128x128')

        else:
            fire_hard_mask = np.any(stage_mask128[0:1]>args.seg_mask_thresh, axis=0).astype(np.uint8)
            fire_hard_mask = normalize_minmax(fire_hard_mask, [0, 255], input_range=[0,1])
            fire_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=fire_hard_mask, topleft_name='Hard Fire Seg-mask')
            if label[0,1,0] == 2:
                smoke_hard_mask = np.any(stage_mask128[1:2]>args.seg_mask_thresh, axis=0).astype(np.uint8)
                smoke_hard_mask = normalize_minmax(smoke_hard_mask, [0, 255], input_range=[0,1])
            else:
                smoke_hard_mask = np.zeros_like(fire_hard_mask)
            smoke_mask128 = draw_layout(label, bbox, [256,256], class_names, input_img=smoke_hard_mask, topleft_name='Hard Fire Seg-mask')

        
        if fire_points/total_points > 0.3:
            if use_bkg_cls:
                output_images = combine_images([fire_images, non_fire_images, fake_images, fire_mask128, smoke_mask128,
                                                embed_mask64, soft_mask16, soft_mask32, soft_mask64, soft_mask128, 
                                                embed_mask64_bkg, soft_mask16_bkg, soft_mask32_bkg, soft_mask64_bkg, soft_mask128_bkg], [256,256])
            else:
                output_images = combine_images([fire_images, non_fire_images, fake_images, fire_mask128, smoke_mask128,
                                                    embed_mask64, soft_mask16, soft_mask32, soft_mask64, soft_mask128], [256,256])

            
            if not save_results:
                cv2.imshow("Test generating Fire + Mask", cv2.cvtColor(output_images.astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imshow("Test1", cv2.cvtColor(non_fire_images[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imshow("Test2", cv2.cvtColor(fake_images[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                
                #image for paper
                layout = np.zeros((256, 256, 3), np.uint8) + 200
                layout = draw_layout(label, bbox, [256,256], class_names, layout, layout_size=0.8)
                cv2.imshow("Test layout", cv2.cvtColor(layout[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                fire_mask = draw_layout(label[:,0::2,:], bbox[:,0::2,:], [256,256], class_names, input_img=fire_bin_mask128)
                cv2.imshow("Test fire mask", cv2.cvtColor(fire_mask[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                
                if cv2.waitKey() == ord('s'):
                    # cv2.imwrite("./outputs/Non_fire.png", cv2.cvtColor(non_fire_images[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                    # cv2.imwrite("./outputs/Fake_fire.png", cv2.cvtColor(fake_images[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    print(bbox)
                    cv2.imwrite("./outputs/paper_imgs/fake_fire_{}.png".format(idx), cv2.cvtColor(fake_images[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("./outputs/paper_imgs/fake_fire_no_box_{}.png".format(idx), cv2.resize(cv2.cvtColor(saved_fake_images, cv2.COLOR_RGB2BGR), (256, 256)))
                    cv2.imwrite("./outputs/paper_imgs/fire_mask_{}.png".format(idx), cv2.resize(cv2.cvtColor(np.repeat(np.expand_dims(fire_bin_mask128, axis=2), axis=2, repeats=3).astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
                    
                    cv2.imwrite("./outputs/paper_imgs/non_fire_{}.png".format(idx), cv2.cvtColor(non_fire_images_no_box[26:256+25,26:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("./outputs/paper_imgs/fire_layout_{}.png".format(idx), cv2.cvtColor(layout[5:256+25,25:256+25].astype(np.uint8), cv2.COLOR_RGB2BGR))

                    cv2.imwrite("./outputs/paper_imgs/fake_fire_no_box_cropped_{}.png".format(idx), cv2.resize(cv2.cvtColor((saved_fake_images*saved_weight_map + 255*(1-saved_weight_map)).astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
                    saved_weight_map = cv2.resize(saved_weight_map, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite("./outputs/paper_imgs/non_fire_cropped_{}.png".format(idx), cv2.cvtColor((non_fire_images_no_box[26:256+26,26:256+26]*saved_weight_map + 255*(1-saved_weight_map)).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pass
        
        # from pytorch_gan_metrics import get_inception_score
        # import torchvision.transforms as T
        # toten = T.ToTensor()
        # a = toten(non_fire_images[25:25+256, 25:25+256].astype(np.uint8))
        # b = toten(fake_images[25:25+256, 25:25+256].astype(np.uint8))
        # c = toten(fire_images[25:25+256, 25:25+256].astype(np.uint8))
        # d = torch.concat([a.unsqueeze(0),b.unsqueeze(0),c.unsqueeze(0)], axis=0)
        # get_inception_score(d)
        
        # if save_results:
        #     r = (saved_fake_images*(1-saved_weight_map))[:,:,0].sum() / ((1-saved_weight_map)[:,:,0].sum())
        #     g = (saved_fake_images*(1-saved_weight_map))[:,:,1].sum() / ((1-saved_weight_map)[:,:,1].sum())
        #     b = (saved_fake_images*(1-saved_weight_map))[:,:,2].sum() / ((1-saved_weight_map)[:,:,2].sum())
            
            # if save_results and ((2*r-g-b)/r)>0.4:
            if save_results:
                id_img+=1
                #Gen image names
                if id_img<10: name = 'fire_0000'+str(id_img) + "_rgb.png"
                elif id_img<100: name = 'fire_000'+str(id_img) + "_rgb.png"
                elif id_img<1000: name = 'fire_00'+str(id_img) + "_rgb.png"
                elif id_img<10000: name = 'fire_0'+str(id_img) + "_rgb.png"
                else: name = 'fire_'+str(id_img) + "_rgb.png"
                list_train_txt.append("./images/train/"+name)
                # save fire images
                cv2.imwrite("./dataset_det_seg/images/"+name, cv2.resize(cv2.cvtColor(saved_fake_images, cv2.COLOR_RGB2BGR), (256, 256)))

                #Saving det labels
                label_bboxes = []
                for id, cls in enumerate(label[0]):
                    #convert xymin_wh -> xywh
                    if cls[0] == 1 or cls[0] == 2:
                        tmp = bbox[0][id].tolist()
                        tmp[0] = tmp[0] + tmp[2]/2
                        tmp[1] = tmp[1] + tmp[3]/2
                    if cls[0] == 1:
                        label_bboxes.append([0]+tmp)
                    elif cls[0] == 2:
                        label_bboxes.append([1]+tmp)
                
                with open("./dataset_det_seg/det_labels/"+name.replace('.png','.txt'), 'w') as f:
                    for box in label_bboxes:
                        assert np.all(np.array(box)>=0), f"there are negative values: {box}"
                        f.write(" ".join(list(map(str,box))) + "\n")

                #Save seg labels
                seg_mask = np.repeat(np.expand_dims(fire_bin_mask128, axis=2), axis=2, repeats=3).astype(np.uint8)
                cv2.imwrite("./dataset_det_seg/seg_labels/"+name.replace("rgb",'gt'), cv2.resize(seg_mask, (256, 256)))
                

                sys.stdout.write(f"\rProcessed images : {id_img} / {idx}")
            cv2.destroyAllWindows()

    if save_results:
        with open("./dataset_det_seg/train.txt", 'w') as f:
            for file in list_train_txt:
                f.write(file+'\n')

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str,   default="train",             help="processing phase: train, val")
    parser.add_argument('--dataset',        type=str,   default='fire8',              help='training dataset')
    parser.add_argument('--img_size',       type=int,   default=128,                help='test input resolution')
    parser.add_argument('--G_path',     type=str,   default="./outputs/model_test/HXNGHIA3/G_200.pth",
                                                                                   help='which epoch to load')
    parser.add_argument('--sample_path',    type=str,   default='samples',          help='path to save generated images')
    args = parser.parse_args()
    main(args)
