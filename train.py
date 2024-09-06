import os, sys
import argparse, logging, time, datetime

from torch import Tensor
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import cv2
from piqa import SSIM

from data.cocostuff_loader import CocoSceneGraphDataset
from data.data_loader import FireDataset
from model.resnet_generator import ResnetGenerator128
from model.rcnn_discriminator import CombineDiscriminator128, BkgResnetDiscriminator128
from utils.util import VGGLoss, draw_layout, truncted_random, combine_images, normalize_minmax


def add_normal_noise_input_D(images, mean=0, std=0.1):
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    return noisy_images

class SSIMLoss(SSIM):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return 1. - super().forward(x, y)

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def main(args):
    '''Configuration setup'''
    debug_phase = False
    #Common
    args.mode = 'train'
    args.batch_size = 32 if not debug_phase else 4
    args.total_epoch = 200 if not debug_phase else 10
    args.num_epoch_to_save = 5 if not debug_phase else 2
    args.print_freq = 150 if not debug_phase else 1
    args.num_workers = 4 if not debug_phase else 0
    args.seg_mask_thresh = 0.2

    #Special configurations: Developing phase
    weight_map_type = 'extreme'     #'extreme' creates weight_map 0 inside/1 outside bboxes
    
    max_num_obj = 2                 #if max_obj=2, get random first fire and smoke
    use_bkg_cls = True             #bboxes do not cover whole image --> True: add 1 bkg_cls + bkg_noise_embedding_input as random
    bkg_bbox_cover_whole = True    #bbox of bkg_cls cover whole image
    
    if use_bkg_cls and not bkg_bbox_cover_whole:
        max_num_obj *= 2
    elif use_bkg_cls and bkg_bbox_cover_whole:
        max_num_obj += 1

    use_res11 = False                               #use residual block 11
    use_D1_img_loss = 0                             #cases: 0 - no use, 1 - use for only fire image, 2 - use for both fire and non-fire image
    use_bkg_net_D = True                            #use bkg_D for background region
    use_mask_to_add_bkg_in_obj_loss_in_epoch = 0  #compute bkg_region(inside bbox) loss using binary mask from prediction, 0 = not used
    use_ssim_net_G = False                          #replace L1-loss by ssim-loss
    use_identity_loss = False                       #Later: use identity loss when input as fire-images
    use_instance_noise_input_D = False              #add Gaussian noise to input of D
    use_accuracy_constrain_D = False                #constraint the accuracy of whole_D: 0.8
    use_random_input_noise_w_enc_feat = False       #Later: use random input noise concatenating with enc_feat

    #Model
    z_obj_random_dim = 128
    z_obj_cls_dim = 128
    img_size = (args.img_size, args.img_size)
    lamb_obj = 1.0
    lamb_img = 0.05
    lamb_iden = 0.2
    g_lr, d_lr = args.g_lr, args.d_lr
    

    #Training Initilization
    dataset_path =      os.path.join("./datasets", args.dataset)
    if args.dataset == 'coco':
        train_img_dir =     os.path.join(dataset_path, "train2017")
        instances_json =    os.path.join(dataset_path, "annotations/instances_train2017.json")
        stuff_json =        os.path.join(dataset_path, "annotations/stuff_train2017.json")
        num_classes = 184

        train_data = CocoSceneGraphDataset(image_dir=train_img_dir,
                                       instances_json=instances_json,
                                       stuff_json=stuff_json,
                                       stuff_only=True, image_size=img_size, left_right_flip=True)

    elif 'fire' in args.dataset:
        train_fire_img_dir   = os.path.join(dataset_path, args.mode+"_images_A")
        train_non_fire_img_dir   = os.path.join(dataset_path, args.mode+"_images_B")
        classname_file  = os.path.join(dataset_path, "class_names.txt")
        num_classes = 3
        if use_bkg_cls: num_classes+=1
        
        train_data = FireDataset(fire_image_dir=train_fire_img_dir, 
                                non_fire_image_dir=train_non_fire_img_dir, 
                                classname_file=classname_file,
                                image_size=img_size, 
                                left_right_flip=True,
                                max_objects_per_image=max_num_obj,
                                weight_map_type=weight_map_type,
                                debug_phase=debug_phase)

        with open(os.path.join(dataset_path, "class_names.txt"), "r") as f:
            class_names = f.read().splitlines()

    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.num_workers)

    #Model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3, z_obj_random_dim=z_obj_random_dim, z_obj_class_dim=z_obj_cls_dim, 
                              random_input_noise=use_random_input_noise_w_enc_feat, use_res11=use_res11).cuda()
    netD = CombineDiscriminator128(num_classes=num_classes).cuda()
    if use_bkg_net_D:
        netD2 = BkgResnetDiscriminator128(num_classes=num_classes).cuda()

    #Optimizers
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr*0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]
    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0.5, 0.999))

    #disc: fire/fake-fire
    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0.5, 0.999))

    if use_bkg_net_D:
        #bkg: non-fire/fake-non-fire
        dis2_parameters = []
        for key, value in dict(netD2.named_parameters()).items():
            if value.requires_grad:
                dis2_parameters += [{'params': [value], 'lr': d_lr}]
        d2_optimizer = torch.optim.Adam(dis2_parameters, betas=(0.5, 0.999))

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.mkdir(os.path.join(args.out_path, 'model/'))
    if not os.path.exists(os.path.join(args.out_path, 'samples/')):
        os.mkdir(os.path.join(args.out_path, 'samples/'))
    if not os.path.exists(os.path.join(args.out_path, 'model/log/')):
        os.mkdir(os.path.join(args.out_path, 'model/log/'))

    #tensorboard summary writer
    writer  = SummaryWriter(os.path.join(args.out_path, 'model/log/'))
    global_steps = torch.LongTensor([1]).cuda()
    steps_per_epochs = len(dataloader)

    logger = setup_logger("lostGAN", args.out_path, 0)
    # logger.info(netG)
    # logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()                #average L2-norm between reference and reconstructed
    l1_loss = nn.L1Loss()
    ssim = SSIMLoss().cuda()
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()

        if use_bkg_net_D:
            netD2.train()
        else:
            d2_loss_rimg, d2_loss_robj, d2_loss_fimg, d2_loss_fobj, d2_loss = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
            g2_loss_fimg, g2_loss_fobj =  torch.tensor(0), torch.tensor(0)
        
        if use_ssim_net_G:
            pixel_loss =    torch.tensor(0)
            obj_pixel_loss = torch.tensor(0)     
        else:
            ssim_loss =     torch.tensor(0)
            obj_ssim_loss = torch.tensor(0)
        
        if not use_identity_loss:
            rec_pixel_loss = torch.tensor(0)
            rec_feat_loss = torch.tensor(0)
        
        d1_real_img, d1_real_obj, d1_fake_img, d1_fake_obj, d1_all = 0,0,0,0,0                                  #losses D1
        d2_real_img, d2_real_obj, d2_fake_img, d2_fake_obj, d2_all = 0,0,0,0,0                                  #losses D2 (background)
        g_fake_img, g_fake_obj, g2_fake_img, g2_fake_obj, g_l1, g_vgg, g_ssim, g_obj_l1, g_obj_vgg, g_obj_ssim, g_rec_l1, g_rec_vgg, g_all = 0,0,0,0,0,0,0,0,0,0,0,0,0 #losses G
        d1_real_acc_cnt, d1_fake_acc_cnt, d1_real_num_sample, d1_fake_num_sample = 0,0,0,0
        
        for idx, data in enumerate(dataloader):
            [fire_images, non_fire_images], label, bbox, weight_map_orig = data
            fire_images, non_fire_images        = fire_images.cuda(), non_fire_images.cuda()
            label, bbox                         = label.long().cuda().unsqueeze(-1), bbox.float()                   #keep bbox in cpu --> make input of netG,netD in gpu
            weight_map_orig  = weight_map_orig.float().cuda()
            #weight_map for only 2 objects (also in case 3 objects)
            weight_map = torch.all(weight_map_orig, dim=1, keepdim=True).expand(fire_images.shape).type(torch.cuda.IntTensor)
            
            #obj noise
            z_obj = torch.randn(fire_images.size(0), max_num_obj, z_obj_random_dim).cuda()     #[batch, num_obj, 128]

            #Forward()
            fake_images, stage_mask128, _ = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))

            #process binary masks to compute bkg_inside_box loss
            if use_bkg_cls and use_mask_to_add_bkg_in_obj_loss_in_epoch and epoch>use_mask_to_add_bkg_in_obj_loss_in_epoch:
                if max_num_obj == 3:
                    fire = torch.unsqueeze(torch.argmax(torch.concat([stage_mask128[:,2:3], stage_mask128[:,0:1]], dim=1), dim=1), dim=1)
                    smoke = torch.unsqueeze(torch.argmax(torch.concat([stage_mask128[:,2:3]*0.5, stage_mask128[:,1:2]], dim=1), dim=1), dim=1)
                elif max_num_obj == 4:
                    fire = torch.unsqueeze(torch.argmax(torch.concat([stage_mask128[:,1:2], stage_mask128[:,0:1]], dim=1), dim=1), dim=1)
                    smoke = torch.unsqueeze(torch.argmax(torch.concat([stage_mask128[:,3:4], 1-stage_mask128[:,2:3]], dim=1), dim=1), dim=1)
                weight_map_3 = torch.concat([fire, smoke], dim=1)
                weight_map_3 = torch.all(weight_map_3, dim=1, keepdim=True).expand(fire_images.shape).type(torch.cuda.IntTensor)                #weight_map from binary mask

            if use_accuracy_constrain_D:
                with torch.no_grad():
                    if use_instance_noise_input_D:
                        d_out_real, _ = netD(add_normal_noise_input_D(fire_images), bbox.cuda(), label)
                        d_out_fake, _ = netD(add_normal_noise_input_D(fake_images.detach()), bbox.cuda(), label)
                    else:
                        d_out_real, _ = netD(fire_images, bbox.cuda(), label)
                        d_out_fake, _ = netD(fake_images.detach(), bbox.cuda(), label)
                    d_target = torch.ones([d_out_real.shape[0],1], dtype=torch.bool).cuda()
                    d_real_acc_tmp = torch.sum((d_out_real>0) == d_target).item() / d_target.shape[0]
                    d_fake_acc_tmp = torch.sum((d_out_fake<0) == d_target).item() / d_target.shape[0]
                
                if not (d_real_acc_tmp>0.8 and d_fake_acc_tmp>0.8):
                    # update D network
                    netD.zero_grad()
                    #real image+objects
                    if use_instance_noise_input_D:
                        d_out_real, d_out_robj = netD(add_normal_noise_input_D(fire_images), bbox.cuda(), label)
                    else:
                        d_out_real, d_out_robj = netD(fire_images, bbox.cuda(), label)
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                    d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
                    
                    if use_instance_noise_input_D:
                        d_out_fake, d_out_fobj = netD(add_normal_noise_input_D(fake_images.detach()), bbox.cuda(), label)
                    else:
                        d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox.cuda(), label)
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                    d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()

                    d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + (lamb_img) * (d_loss_real + d_loss_fake)
                    d_loss.backward()
                    d_optimizer.step()
            
            else: #normal training of D
                # update D network
                netD.zero_grad()
                if not use_bkg_cls and not bkg_bbox_cover_whole:
                    label_obj = label.clone()
                    bbox_obj = bbox.clone()
                    # label_bkg = None
                    # bbox_bkg = None
                elif use_bkg_cls and bkg_bbox_cover_whole:
                    label_obj = label[:,0:max_num_obj-1,:]
                    bbox_obj = bbox[(label.cpu()!=3).squeeze()].view(-1,max_num_obj-1,4)
                    # label_bkg = label[:,max_num_obj-1:,:]
                    # bbox_bkg = bbox[(label.cpu()==3).squeeze()].view(-1,1,4)
                elif use_bkg_cls and not bkg_bbox_cover_whole:
                    label_obj = label[:,0::2,:]
                    bbox_obj = bbox[:,0::2,:]
                    # label_bkg = label[:,1::2,:]
                    # bbox_bkg = bbox[:,1::2,:]
                else:
                    raise ValueError("Configuration is wrong!: use_bkg_cls={} but bkg_bbox_cover_whole={}".format(use_bkg_cls, bkg_bbox_cover_whole))
                
                #1) real fire-image+objects(including bkg)
                if use_instance_noise_input_D:
                    d_out_rimg_fire, d_out_robj = netD(add_normal_noise_input_D(fire_images), bbox_obj.cuda(), label_obj)
                else:
                    d_out_rimg_fire, d_out_robj = netD(fire_images, bbox_obj.cuda(), label_obj)
                d_loss_rimg_fire = torch.tensor(0).cuda() #if use_D1_img_loss==0 else torch.nn.ReLU()(1.0 - d_out_rimg_fire).mean()
                d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
                
                # #2) real nonfire-image+bkg
                # if use_instance_noise_input_D:
                #     d_out_rimg_nonfire, d_out_rbkg = netD(add_normal_noise_input_D(non_fire_images), bbox_bkg.cuda() if bbox_bkg!=None else None, label_bkg)
                # else:
                #     d_out_rimg_nonfire, d_out_rbkg = netD(non_fire_images, bbox_bkg.cuda() if bbox_bkg!=None else None, label_bkg)
                # d_loss_rimg_nonfire = torch.tensor([0]).cuda() if (use_D1_img_loss==0 or use_D1_img_loss==1) else torch.nn.ReLU()(1.0 - d_out_rimg_nonfire).mean()
                # if d_out_rbkg==None:
                #     d_loss_rbkg = torch.tensor([0]).cuda()
                # else:
                #     d_loss_rbkg = torch.nn.ReLU()(1.0 - d_out_rbkg).mean()
                
                #3) fake fire-image+objects+bkg
                if use_instance_noise_input_D:
                    d_out_fimg_fire, d_out_fobj = netD(add_normal_noise_input_D(fake_images.detach()), bbox_obj.cuda(), label_obj)
                    # _, d_out_fbkg = netD(add_normal_noise_input_D(fake_images.detach()), bbox_bkg.cuda() if bbox_bkg!=None else None, label_bkg)
                else:
                    d_out_fimg_fire, d_out_fobj = netD(fake_images.detach(), bbox_obj.cuda(), label_obj)
                    # _, d_out_fbkg = netD(fake_images.detach(), bbox_bkg.cuda() if bbox_bkg!=None else None, label_bkg)
                d_loss_fimg_fire = torch.tensor(0).cuda() #if use_D1_img_loss==0 else torch.nn.ReLU()(1.0 + d_out_fimg_fire).mean()
                d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
                # if d_out_fbkg==None:
                #     d_loss_fbkg = torch.tensor([0]).cuda()
                # else:
                #     d_loss_fbkg = torch.nn.ReLU()(1.0 + d_out_fbkg).mean()

                #7 losses: real fire, real non-fire, fake fire, real_obj, real_bkg, fake_obj, fake_bkg

                d_loss = lamb_obj * (d_loss_robj + d_loss_fobj)# + d_loss_rbkg*0.1 + d_loss_fbkg*0.1)
                d_loss += lamb_img * (d_loss_rimg_fire + d_loss_fimg_fire)# + d_loss_rimg_nonfire)
                d_loss.backward()
                d_optimizer.step()
                """For D1: use only fake/real fire obj_loss in this version"""

            d_target = torch.ones([d_out_rimg_fire.shape[0],1], dtype=torch.bool).cuda()
            d1_real_acc_cnt += torch.sum((d_out_rimg_fire>0) == d_target).item()
            d1_fake_acc_cnt += torch.sum((d_out_fimg_fire<0) == d_target).item()
            d1_real_num_sample += d_out_rimg_fire.shape[0]
            d1_fake_num_sample += d_out_fimg_fire.shape[0]

            writer.add_scalar("iter_d1_loss/d1_real_img", d_loss_rimg_fire*(lamb_img), global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_fake_img", d_loss_fimg_fire*(lamb_img), global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_real_obj", d_loss_robj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_fake_obj", d_loss_fobj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_total", d_loss, global_step=global_steps)
            
            d1_real_img += d_loss_rimg_fire*(lamb_img)
            d1_fake_img += d_loss_fimg_fire*(lamb_img)
            d1_real_obj += d_loss_robj*lamb_obj
            d1_fake_obj += d_loss_fobj*lamb_obj
            d1_all += d_loss

            if use_bkg_net_D:
                #update D2 network
                netD2.zero_grad()
                #real bkg_region
                if use_instance_noise_input_D:
                    d2_out_rimg, _ = netD2(add_normal_noise_input_D(non_fire_images*(weight_map)), bbox_obj.cuda(), label_obj)
                else:
                    d2_out_rimg, _ = netD2(non_fire_images*(weight_map), bbox_obj.cuda(), label_obj)
                d2_loss_rimg = torch.nn.ReLU()(1.0 - d2_out_rimg).mean()
                d2_loss_robj = torch.tensor(0).cuda() #torch.nn.ReLU()(1.0 - d2_out_robj).mean()
                
                #fake bkg_region
                if use_instance_noise_input_D:
                    d2_out_fimg, _ = netD2(add_normal_noise_input_D(fake_images.detach()*(weight_map)), bbox_obj.cuda(), label_obj)
                else:
                    d2_out_fimg, _ = netD2(fake_images.detach()*(weight_map), bbox_obj.cuda(), label_obj)
                d2_loss_fimg = torch.nn.ReLU()(1.0 + d2_out_fimg).mean()
                d2_loss_fobj = torch.tensor(0).cuda() #torch.nn.ReLU()(1.0 + d2_out_fobj).mean()

                d2_loss = lamb_img * (d2_loss_robj + d2_loss_fobj)
                d2_loss += lamb_img * (d2_loss_rimg + d2_loss_fimg)
                d2_loss.backward()
                d2_optimizer.step()
                """For D2: use only fake/real nonfire_region img_loss in this version"""

            writer.add_scalar("iter_d2_loss/d2_real_img", d2_loss_rimg*lamb_img*5, global_step=global_steps)
            writer.add_scalar("iter_d2_loss/d2_fake_img", d2_loss_fimg*lamb_img*5, global_step=global_steps)
            writer.add_scalar("iter_d2_loss/d2_real_obj", d2_loss_robj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_d2_loss/d2_fake_obj", d2_loss_fobj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_d2_loss/d2_total", d2_loss, global_step=global_steps)
            d2_real_img += d2_loss_rimg*lamb_img*5
            d2_fake_img += d2_loss_fimg*lamb_img*5
            d2_real_obj += d2_loss_robj*lamb_obj
            d2_fake_obj += d2_loss_fobj*lamb_obj
            d2_all += d2_loss

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                #Adversarial loss from D1
                if use_instance_noise_input_D:
                    _, g_out_fobj = netD(add_normal_noise_input_D(fake_images), bbox_obj.cuda(), label_obj)
                    # _, g_out_fbkg = netD(add_normal_noise_input_D(fake_images), bbox_bkg.cuda() if bbox_bkg!=None else None, label_bkg)
                else:
                    _, g_out_fobj = netD(fake_images, bbox_obj.cuda(), label_obj)
                    # _, g_out_fbkg = netD(fake_images, bbox_bkg.cuda() if bbox_bkg!=None else None, label_bkg)
                g_loss_fimg = torch.tensor(0).cuda() #if use_D1_img_loss==0 else -g_out_fimg.mean()
                g_loss_fobj = -g_out_fobj.mean()
                # if g_out_fbkg==None:
                #     g_loss_fbkg = torch.tensor([0]).cuda()
                # else:
                #     g_loss_fbkg = -g_out_fbkg.mean()
                
                #Adversarial loss from D2
                if use_bkg_net_D:
                    if use_instance_noise_input_D:
                        g2_out_fimg, _ = netD2(add_normal_noise_input_D(fake_images*(weight_map)), bbox_obj.cuda(), label_obj)
                    else:
                        g2_out_fimg, _ = netD2(fake_images*(weight_map), bbox_obj.cuda(), label_obj)
                    g2_loss_fimg = - g2_out_fimg.mean()
                    g2_loss_fobj = torch.tensor(0).cuda() #- g2_out_fobj.mean()

                #structure similarity loss
                if use_ssim_net_G:
                    ssim_loss = ssim((fake_images*0.5+0.5)*(weight_map), (non_fire_images*0.5+0.5)*(weight_map))
                    obj_ssim_loss = ssim((fake_images*0.5+0.5)*(1-weight_map), (fire_images*0.5+0.5)*(1-weight_map))
                else:
                    pixel_loss = l1_loss(fake_images*(weight_map), non_fire_images*(weight_map)).mean()
                    obj_pixel_loss = l1_loss(fake_images*(1-weight_map), fire_images*(1-weight_map)).mean()
                    if use_bkg_cls:
                        bkg_pixel_loss = l1_loss(fake_images*(1-weight_map), non_fire_images*(1-weight_map)).mean()     #bkg_region inside bbox

                #reconstruction loss
                feat_loss = vgg_loss(fake_images*(weight_map), non_fire_images*(weight_map)).mean()
                obj_feat_loss = vgg_loss(fake_images*(1-weight_map), fire_images*(1-weight_map)).mean()
                if use_bkg_cls:
                    bkg_feat_loss = vgg_loss(fake_images*(1-weight_map), non_fire_images*(1-weight_map)).mean()

                #Identity loss
                if use_identity_loss:
                    rec_images, _, _ = netG(z_img=fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))
                    rec_pixel_loss = l1_loss(rec_images*(1-weight_map), fire_images*(1-weight_map)).mean()
                    rec_feat_loss = vgg_loss(rec_images*(1-weight_map), fire_images*(1-weight_map)).mean()

                if use_mask_to_add_bkg_in_obj_loss_in_epoch and epoch>use_mask_to_add_bkg_in_obj_loss_in_epoch:
                    feat_loss = vgg_loss(fake_images*(1-weight_map_3), non_fire_images*(1-weight_map_3)).mean()
                    pixel_loss = l1_loss(fake_images*(1-weight_map_3), non_fire_images*(1-weight_map_3)).mean()

                    # g_loss = (g_loss_fobj + g_loss_fbkg) * lamb_obj + g_loss_fimg * (lamb_img/2) + pixel_loss + feat_loss
                    g_loss = g_loss_fobj * lamb_obj + g_loss_fimg * lamb_img + pixel_loss + feat_loss

                else:
                    #Total losses
                    # g_loss = (g_loss_fobj + g_loss_fbkg) * lamb_obj + g_loss_fimg * (lamb_img/2) + feat_loss + obj_feat_loss
                    g_loss = g_loss_fobj * lamb_obj + g_loss_fimg * lamb_img + feat_loss + obj_feat_loss        # D1_adv_fake_obj + feat + obj_feat
                    #-------------------------
                    if use_ssim_net_G:
                        g_loss += ssim_loss + obj_ssim_loss
                    else:
                        g_loss += pixel_loss + obj_pixel_loss                                                   # + pixel + obj_pixel
                    #-------------------------
                    if use_bkg_net_D:
                        g_loss += g2_loss_fobj*lamb_img + g2_loss_fimg*lamb_img                                 # + D2_adv_fake_img*0.05
                    #-------------------------
                    if use_bkg_cls:
                        g_loss += (bkg_pixel_loss + bkg_feat_loss) * lamb_img                                   # + (bkg_pixel + bkg_feat)*0.05
                    # skip now ---------------
                    if use_identity_loss:
                        g_loss += (rec_pixel_loss + rec_feat_loss) * lamb_obj * lamb_iden

                g_loss.backward()
                g_optimizer.step()

            writer.add_scalar("iter_g_loss/g_fake_img", g_loss_fimg*(lamb_img), global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_fake_obj", g_loss_fobj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_fake_img", g2_loss_fimg*lamb_img*5, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_fake_obj", g2_loss_fobj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_l1", pixel_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_vgg", feat_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_ssim", ssim_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/obj_g_l1", obj_pixel_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/obj_g_vgg", obj_feat_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/obj_g_ssim", obj_ssim_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_rec_l1", rec_pixel_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_rec_vgg", rec_feat_loss, global_step=global_steps)
            writer.add_scalar("iter_g_loss/g_total", g_loss, global_step=global_steps)
            
            g_fake_img += g_loss_fimg*(lamb_img)
            g_fake_obj += g_loss_fobj*lamb_obj
            g_fake_img += g2_loss_fimg*lamb_img*5
            g_fake_obj += g2_loss_fobj*lamb_obj
            g_l1 += pixel_loss
            g_vgg += feat_loss
            g_ssim += ssim_loss
            g_obj_l1 += obj_pixel_loss
            g_obj_vgg += obj_feat_loss
            g_obj_ssim += obj_ssim_loss
            g_rec_l1 += rec_pixel_loss
            g_rec_vgg += rec_feat_loss
            g_all += g_loss

            if (idx+1) % args.print_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Step[{}/{}],  d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                        idx + 1,
                                                                                                        d_loss_rimg_fire.item(),
                                                                                                        d_loss_fimg_fire.item(),
                                                                                                        g_loss_fimg.item()))
                logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                                                                                                        d_loss_robj.item(),
                                                                                                        d_loss_fobj.item(),
                                                                                                        g_loss_fobj.item()))
                logger.info("             ssim_loss: {:.4f}, pixel_loss: {:.4f}, feat_loss: {:.4f}".format(
                                                                                                        ssim_loss.item(), 
                                                                                                        pixel_loss.item(), 
                                                                                                        feat_loss.item()))
                logger.info("             obj_ssim_loss: {:.4f}, obj_pixel_loss: {:.4f}, obj_feat_loss: {:.4f}".format(
                                                                                                        obj_ssim_loss.item(), 
                                                                                                        obj_pixel_loss.item(), 
                                                                                                        obj_feat_loss.item()))
                logger.info("             rec_pixel_loss: {:.4f}, rec_feat_loss: {:.4f}, total_loss: {:.4f}".format(
                                                                                                        rec_pixel_loss.item(), 
                                                                                                        rec_feat_loss.item(),
                                                                                                        g_loss.item()))
                
                if use_bkg_net_D:
                    logger.info("             d2_out_rimg: {:.4f}, d2_out_fimg: {:.4f}, g2_out_fimg: {:.4f} ".format(d2_loss_rimg.item(), d2_loss_fimg.item(), g2_loss_fimg.item()))
                    logger.info("             d2_out_robj: {:.4f}, d2_out_fobj: {:.4f}, g2_out_fobj: {:.4f} ".format(d2_loss_robj.item(), d2_loss_fobj.item(), g2_loss_fobj.item()))
                # logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))

            global_steps += 1


        #End of each epoch
        writer.add_scalar("epoch_d1_loss/d1_real_img", d1_real_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_real_obj", d1_real_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_fake_img", d1_fake_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_fake_obj", d1_fake_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_total", d1_all/steps_per_epochs, global_step=epoch+1)

        writer.add_scalar("epoch_d2_loss/d2_real_img", d2_real_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d2_loss/d2_fake_img", d2_fake_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d2_loss/d2_real_obj", d2_real_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d2_loss/d2_fake_obj", d2_fake_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d2_loss/d2_total", d2_all/steps_per_epochs, global_step=epoch+1)

        writer.add_scalar("epoch_g_loss/g_fake_img", g_fake_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_fake_obj", g_fake_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_fake_img", g_fake_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_fake_obj", g_fake_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_l1", g_l1/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_vgg", g_vgg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_ssim", g_ssim/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/obj_g_l1", g_obj_l1/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/obj_g_vgg", g_obj_vgg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/obj_g_ssim", g_obj_ssim/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_rec_l1", g_rec_l1/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_rec_vgg", g_rec_vgg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_total", g_all/steps_per_epochs, global_step=epoch+1)

        writer.add_scalar("epoch_d1_accuracy/d1_real_acc", d1_real_acc_cnt/d1_real_num_sample, global_step=epoch+1)
        writer.add_scalar("epoch_d1_accuracy/d1_fake_acc", d1_fake_acc_cnt/d1_fake_num_sample, global_step=epoch+1)
        writer.add_scalar("epoch_d1_accuracy/d1_total_acc", (d1_real_acc_cnt+d1_fake_acc_cnt)/(d1_real_num_sample+d1_fake_num_sample), global_step=epoch+1)

        # save model
        if (epoch + 1) % args.num_epoch_to_save == 0:
            torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch+1)))
            torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch+1)))
            
            for idx, data in enumerate(dataloader):
                if idx == 0:
                    [fire_images, non_fire_images], label, bbox, weight_map_orig = data
                    fire_images, non_fire_images = fire_images[0:1].cuda(), non_fire_images[0:1].cuda()
                    label, bbox = label[0:1].long().cuda().unsqueeze(-1), bbox[0:1].float()
                    weight_map_orig = weight_map_orig.float().cuda()
                    z_obj = torch.from_numpy(truncted_random(z_obj_dim=z_obj_random_dim, num_o=max_num_obj, thres=2.0)).float().cuda()
                    break

            #Network() processing    
            netG.eval()
            netD.eval()
            fake_images, stage_mask128, [bbox_mask64, stage_mask16, stage_mask32, stage_mask64] = netG.forward(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
            g_out_fake, _ = netD(fake_images, None, None)
            g_out_real, _ = netD(fire_images, None, None)

            #Img_show() processing
            #1) fake-fire
            fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            fake_images = np.array(fake_images*255, np.uint8)
            g_out_fake  = g_out_fake[0,0].cpu().detach()
            g_out_fake = -1 if g_out_fake>1 else 1 if g_out_fake<-1 else -torch.round(g_out_fake,decimals=2)
            fake_images = draw_layout(label, bbox, [256,256], class_names, fake_images, g_out_fake, topleft_name='Fake-fire image')
            #2) real-fire
            fire_images = fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            fire_images = np.array(fire_images*255, np.uint8)
            g_out_real  = g_out_real[0,0].cpu().detach()
            g_out_real = -1 if g_out_real<-1 else 1 if g_out_real>1 else torch.round(g_out_real,decimals=2)
            fire_images = draw_layout(label, bbox, [256,256], class_names, fire_images, g_out_real, topleft_name='Real-fire image')
            #3) non-fire
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
                    fire_bin_mask128 = normalize_minmax(np.argmax((1-weight_map_orig.cpu().numpy()[0,0:1])*np.concatenate([stage_mask128[1:2],stage_mask128[0:1]], axis=0), axis=0), [0,255], [0,1])
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

            
            if use_bkg_cls:
                output_images = combine_images([fire_images, non_fire_images, fake_images, fire_mask128, smoke_mask128,
                                                embed_mask64, soft_mask16, soft_mask32, soft_mask64, soft_mask128, 
                                                embed_mask64_bkg, soft_mask16_bkg, soft_mask32_bkg, soft_mask64_bkg, soft_mask128_bkg], [256,256])
            else:
                output_images = combine_images([fire_images, non_fire_images, fake_images, fire_mask128, smoke_mask128,
                                                embed_mask64, soft_mask16, soft_mask32, soft_mask64, soft_mask128], [256,256])
            
            cv2.imwrite(args.out_path+"samples/"+ 'G_epoch_%d.png'%(epoch+1), cv2.cvtColor(output_images.astype(np.uint8), cv2.COLOR_RGB2BGR))
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str,   default="train",            help="processing phase: train, test")
    parser.add_argument('--dataset',        type=str,   default="fire8",            help="dataset used for training")
    parser.add_argument('--img_size',       type=int,   default=128,                help="training input image size. Default: 128x128")
    parser.add_argument('--batch_size',     type=int,   default=16,                 help="training batch size. Default: 8")
    parser.add_argument('--total_epoch',    type=int,   default=200,                help="numer of total training epochs")
    parser.add_argument('--g_lr',           type=float, default=0.0001,             help="learning rate of generator")
    parser.add_argument('--d_lr',           type=float, default=0.0001,             help="learning rate of discriminator")
    parser.add_argument('--out_path',       type=str,   default="./outputs/",       help="path to output files")
    parser.add_argument('--num_workers',    type=int,   default=0,                  help="Number of workers for dataset parallel processing")
    args = parser.parse_args()
    main(args)

