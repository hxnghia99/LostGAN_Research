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
    args.batch_size = 16 if not debug_phase else 4
    args.total_epoch = 200
    args.num_epoch_to_save = 5 if not debug_phase else 2
    args.print_freq = 150 if not debug_phase else 1
    args.num_workers = 4 if not debug_phase else 1
    args.seg_mask_thresh = 0.5
    
    #Special : Test
    use_noised_input = False
    weight_map_type = 'extreme'
    max_num_obj = 2                 #if max_obj=2, get only first fire and smoke
    get_first_fire_smoke = True if max_num_obj==2 else False    #bboxes do not cover whole image --> add 1 __background__ class
    
    use_ssim_net_G = False       #replace L1-loss by ssim-loss
    use_bkg_net_D = False
    use_instance_noise_input_D = False
    use_accuracy_constrain_D = False
    use_identity_loss = False
    use_weight_map_from_stage_bbox_masks = False

    #Model
    z_obj_random_dim = 128
    z_obj_cls_dim = 128
    normalized = True  #re-scale from [0,1] to [-1,1]
    img_size = (args.img_size, args.img_size)
    lamb_obj = 1.0
    lamb_img = 0.05
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
        
        train_data = FireDataset(fire_image_dir=train_fire_img_dir, non_fire_image_dir=train_non_fire_img_dir, 
                                classname_file=classname_file,
                                image_size=img_size, left_right_flip=True,
                                max_objects_per_image=max_num_obj,
                                get_first_fire_smoke=get_first_fire_smoke,
                                use_noised_input=use_noised_input,
                                weight_map_type=weight_map_type,
                                normalize_images=normalized,
                                debug_phase=debug_phase)

        with open(os.path.join(dataset_path, "class_names.txt"), "r") as f:
            class_names = f.read().splitlines()

    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.num_workers)

    #Model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3, z_obj_random_dim=z_obj_random_dim, z_obj_class_dim=z_obj_cls_dim, normalized_data=normalized).cuda()
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
            dis_parameters += [{'params': [value], 'lr': d_lr/2}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0.5, 0.999))
    # d_optimizer = torch.optim.SGD(dis_parameters)
    if use_bkg_net_D:
        #bkg: non-fire/fake-non-fire
        dis2_parameters = []
        for key, value in dict(netD2.named_parameters()).items():
            if value.requires_grad:
                dis2_parameters += [{'params': [value], 'lr': d_lr/2}]
        d2_optimizer = torch.optim.Adam(dis2_parameters, betas=(0.5, 0.999))
        # d2_optimizer = torch.optim.SGD(dis2_parameters)

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
        if epoch >= 150:
            use_weight_map_from_stage_bbox_masks = True
        if use_bkg_net_D:
            netD2.train()
        else:
            d2_loss_real = torch.tensor([0])
            d2_loss_fake = torch.tensor([0])
            d2_loss = torch.tensor([0])
            g2_loss_fake = torch.tensor([0])
        if not use_ssim_net_G:
            ssim_loss = torch.tensor([0])
        else:
            pixel_loss = torch.tensor([0])
        if not use_identity_loss:
            rec_pixel_loss = torch.tensor([0])
            rec_feat_loss = torch.tensor([0])
        pixel_loss = torch.tensor([0])
        d1_real_img, d1_real_obj, d1_fake_img, d1_fake_obj, d1_all = 0,0,0,0,0
        d2_real_bkg, d2_fake_bkg, d2_all = 0,0,0
        g_fake_img, g_fake_obj, g_fake_bkg, g_l1, g_vgg, g_ssim, g_rec_l1, g_rec_vgg, g_all = 0,0,0,0,0,0,0,0,0
        d1_real_acc_cnt, d1_fake_acc_cnt, d1_real_num_sample, d1_fake_num_sample = 0,0,0,0
        for idx, data in enumerate(dataloader):
            [fire_images, non_fire_images, non_fire_crops], label, bbox, weight_map = data
            fire_images, label, bbox, weight_map = fire_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float(), weight_map.float().cuda()      #keep bbox in cpu --> make input of netG,netD in gpu
            non_fire_images, non_fire_crops = non_fire_images.cuda(), non_fire_crops.cuda()
            weight_map = torch.all(weight_map, dim=1, keepdim=True).expand(fire_images.shape).type(torch.cuda.IntTensor)
            
            #fake image+objects
            if normalized:
                z_obj = torch.randn(fire_images.size(0), max_num_obj, z_obj_random_dim).cuda()     #[batch, num_obj, 128]
            else:
                z_obj = torch.rand(fire_images.size(0), max_num_obj, z_obj_random_dim).cuda()

            fake_images, stage_bbox_masks = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))
            stage_bbox_masks = F.interpolate(stage_bbox_masks, size=img_size, mode="nearest")
            weight_map_from_stage_bbox_masks = 1 - torch.unsqueeze(torch.logical_or(stage_bbox_masks[:,0,:,:]>args.seg_mask_thresh, stage_bbox_masks[:,1,:,:]>args.seg_mask_thresh).type(torch.cuda.FloatTensor), dim=1)
            weight_map_from_stage_bbox_masks = weight_map_from_stage_bbox_masks.expand(stage_bbox_masks.shape[0], 3, stage_bbox_masks.shape[2], stage_bbox_masks.shape[3])

            if not normalized:
                fake_images = fake_images*0.5+0.5   #scale to [0,1]

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

                    d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
                    d_loss.backward()
                    d_optimizer.step()
            
            else:
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

                d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
                d_loss.backward()
                d_optimizer.step()


            d_target = torch.ones([d_out_real.shape[0],1], dtype=torch.bool).cuda()
            d1_real_acc_cnt += torch.sum((d_out_real>0) == d_target).item()
            d1_fake_acc_cnt += torch.sum((d_out_fake<0) == d_target).item()
            d1_real_num_sample += d_out_real.shape[0]
            d1_fake_num_sample += d_out_fake.shape[0]

            writer.add_scalar("iter_d1_loss/d1_real_img", d_loss_real*lamb_img, global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_fake_img", d_loss_fake*lamb_img, global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_real_obj", d_loss_robj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_fake_obj", d_loss_fobj*lamb_obj, global_step=global_steps)
            writer.add_scalar("iter_d1_loss/d1_total", d_loss, global_step=global_steps)
            
            d1_real_img += d_loss_real*lamb_img
            d1_fake_img += d_loss_fake*lamb_img
            d1_real_obj += d_loss_robj*lamb_obj
            d1_fake_obj += d_loss_fobj*lamb_obj
            d1_all += d_loss

            if use_bkg_net_D:
                #update D2 network
                netD2.zero_grad()
                #real bkg
                if use_instance_noise_input_D:
                    d2_out_real = netD2(add_normal_noise_input_D(non_fire_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks)))
                else:
                    d2_out_real = netD2(non_fire_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks))
                d2_loss_real = torch.nn.ReLU()(1.0 - d2_out_real).mean()
                #fake bkg
                if use_instance_noise_input_D:
                    d2_out_fake = netD2(add_normal_noise_input_D(fake_images.detach()*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks)))
                else:
                    d2_out_fake = netD2(fake_images.detach()*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks))
                d2_loss_fake = torch.nn.ReLU()(1.0 + d2_out_fake).mean()

                d2_loss = lamb_img * (d2_loss_real + d2_loss_fake)
                d2_loss.backward()
                d2_optimizer.step()

            writer.add_scalar("iter_d2_loss/d2_real_bkg", d2_loss_real*lamb_img, global_step=global_steps)
            writer.add_scalar("iter_d2_loss/d2_fake_bkg", d2_loss_fake*lamb_img, global_step=global_steps)
            writer.add_scalar("iter_d2_loss/d2_total", d2_loss, global_step=global_steps)
            d2_real_bkg += d2_loss_real*lamb_img
            d2_fake_bkg += d2_loss_fake*lamb_img
            d2_all += d2_loss

            # update G network
            if (idx % 1) == 0:      
                netG.zero_grad()
                #Adversarial loss from D1
                if use_instance_noise_input_D:
                    g_out_fake, g_out_fobj = netD(add_normal_noise_input_D(fake_images), bbox.cuda(), label)
                else:
                    g_out_fake, g_out_fobj = netD(fake_images, bbox.cuda(), label)
                g_loss_fake =  - g_out_fake.mean()
                g_loss_fobj = - g_out_fobj.mean()
                
                #Adversarial loss from D2
                if use_bkg_net_D:
                    if use_instance_noise_input_D:
                        g2_out_fake = netD2(add_normal_noise_input_D(fake_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks)))
                    else:
                        g2_out_fake = netD2(fake_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks))
                    g2_loss_fake = - g2_out_fake.mean()

                #structure similarity loss
                if use_ssim_net_G:
                    ssim_loss = ssim((fake_images*0.5+0.5)*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks), (non_fire_images*0.5+0.5)*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks))
                else:
                    pixel_loss = l1_loss(fake_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks), non_fire_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks)).mean()

                #background reconstruction loss
                feat_loss = vgg_loss(fake_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks), non_fire_images*(weight_map if not use_weight_map_from_stage_bbox_masks else weight_map_from_stage_bbox_masks)).mean()

                #Identity loss
                if use_identity_loss:
                    rec_images, _ = netG(z_img=fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))
                    rec_pixel_loss = l1_loss(rec_images*(1-weight_map), fire_images*(1-weight_map)).mean()
                    rec_feat_loss = vgg_loss(rec_images*(1-weight_map), fire_images*(1-weight_map)).mean()

                g_loss = g_loss_fobj * lamb_obj + g_loss_fake * lamb_img + feat_loss
                if use_bkg_net_D:
                    g_loss += g2_loss_fake * lamb_img
                if use_ssim_net_G:
                    g_loss += ssim_loss
                else:
                    g_loss += pixel_loss
                if use_identity_loss:
                    g_loss += (rec_pixel_loss + rec_feat_loss) * lamb_img * 0.2

                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar("iter_g_loss/g_fake_img", g_loss_fake*lamb_img, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_fake_obj", g_loss_fobj*lamb_obj, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_fake_bkg", g2_loss_fake*lamb_img, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_l1", pixel_loss, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_vgg", feat_loss, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_ssim", ssim_loss, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_rec_l1", rec_pixel_loss, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_rec_vgg", rec_feat_loss, global_step=global_steps)
                writer.add_scalar("iter_g_loss/g_total", g_loss, global_step=global_steps)
                g_fake_img += g_loss_fake*lamb_img
                g_fake_obj += g_loss_fobj*lamb_obj
                g_fake_bkg += g2_loss_fake*lamb_img
                g_l1 += pixel_loss
                g_vgg += feat_loss
                g_ssim += ssim_loss
                g_rec_l1 += rec_pixel_loss
                g_rec_vgg += rec_feat_loss
                g_all += g_loss

            if (idx+1) % args.print_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                        idx + 1,
                                                                                                        d_loss_real.item(),
                                                                                                        d_loss_fake.item(),
                                                                                                        g_loss_fake.item()))
                logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                                                                                                        d_loss_robj.item(),
                                                                                                        d_loss_fobj.item(),
                                                                                                        g_loss_fobj.item()))
                logger.info("             ssim_loss: {:.4f}, pixel_loss: {:.4f}, feat_loss: {:.4f}".format(
                                                                                                        ssim_loss.item(), 
                                                                                                        pixel_loss.item(), 
                                                                                                        feat_loss.item()))
                logger.info("             rec_pixel_loss: {:.4f}, rec_feat_loss: {:.4f}".format(
                                                                                                        rec_pixel_loss.item(), 
                                                                                                        rec_feat_loss.item()))
                
                if use_bkg_net_D:
                    logger.info("             d2_out_real: {:.4f}, d2_out_fake: {:.4f}, g2_out_fake: {:.4f} ".format(d2_loss_real.item(), d2_loss_fake.item(), g2_loss_fake.item()))
                # logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))

            global_steps += 1


        #End of each epoch
        writer.add_scalar("epoch_d1_loss/d1_real_img", d1_real_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_real_obj", d1_real_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_fake_img", d1_fake_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_fake_obj", d1_fake_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d1_loss/d1_total", d1_all/steps_per_epochs, global_step=epoch+1)

        writer.add_scalar("epoch_d2_loss/d2_real_bkg", d2_real_bkg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d2_loss/d2_fake_bkg", d2_fake_bkg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_d2_loss/d2_total", d2_all/steps_per_epochs, global_step=epoch+1)

        writer.add_scalar("epoch_g_loss/g_fake_img", g_fake_img/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_fake_obj", g_fake_obj/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_fake_bkg", g_fake_bkg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_l1", g_l1/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_vgg", g_vgg/steps_per_epochs, global_step=epoch+1)
        writer.add_scalar("epoch_g_loss/g_ssim", g_ssim/steps_per_epochs, global_step=epoch+1)
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
                    [fire_images, non_fire_images, non_fire_crops], label, bbox, _ = data
                    label, bbox = label[0:1].long().cuda().unsqueeze(-1), bbox[0:1].float()
                    fire_images = fire_images[0:1].cuda()
                    non_fire_images = non_fire_images[0:1].cuda()
                    non_fire_crops = non_fire_crops[0:1].cuda()
                    if normalized:
                        z_obj = torch.from_numpy(truncted_random(z_obj_dim=z_obj_random_dim, num_o=max_num_obj, thres=2.0)).float().cuda()
                    else:
                        z_obj = torch.rand(fire_images.size(0), max_num_obj, z_obj_random_dim).cuda()
                    break

            #Network() processing    
            netG.eval()
            netD.eval()
            fake_images, stage_bbox_masks = netG.forward(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
            if not normalized:
                fake_images = fake_images*0.5+0.5
            g_out_fake, _ = netD(fake_images, bbox.cuda(), label)
            g_out_real, _ = netD(fire_images, bbox.cuda(), label)

            #Img_show() processing
            #1) fake-fire
            if not normalized:
                fake_images = (fake_images-0.5)*2
            fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            fake_images = np.array(fake_images*255, np.uint8)
            g_out_fake  = g_out_fake[0,0].cpu().detach()
            g_out_fake = -1 if g_out_fake>1 else 1 if g_out_fake<-1 else -torch.round(g_out_fake,decimals=2)
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
            stage_bbox_masks = stage_bbox_masks[0].cpu().detach().numpy()   #shape [3 objs, 128, 128]
            #4) soft-mask
            soft_mask = normalize_minmax(np.clip(np.sum(stage_bbox_masks[0:2], axis=0), a_min=0, a_max=1), [0, 255], input_range=[0,1])
            soft_mask = draw_layout(label, bbox, [256,256], class_names, input_img=soft_mask, topleft_name='Soft seg-mask')
            #5) hard-mask
            hard_mask = np.array(np.any(stage_bbox_masks[0:2]>args.seg_mask_thresh, axis=0), dtype=np.uint8)
            hard_mask = normalize_minmax(hard_mask, [0, 255], input_range=[0,1])
            hard_mask = draw_layout(label, bbox, [256,256], class_names, input_img=hard_mask, topleft_name='Hard seg-mask')

            output_images = combine_images([fire_images, non_fire_images, fake_images, soft_mask, hard_mask], [256,256])

            cv2.imwrite(args.out_path+"samples/"+ 'G_epoch_%d.png'%(epoch+1), cv2.cvtColor(output_images.astype(np.uint8), cv2.COLOR_RGB2BGR))
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str,   default="train",            help="processing phase: train, test")
    parser.add_argument('--dataset',        type=str,   default="fire3",            help="dataset used for training")
    parser.add_argument('--img_size',       type=int,   default=128,                help="training input image size. Default: 128x128")
    parser.add_argument('--batch_size',     type=int,   default=16,                 help="training batch size. Default: 8")
    parser.add_argument('--total_epoch',    type=int,   default=200,                help="numer of total training epochs")
    parser.add_argument('--g_lr',           type=float, default=0.0001,             help="learning rate of generator")
    parser.add_argument('--d_lr',           type=float, default=0.0001,             help="learning rate of discriminator")
    parser.add_argument('--out_path',       type=str,   default="./outputs/",       help="path to output files")
    parser.add_argument('--num_workers',    type=int,   default=0,                  help="Number of workers for dataset parallel processing")
    args = parser.parse_args()
    main(args)

