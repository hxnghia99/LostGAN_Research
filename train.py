import os, sys
import argparse, logging, time, datetime

from torch import Tensor
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
# from pytorch_ssim import ssim
import numpy as np

import cv2
from piqa import SSIM

from data.cocostuff_loader import CocoSceneGraphDataset
from data.data_loader import FireDataset
from model.resnet_generator import ResnetGenerator128
from model.rcnn_discriminator import CombineDiscriminator128, BkgResnetDiscriminator128
from utils.util import VGGLoss, draw_layout

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
    #Configuration setup
    dataset_path =      os.path.join("./datasets", args.dataset)
    mode = args.mode
    
    use_noised_input = False
    weight_map_type = 'extreme'
    max_num_obj = 2                 #if max_obj=2, get only first fire and smoke
    get_first_fire_smoke = True if max_num_obj==2 else False
    
    use_bkg_net_D = False

    z_dim = 128
    lamb_obj = 1.0
    lamb_img = 0.1
    img_size = (args.img_size, args.img_size)
    g_lr, d_lr = args.g_lr, args.d_lr
    
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
        train_fire_img_dir   = os.path.join(dataset_path, mode+"_images_A")
        train_non_fire_img_dir   = os.path.join(dataset_path, mode+"_images_B")
        classname_file  = os.path.join(dataset_path, "class_names.txt")
        num_classes = 3
        
        train_data = FireDataset(fire_image_dir=train_fire_img_dir, non_fire_image_dir=train_non_fire_img_dir, 
                                classname_file=classname_file,
                                image_size=img_size, left_right_flip=True,
                                max_objects_per_image=max_num_obj,
                                get_first_fire_smoke=get_first_fire_smoke,
                                use_noised_input=use_noised_input,
                                weight_map_type=weight_map_type)

        with open(os.path.join(dataset_path, "class_names.txt"), "r") as f:
            class_names = f.read().splitlines()

    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.num_workers)

    #Model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()
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
                gen_parameters += [{'params': [value], 'lr': g_lr*0.1}]
    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    #disc: fire/fake-fire
    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    if use_bkg_net_D:
        #bkg: non-fire/fake-non-fire
        dis2_parameters = []
        for key, value in dict(netD2.named_parameters()).items():
            if value.requires_grad:
                dis2_parameters += [{'params': [value], 'lr': d_lr}]
        d2_optimizer = torch.optim.Adam(dis2_parameters, betas=(0, 0.999))

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.mkdir(os.path.join(args.out_path, 'model/'))
    if not os.path.exists(os.path.join(args.out_path, 'samples/')):
        os.mkdir(os.path.join(args.out_path, 'samples/'))

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

        for idx, data in enumerate(dataloader):
            [fire_images, non_fire_images, non_fire_crops], label, bbox, weight_map = data
            fire_images, label, bbox, weight_map = fire_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float(), weight_map.float().cuda()      #keep bbox in cpu --> make input of netG,netD in gpu
            non_fire_images, non_fire_crops = non_fire_images.cuda(),  non_fire_crops.cuda()
            # update D network
            netD.zero_grad()
            #real image+objects
            d_out_real, d_out_robj = netD(fire_images, bbox.cuda(), label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            #fake image+objects
            z_obj = torch.randn(fire_images.size(0), max_num_obj, z_dim).cuda()     #[batch, num_obj, 128]
            fake_images, _ = netG(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))
            d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox.cuda(), label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_optimizer.step()

            if use_bkg_net_D:
                #update D2 network
                netD2.zero_grad()
                #real bkg
                d2_out_real = netD2(non_fire_crops)
                d2_loss_real = torch.nn.ReLU()(1.0 - d2_out_real).mean()
                #fake bkg
                d2_out_fake = netD2(fake_images.detach()*weight_map)
                d2_loss_fake = torch.nn.ReLU()(1.0 + d2_out_fake).mean()

                d2_loss = d2_loss_real + d2_loss_fake
                d2_loss.backward()
                d2_optimizer.step()

            # update G network
            if (idx % 1) == 0:      
                netG.zero_grad()
                g_out_fake, g_out_obj = netD(fake_images, bbox.cuda(), label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                
                if use_bkg_net_D:
                    g2_out_fake = netD2(fake_images*weight_map)
                    g2_loss_fake = - g2_out_fake.mean()

                ssim_loss = ssim((fake_images*0.5+0.5)*weight_map, (non_fire_images*0.5+0.5)*weight_map)

                pixel_loss = l1_loss(fake_images*weight_map, non_fire_images*weight_map).mean()
                feat_loss = vgg_loss(fake_images*weight_map, non_fire_images*weight_map).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + ssim_loss + pixel_loss + feat_loss
                if use_bkg_net_D:
                    g_loss += g2_loss_fake
                g_loss.backward()
                g_optimizer.step()

            if (idx+1) % 100 == 0:
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
                                                                                                        g_loss_obj.item()))
                logger.info("             ssim_loss: {:.4f}, pixel_loss: {:.4f}, feat_loss: {:.4f}".format(ssim_loss.item(), pixel_loss.item(), feat_loss.item()))
                if use_bkg_net_D:
                    logger.info("             d2_out_real: {:.4f}, d2_out_fake: {:.4f}, g2_out_fake: {:.4f} ".format(d2_loss_real.item(), d2_loss_fake.item(), g2_loss_fake.item()))
                # logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch+1)))
            
            for idx, data in enumerate(dataloader):
                if idx == 0:
                    [fire_images, non_fire_images, non_fire_crops], label, bbox, weight_map = data
                    label, bbox, weight_map = label[0:1].long().cuda().unsqueeze(-1), bbox[0:1].float(), weight_map[0:1].float().cuda()
                    fire_images = fire_images[0:1].cuda()
                    non_fire_images = non_fire_images[0:1].cuda()
                    non_fire_crops = non_fire_crops[0:1].cuda()
                    z_obj = torch.from_numpy(truncted_random(num_o=max_num_obj, thres=2.0)).float().cuda()
                    break
                
            netG.eval()
            fake_images, _ = netG.forward(z_img=non_fire_images, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
            fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            fake_images = np.array(fake_images*255, np.uint8)
            fake_images = draw_layout(label, bbox, [256,256], class_names, fake_images)
            
            fire_images = fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            fire_images = np.array(fire_images*255, np.uint8)
            fire_images = draw_layout(label, bbox, [256,256], class_names, fire_images)

            non_fire_images = non_fire_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            non_fire_images = np.array(non_fire_images*255, np.uint8)
            non_fire_images = draw_layout(label, bbox, [256,256], class_names, non_fire_images)

            cv2.imwrite(args.out_path+"samples/"+ 'G_%d_real-fire.png'%(epoch+1), cv2.resize(cv2.cvtColor(fire_images.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
            cv2.imwrite(args.out_path+"samples/"+ 'G_%d_fake-fire.png'%(epoch+1), cv2.resize(cv2.cvtColor(fake_images.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
            cv2.imwrite(args.out_path+"samples/"+ 'G_%d_non-fire.png'%(epoch+1), cv2.resize(cv2.cvtColor(non_fire_images.astype(np.uint8), cv2.COLOR_RGB2BGR), (256, 256)))
            netG.train()


def truncted_random(num_o=8, thres=1.0):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = np.random.normal()
    return z

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str,   default="train",            help="processing phase: train, test")
    parser.add_argument('--dataset',        type=str,   default="fire3",            help="dataset used for training")
    parser.add_argument('--img_size',       type=int,   default=128,                help="training input image size. Default: 128x128")
    parser.add_argument('--batch_size',     type=int,   default=16,                 help="training batch size. Default: 8")
    parser.add_argument('--total_epoch',    type=int,   default=100,                help="numer of total training epochs")
    parser.add_argument('--g_lr',           type=float, default=0.0001,             help="learning rate of generator")
    parser.add_argument('--d_lr',           type=float, default=0.0001,             help="learning rate of discriminator")
    parser.add_argument('--out_path',       type=str,   default="./outputs/",       help="path to output files")
    parser.add_argument('--num_workers',    type=int,   default=0,                  help="Number of workers for dataset parallel processing")
    args = parser.parse_args()
    main(args)

