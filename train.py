import os, sys
import argparse, logging, time, datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

import cv2
import colorsys

from data.cocostuff_loader import CocoSceneGraphDataset
from data.data_loader import FireDataset
from model.resnet_generator import ResnetGenerator128
from model.rcnn_discriminator import CombineDiscriminator128
from utils.util import VGGLoss



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
    
    lamb_obj = 1.0
    lamb_img = 0.1
    img_size = (args.img_size, args.img_size)
    g_lr, d_lr = args.g_lr, args.d_lr
    
    if args.dataset == 'coco':
        train_img_dir =     os.path.join(dataset_path, "train2017")
        instances_json =    os.path.join(dataset_path, "annotations/instances_train2017.json")
        stuff_json =        os.path.join(dataset_path, "annotations/stuff_train2017.json")
        num_classes = 184
        num_obj = 8
        z_dim = 128

        train_data = CocoSceneGraphDataset(image_dir=train_img_dir,
                                       instances_json=instances_json,
                                       stuff_json=stuff_json,
                                       stuff_only=True, image_size=img_size, left_right_flip=True)

    elif args.dataset == 'fire':
        train_img_dir   = os.path.join(dataset_path, "images")
        classname_file  = os.path.join(dataset_path, "class_names.txt")
        num_classes = 4
        num_obj = 8
        z_dim = 128
        train_data = FireDataset(image_dir=train_img_dir, classname_file=classname_file,
                                image_size=img_size, left_right_flip=True)

        with open("./datasets/fire/class_names.txt", "r") as f:
            class_names = f.read().splitlines()

    #Training pre-steps: dataloader, model, optimizer
    #Data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=0)#num_workers=args.num_workers)
    
    #Model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()
    netD = CombineDiscriminator128(num_classes=num_classes).cuda()

    #Optimizers
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr*0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]
    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.mkdir(os.path.join(args.out_path, 'model/'))

    writer = SummaryWriter(os.path.join(args.out_path, 'log'))
    
    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)


    #Read from here
    start_time = time.time()
    vgg_loss = VGGLoss()                #average L2-norm between reference and reconstructed
    l1_loss = nn.L1Loss()
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()

        for idx, data in enumerate(dataloader):
            real_images, label, bbox = data
            real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float()      #keep bbox in cpu --> make input of netG,netD in gpu

            # update D network
            netD.zero_grad()
            d_out_real, d_out_robj = netD(real_images, bbox.cuda(), label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()     #[batch, num_obj, 128]
            fake_images = netG(z_img=None, z_obj=z, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))
            d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox.cuda(), label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj = netD(fake_images, bbox.cuda(), label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                
                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss
                g_loss.backward()
                g_optimizer.step()

            if (idx+1) % 250 == 0:
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
                logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))

                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4), epoch*len(dataloader) + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4), epoch*len(dataloader) + idx + 1)

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch+1)))
            
            for idx, data in enumerate(dataloader):
                if idx == 0:
                    real_images, label, bbox = data
                    real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float()
                    z_obj = torch.from_numpy(truncted_random(num_o=8, thres=2.0)).float().cuda()
                    z_im = torch.from_numpy(truncted_random(num_o=1, thres=2.0)).view(1, -1).float().cuda()
                    break
                
            netG.eval()
            fake_images = netG.forward(z_img=z_im, z_obj=z_obj, bbox=bbox.cuda(), class_label=label.squeeze(dim=-1))                 #bbox: 8x4 (coors), z_obj:8x128 random, z_im: 128
            fake_images = fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5
            fake_images = np.array(fake_images*255, np.uint8)
            layout_img = draw_layout(label, bbox, [256,256], class_names)

            cv2.imwrite("./samples/"+ 'image_G_%d.png'%(epoch+1), cv2.resize(fake_images, (256, 256)))
            cv2.imwrite("./samples/"+ 'layout_G_%d.png'%(epoch+1), cv2.resize(layout_img, (256, 256)))


def draw_layout(label, bbox, size, class_names):
    temp_img = np.zeros([size[0]+50,size[1]+50,3])
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
        x,y = x+25, y+25
        class_name = class_names[label[i]]
        cv2.rectangle(temp_img, (x, y), (x + width, y + height), bbox_color, 1)  # (0, 255, 0) is the color (green), 2 is the thickness
        cv2.putText(temp_img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
   
    return temp_img


def truncted_random(num_o=8, thres=1.0):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = np.random.normal()
    return z

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,   default="fire",             help="dataset used for training")
    parser.add_argument('--img_size',       type=int,   default=128,                help="training input image size. Default: 128x128")
    parser.add_argument('--batch_size',     type=int,   default=32,                  help="training batch size. Default: 8")
    parser.add_argument('--total_epoch',    type=int,   default=200,                help="numer of total training epochs")
    parser.add_argument('--g_lr',           type=float, default=0.0001,             help="learning rate of generator")
    parser.add_argument('--d_lr',           type=float, default=0.0001,             help="learning rate of discriminator")
    parser.add_argument('--out_path',       type=str,   default="./outputs/",       help="path to output files")
    parser.add_argument('--num_workers',    type=int,   default=1,                  help="Number of workers for dataset parallel processing")
    args = parser.parse_args()
    main(args)

