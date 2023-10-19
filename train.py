import os, sys
import argparse, logging, time, datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


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
    vgg_loss = VGGLoss()
    l1_loss = nn.L1Loss()
    # vgg_loss = nn.DataParallel(vgg_loss)
    # l1_loss = nn.DataParallel(nn.L1Loss())
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()

        for idx, data in enumerate(dataloader):
            real_images, label, bbox = data
            real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float().cuda()

            # update D network
            netD.zero_grad()
            real_images, label = real_images.cuda(), label.long().cuda()
            d_out_real, d_out_robj = netD(real_images, bbox, label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()     #[batch, num_obj, 128]
            fake_images = netG(z_img=None, z_obj=z, bbox=bbox, class_label=label.squeeze(dim=-1))
            d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox, label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj = netD(fake_images, bbox, label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                
                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss
                g_loss.backward()
                g_optimizer.step()

            if (idx+1) % 500 == 0:
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



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,   default="fire",             help="dataset used for training")
    parser.add_argument('--img_size',       type=int,   default=128,                help="training input image size. Default: 128x128")
    parser.add_argument('--batch_size',     type=int,   default=16,                  help="training batch size. Default: 8")
    parser.add_argument('--total_epoch',    type=int,   default=200,                help="numer of total training epochs")
    parser.add_argument('--g_lr',           type=float, default=0.0001,             help="learning rate of generator")
    parser.add_argument('--d_lr',           type=float, default=0.0001,             help="learning rate of discriminator")
    parser.add_argument('--out_path',       type=str,   default="./outputs/",       help="path to output files")
    parser.add_argument('--num_workers',    type=int,   default=1,                  help="Number of workers for dataset parallel processing")
    args = parser.parse_args()
    main(args)

