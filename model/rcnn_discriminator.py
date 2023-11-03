import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign as ROIAlign



class CombineDiscriminator128(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator128, self).__init__()
        self.obD = ResnetDiscriminator128(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]       #w -> xmax
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]       #h -> ymax
        bbox = bbox * images.size(2)                        #convert [0,1] to [0, image_size]
        bbox = torch.cat((idx, bbox.float()), dim=2)        #dim-2 : [batch_id, xmin, ymin, xmax, ymax]
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)               #position of objects in bbox shape [b*o, bbox]
        bbox = bbox[idx]                                    #extract bbox and class_id
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)    #classify image, ROI Align, classify obj
        return d_out_img, d_out_obj
    
class ResnetDiscriminator128(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock(ch, ch*2, downsample=True)
        self.block3 = ResBlock(ch*2, ch*4, downsample=True)
        self.block4 = ResBlock(ch*4, ch*8, downsample=True)
        self.block5 = ResBlock(ch*8, ch*16, downsample=True)
        self.block6 = ResBlock(ch*16, ch*16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 8.0, int(0))

        self.block_obj3 = ResBlock(ch*2, ch*4, downsample=False)
        self.block_obj4 = ResBlock(ch*4, ch*8, downsample=False)
        self.block_obj5 = ResBlock(ch*8, ch*16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch*16))

    def forward(self, x, y=None, bbox=None):
        num_bbox = bbox.size(0)

        x = self.block1(x)      # 64x64x64
        x1 = self.block2(x)     # 32x32x128
        x2 = self.block3(x1)    # 16x16x256
        x = self.block4(x2)     # 8x8x512
        x = self.block5(x)      # 4x4x1024
        x = self.block6(x)      # 4x4x1024
        x = self.activation(x)  # [batch, 1024, 4, 4]
        x = torch.sum(x, dim=(2, 3))    #[batch, 1024]
        out_im = self.l7(x)     # [batch, 1]

        # ROI Alignment
        # seperate small and large bbox
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64) #bbox < 64x64 --> small, other --> large
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]  #bbox
        y_l, y_s = y[~s_idx], y[s_idx]              #class_id

        obj_feat_s = self.block_obj3(x1)            #32x32x256
        obj_feat_s = self.block_obj4(obj_feat_s)    #32x32x512
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj4(x2)            #16x16x512
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj5(obj_feat)        #[num_obj, 1024, 4, 4]
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))  #[num_obj, 1024]
        out_obj = self.l_obj(obj_feat)              #[num_obj, 1]
        out_obj = out_obj + torch.sum(self.l_y(y).view(num_bbox, -1) * obj_feat.view(num_bbox, -1), dim=1, keepdim=True)    #[num_obj, 1]

        return out_im, out_obj


#Residual block: use 2 activation in main branch, conv before downsampling    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=pad), eps=1e-4)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=ksize, padding=pad), eps=1e-4)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0), eps=1e-4)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


#Residual block: not using activation function in the 2nd conv (main branch), shortcut: conv after downsampling
class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=pad), eps=1e-4)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=ksize, padding=pad), eps=1e-4)
        self.conv_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0), eps=1e-4)
        self.activation = nn.ReLU()
        self.downsample = downsample
    
    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.conv_sc(x)
    
    def forward(self, in_feat):
        x = in_feat
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)
