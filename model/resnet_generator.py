import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_module import SpatialAdaptiveBatchNorm2d, SpatialAdaptiveInstanceNorm2d
from .mask_regression import MaskRegressNet

class ResnetGenerator128(nn.Module):
    def __init__(self, ch=64, z_obj_random_dim=128, z_obj_class_dim=180, num_classes=184, output_dim=3, mask_size=16, map_size= 64, input_dim=3):
        super(ResnetGenerator128, self).__init__()
        
        self.mask_size = mask_size
        self.map_size = map_size

        self.label_embedding = nn.Embedding(num_classes, embedding_dim=z_obj_class_dim)   #select embedding dimension as 180
        self.z_random_dim = z_obj_random_dim

        z_obj_dim = z_obj_random_dim + z_obj_class_dim
        self.fc = nn.utils.spectral_norm(nn.Linear(self.z_random_dim, 4*4*16*ch))

        #encoder path
        self.res0 = OptimizedBlock_en(input_dim, ch, downsample=True)
        self.res1 = ResBlock_en(ch, ch*2, downsample=True)
        self.res2 = ResBlock_en(ch*2, ch*4, downsample=True)
        self.res3 = ResBlock_en(ch*4, ch*8, downsample=True)
        self.res4 = ResBlock_en(ch*8, ch*16, downsample=True)
        self.res5 = ResBlock_en(ch*16, ch*16, downsample=False)
        self.activation = nn.ReLU()

        #decoder path
        self.res6 = ResBlock(ch*16, ch*16, upsample=True, num_w=z_obj_dim, num_classes=num_classes) #channel: 1024->1024
        self.res7 = ResBlock(ch*16, ch*8, upsample=True, num_w=z_obj_dim, num_classes=num_classes)  #channel: 1024->512
        self.res8 = ResBlock(ch*8, ch*4, upsample=True, num_w=z_obj_dim, num_classes=num_classes)  #channel: 512->256
        self.res9 = ResBlock(ch*4, ch*2, upsample=True, num_w=z_obj_dim, num_classes=num_classes, psp_module=True)  #channel: 256->128
        self.res10 = ResBlock(ch*2, ch*1, upsample=True, num_w=z_obj_dim, num_classes=num_classes, predict_mask=False)  #channel: 128->64

        self.final = nn.Sequential(nn.BatchNorm2d(ch),
                                   nn.ReLU(),
                                   nn.utils.spectral_norm(nn.Conv2d(ch, output_dim, kernel_size=3, padding=1), eps=1e-4),
                                   nn.Tanh())
                                   
        # mapping function
        mapping = list()
        self.mappping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, num_classes, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, num_classes, 1))
        
        self.sigmoid = nn.Sigmoid()
        
        self.mask_regress = MaskRegressNet(z_obj_dim, mask_size=mask_size, map_size=map_size)
        self.init_parameter()
        

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

    def _bbox_mask_generator(self, z_obj, bbox, H, W):
        b, o, _ = bbox.size()
        bo = b*o

        bbox_1 = bbox.float().view(-1, 4)       #[b*o, 4]
        xm, ym, ww, hh = bbox_1[:, 0], bbox_1[:,1], bbox_1[:,2], bbox_1[:,3]

        xm = xm.contiguous().view(bo, 1).expand(bo, H)      
        ww = ww.contiguous().view(bo, 1).expand(bo, H)
        ym = ym.contiguous().view(bo, 1).expand(bo, W)
        hh = hh.contiguous().view(bo, 1).expand(bo, W)

        X = torch.linspace(0, 1, steps=W).view(1, W).expand(bo, W).cuda(device=z_obj.device)
        Y = torch.linspace(0, 1, steps=H).view(1, H).expand(bo, H).cuda(device=z_obj.device)

        X = (X - xm) / ww       #([bo, W] - [bo, H])/[bo, H]
        Y = (Y - ym) / hh       #([bo, H] - [bo, W])/[bo, W]

        X_out_mask = ((X < 0) + (X > 1)).view(bo, 1, W).expand(bo, H, W)
        Y_out_mask = ((Y < 0) + (Y > 1)).view(bo, H, 1).expand(bo, H, W)

        out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
        return out_mask.view(b, o, H, W)
    

    def _batched_index_select(self, input, dim, index):
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)
        

    def forward(self, z_img, z_obj, bbox, class_label):
        b, o = z_obj.size(0), z_obj.size(1)
        class_label_embedding = self.label_embedding(class_label)

        z_obj = z_obj.view(b*o, -1)     #[b*o, 128]
        class_label_embedding = class_label_embedding.view(b*o, -1) #[b*o, 180]

        latent_vector = torch.concat((z_obj, class_label_embedding), dim=1).view(b, o, -1)  #[b, o, 128+180]
        latent_vector = self.mappping(latent_vector.view(b*o, -1))      #identity mapping at the momemt

        # preprocess bbox
        bbox_mask = self.mask_regress(latent_vector, bbox)      #encoding latent_vector+bbox --> [b, o, H(64), W(64)]
        
        if z_img is None:
            z_img = torch.randn((b, self.z_random_dim)).cuda()  #shape [b, 128]
        
        bbox_mask_random = self._bbox_mask_generator(z_obj, bbox, self.map_size, self.map_size)

        # #4x4
        # x = self.fc(z_img).view(b, -1, 4, 4)        #map [b,128]->[b, 4*4*16*64] --> [b, 1024, 4, 4]

        x = self.res0(z_img)      # 64x64x64
        x1 = self.res1(x)     # 32x32x128
        x2 = self.res2(x1)    # 16x16x256
        x = self.res3(x2)     # 8x8x512
        x = self.res4(x)      # 4x4x1024
        x = self.res5(x)      # 4x4x1024
        x = self.activation(x)  # [batch, 1024, 4, 4]

        #8x8    
        x, stage_mask = self.res6(x, latent_vector, bbox_mask)      #[b, 1024, 8, 8]
        
        #16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = self._batched_index_select(stage_mask, dim=1, index=class_label.view(b, o, 1, 1))  # [b, o, h, w]
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_random, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=class_label.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bbox_mask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res7(x, latent_vector, stage_bbox)

        #32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = self._batched_index_select(stage_mask, dim=1, index=class_label.view(b, o, 1, 1))  # [b, o, h, w]
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_random, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=class_label.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bbox_mask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res8(x, latent_vector, stage_bbox)

        #64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = self._batched_index_select(stage_mask, dim=1, index=class_label.view(b, o, 1, 1))  # [b, o, h, w]
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_random, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=class_label.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bbox_mask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res9(x, latent_vector, stage_bbox)

        #128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = self._batched_index_select(stage_mask, dim=1, index=class_label.view(b, o, 1, 1))  # [b, o, h, w]
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_random, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=class_label.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bbox_mask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, _ = self.res10(x, latent_vector, stage_bbox)

        # to RGB
        x = self.final(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, ksize=3, pad=1, upsample=False, num_w=128, num_classes=184, predict_mask=True, psp_module=False):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.mid_ch = mid_ch if mid_ch else out_ch
        
        #main branch
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, self.mid_ch, kernel_size=ksize, padding=pad), eps=1e-4)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(self.mid_ch, out_ch, kernel_size=ksize, padding=pad), eps=1e-4)
        self.b1 = SpatialAdaptiveBatchNorm2d(in_ch, num_w=num_w)
        self.b2 = SpatialAdaptiveBatchNorm2d(self.mid_ch, num_w=num_w)
        self.activation = nn.ReLU()

        #learnable_shortcut if upsamping or in_c!=out_c
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0), eps=1e-4)        #must use the same name c_sc --> save weight and load weight correctly

        #predict_mask of different class for checking
        self.predict_mask = predict_mask
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
                                               nn.Conv2d(100, num_classes, kernel_size=1))
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, num_classes, kernel_size=1, padding=0, bias=True))
                
    #Main branch
    def residual(self, in_feat, latent_vector, bbox):
        #apply Adaptive_Norm to input_feature
        x = in_feat
        x = self.b1(x, latent_vector, bbox)
        x = self.activation(x)
        #if upsampling
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        #apply Adaptive_Norm to mid_feature
        x = self.b2(x, latent_vector, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        return x
    #Short_cut
    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    #Usual case: produce feature map + mask
    def forward(self, in_feat, latent_vector, bbox):
        out_feat = self.residual(in_feat, latent_vector, bbox) + self.shortcut(in_feat)
        if self.predict_mask:
            mask = self.conv_mask(out_feat)
        else:
            mask = None
        return out_feat, mask
    


#Residual block: use 2 activation in main branch, conv before downsampling    
class ResBlock_en(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock_en, self).__init__()
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
class OptimizedBlock_en(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(OptimizedBlock_en, self).__init__()
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



class PSPModule(nn.Module):
    """Pyramid scene parsing network
    1) Use Adaptive Avg Pooling + conv --> output size: 1x1, 2x2, 3x3, 6x6
    2) Interpolate those into the same size of input feature map
    3) Apply a bottleneck
    """
    def __init__(self, in_features, out_features=512, sizes=(1,2,3,6)): #sizes: output_size of Adaptive Avg Pooling
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(in_features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),     
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
  
    def _make_stage(self, in_c, out_c, o_size):
        prior = nn.AdaptiveAvgPool2d(output_size=(o_size, o_size))          #32x20x20 --> 32x1x1 / 32x2x2
        conv =  nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        bn =    nn.BatchNorm2d(out_c)
        return nn.Sequential(prior, conv, bn, nn.ReLU())
    
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h,w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.concat(priors, 1))
        return bottle