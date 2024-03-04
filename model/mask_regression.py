
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskRegressNet(nn.Module):
    def __init__(self, obj_feat_dim=128, mask_size=16, map_size=64):
        super(MaskRegressNet, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size
        
        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat_dim, 256*4*4))  #z_obj_dim -> 256x4x4 (4096)

        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)))
        conv1.append(nn.InstanceNorm2d(256))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)))
        conv2.append(nn.InstanceNorm2d(256))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=3, padding=1)))
        conv3.append(nn.InstanceNorm2d(256))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox):
        """
        input params:
            - obj_feat:     (b*num_o, feat_dim) --> latent_vector of objects
            - bbox:   (b, num_o, 4)
        return:
            - bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        x = self.fc(obj_feat)                           #linear mapping: [b*o, 128+180] --> [b*o, 256*4*4]
        x = self.conv1(x.view(b*num_o, 256, 4, 4))      #output [b*o, 256, 4, 4]     
        x = F.interpolate(x, size=8, mode='bilinear')   #output [b*o, 256, 8, 8]     
        x = self.conv2(x)                               #output [b*o, 256, 8, 8]     
        x = F.interpolate(x, size=16, mode='bilinear')  #output [b*o, 256, 16, 16]     
        x = self.conv3(x)                               #output [b*o, 1, 16, 16]     
        x = x.view(b, num_o, self.mask_size, self.mask_size)            
        #above: an encoding of z_obj_random+z_obj_class
        #not have information bbox
        bbmap = self._masks_to_layout(bbox, x, self.map_size).view(b, num_o, self.map_size, self.map_size)
        return bbmap


    def _masks_to_layout(self, boxes, masks, H, W=None):
        """
        Input params:
            - boxes: Tensor of shape [b, num_o, 4] giving bboxes in format [xm, ym, w, h] with [0, 1] coordinate space
            - masks: Tensor of shape [b, num_o, M, M] giving binary masks for each object
            - H, W: size of the output image
        Return:
            - out: Tensor of shape [N, num_o, H, W]
        """
        b, num_o, _ = boxes.size()
        M = masks.size(2)
        assert masks.size() == (b, num_o, self.mask_size, self.mask_size)
        if W is None:
            W = H
        
        #an encoding of bboxes
        grid = self._boxes_to_grid(boxes.view(b*num_o, -1), H, W).float().cuda(device=masks.device)     

        img_in = masks.float().view(b*num_o, 1, M, M)   #from conv
        #input: [bo, 1, 16, 16], [bo, 64, 64, 2]
        # output[bo,:,64,64] is interpolated from size-2 vector grid[bo,h,w]
        sampled = F.grid_sample(img_in, grid, mode='nearest')
        #only have values at bbox_pos + information of class
        return sampled.view(b, num_o, H, W)
        

    def _boxes_to_grid(self, boxes, H, W):
        """
        Input params:
            - boxes: FloatTensor of shape [num_bo, 4] giving bboxees in format [xm, ym, w, h] in coordiate [0, 1]
            - H, W: scalars giving the size of output
        Returns:
            - FloatTensor of shape (num_o, H, W, 2)
        """
        num_bo = boxes.size(0)
        boxes = boxes.view(num_bo, 4, 1, 1)
        
        # All these are (num_bo, 1, 1)
        xm, ym = boxes[:, 0], boxes[:, 1]
        ww, hh = boxes[:, 2], boxes[:, 3]

        X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)      #create line_space [0-1, W values]
        Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

        X = (X - xm) / ww           #([1,1,W] - [B,1,1]) / [B,1,1] --> [B, 1, W]
        Y = (Y - ym) / hh

        #Expand
        X = X.expand(num_bo, H, W)
        Y = Y.expand(num_bo, H, W)
        grid = torch.stack([X, Y], axis=3)  # (num_bo, H, W, 2)

        #Values at bbox_pos in range [0, 1], others <0 or >1
        grid = grid.mul(2).sub(1)   #Transform grid to scale [-1, 1] for next grid_sample()
        return grid