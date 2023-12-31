import torch
import torch.nn as nn
import torch.nn.functional as F


#nn.BatchNorm2D
class SynchronizedBatchNorm2d(nn.BatchNorm2d):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


#Adaptive instance normalization
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1) -> None:
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        #buffer: not considered as model parameters 
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        #projection layer
        self.weight_proj    = nn.Linear(num_w, num_features)       #linear-mapping from 1st input to 2nd input
        self.bias_proj      = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        b, c = x.size(0), x.size(1)     #batch, channel
        running_mean    = self.running_mean.repeat(b)
        running_var     = self.running_var.repeat(b)

        weight, bias = self.weight_proj(w).contiguous().view(-1) + 1, self.bias_proj(w).contiguous().view(-1)   #.contiguous() rearranges data in a contiguous block: not change data/shape

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b*c, *x.size()[2:])     #shape[1, b*c, height, width]
        out = F.batch_norm(x_reshaped, running_mean, running_var, weight, bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')' #return the string represeting the name of object


#Spatial Adaptive Instance Normalization
class SpatialAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(SpatialAdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        #buffer: not considered as model parameters
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("runnnig_var", torch.ones(num_features))

        #projection layer
        self.weight_proj    = nn.Linear(num_w, num_features)        #mapping dimension: num_w -> num_features
        self.bias_proj      = nn.Linear(num_w, num_features)

    def forward(self, x, w, bbox):
        b, o, _, _ = bbox.sise()
        _, c, _, _ = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        #calculate weight and bias
        weight, bias = self.weight_proj(w), self.bias_proj(w)
        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) \
                 + 1
        bias =  torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) 
        
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b*c, *x.size()[2:])     #shape[1, b*c, height, width]
        out = F.batch_norm(x_reshaped, running_mean, running_var, weight, bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])


#Adaptive Batch Normalization
class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaptiveBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        #projection layer
        self.weight_proj    = nn.Linear(num_w, num_features)        #num_w->in_channels
        self.bias_proj      = nn.Linear(num_w, num_features)    

    def forward(self, x, w):                                    #w: the information of z_obj + label_embedding
        self._check_input_dim(x)    #check if x has 4 dimensions
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked +=1
            if self.momentum is None:       #use cummulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum

        #output feature map
        output = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)

        size = output.size()
        weight, bias = self.weight_proj(w)+1, self.bias_proj(w)     
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        
        return weight * output + bias
    

#Spatial Adaptive Batch Normalization
class SpatialAdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(SpatialAdaptiveBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        #projection layer
        self.weight_proj    = nn.Linear(num_w, num_features)        #num_w->in_channels
        self.bias_proj      = nn.Linear(num_w, num_features)   

    def forward(self, x, vector, bbox):
        """input arguments:
            - x:        input feature map (b,c,h,w)
            - vector:   latent vector (b*o, dim_w)
            - bbox:     bbox map (b, o, h, w)
        """
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked +=1
            if self.momentum is None:   #use cummulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:                       #use exponential moving average
                exponential_average_factor = self.momentum
            
        #output feature map
        output = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()
        if bw != w or bh != h:
            bbox = F.interpolate(bbox, size=(h,w), mode="bilinear")
        #calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)
        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) \
                 + 1
        bias =  torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) 
        
        return weight * output + bias
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
    



class SpatialAdaptiveSynBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveSynBatchNorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        output = self.batch_norm2d(x)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'