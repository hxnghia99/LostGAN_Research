import torch
from torchvision import models
import torch.nn as nn
import cv2
import colorsys
import numpy as np
from torchmetrics.image.inception import InceptionScore

# VGG Features matching
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    



def draw_layout(label, bbox, size, class_names, input_img=None, D_class_score=None):
    if input_img is None:
        temp_img = np.zeros([size[0]+50,size[1]+50,3])
    else:
        num_c = input_img.shape[2]
        temp_img = np.zeros([size[0]+50,size[1]+50,num_c])
        input_img = np.expand_dims(cv2.resize(input_img, size), axis=-1) if num_c==1 else cv2.resize(input_img, size)
        temp_img[25:25+size[0], 25:25+size[1],:] = input_img
     
    bbox = (bbox[0]*size[0]).numpy().astype(np.int32)
    label = label[0]
    num_classes = len(class_names)

    rectangle_hsv_tuples     = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    label_hsv_tuples         = [(1.0 * x / num_classes, 1., 1.) for x in range(int(num_classes/2), num_classes)] 
    label_hsv_tuples        += [(1.0 * x / num_classes, 1., 1.) for x in range(0, int(num_classes/2))]                                
    rand_rectangle_colors    = list(map(lambda x: colorsys.hsv_to_rgb(*x), rectangle_hsv_tuples))
    rand_rectangle_colors    = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_rectangle_colors))
    rand_text_colors         = list(map(lambda x: colorsys.hsv_to_rgb(*x), label_hsv_tuples))
    rand_text_colors         = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rand_text_colors))
    
    for i in range(len(bbox)):
        label_color = rand_text_colors[label[i]]
        if num_classes < 5:
            if label[i] == 1:
                label_color = (255, 0, 0)
            elif label[i] == 2:
                label_color = (0, 0, 255)
            else:
                label_color = (0, 255, 0)
        
        x,y,width,height = bbox[i]
        x,y = x+25, y+25
        class_name = class_names[label[i]]
        cv2.rectangle(temp_img, (x, y), (x + width, y + height), label_color, 1)  # (0, 255, 0) is the color (green), 2 is the thickness
        cv2.putText(temp_img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

    if D_class_score is not None:
        if D_class_score>=0:
            D_class_text = "Real: {}%".format(D_class_score*100)
        else:
            D_class_text = "Fake: {}%".format(D_class_score*100)
        cv2.putText(temp_img, D_class_text, (25,size[1]+50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    return temp_img


def create_continuous_map(height, width):
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height//2, width//2
    distances = np.sqrt((y-center_y)**2 + (x-center_x)**2)
    normalized_distances = distances / np.max(distances)
    return normalized_distances


def IS_compute_np(img: np.array):
    img = torch.as_tensor(img)
    img = img.permute((2,0,1))
    img = img.unsqueeze(dim=0)
    inception = InceptionScore()
    inception.update(img)
    mean, std_dev = inception.compute()
    return mean, std_dev