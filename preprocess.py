import numpy as np
from matplotlib import pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.autograd import Variable
import torch.optim as optim
import cv2
from gc_net import *
from util import writePFM
from PIL import Image
temp1 = cv2.cvtColor(cv2.imread('./tp1.png'), cv2.COLOR_BGR2GRAY)


def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)

def preprocessForReal(img):
    kernel = np.ones((1,1),np.uint8)
    img = hist_match(img,temp1).astype("uint8")
    img = cv2.GaussianBlur(img,(3,3),0)
    ret3,img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    img = cv2.erode(img,kernel,iterations = 1)
    img = cv2.dilate(img,kernel,iterations = 1)
    return img

def prepareGCnet(imgl,imgr,isGray):
    tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    maxdisp = 64
    oh = imgl.shape[0]
    ow = imgl.shape[1]
    h = 512
    w = 384
    net = GcNet(h,w,maxdisp)
    net=torch.nn.DataParallel(net).cuda()
    if imgl.shape[2]==3:
        imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    imgl = hist_match(imgl,imgr).astype("uint8")
    imgl = cv2.resize(imgl, (384, 512), interpolation=cv2.INTER_CUBIC)
    imgr = cv2.resize(imgr, (384, 512), interpolation=cv2.INTER_CUBIC)
    if isGray is True:
        imgl = preprocessForReal(imgl)
        imgr = preprocessForReal(imgr)
    
    imageL = np.expand_dims(imgl, axis=-1)
    imageR = np.expand_dims(imgr, axis=-1)
    imL = tsfm(imageL)
    imR = tsfm(imageR)
    imL = imL.unsqueeze(0)
    imR = imR.unsqueeze(0)
    checkpoint = torch.load('./checkpoint/ckpt_1_16.t7')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    loss_mul_list_test = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([1, 1, h, w]) * d)).cuda() #we need to change here for bmp(320x428) image
        loss_mul_list_test.append(loss_mul_temp)
    loss_mul_test = torch.cat(loss_mul_list_test, 1)

    with torch.no_grad():
        print(imL.size())
        result=net(imL,imR)

    disp=torch.sum(result.mul(loss_mul_test),1)
    im=disp.data.cpu().numpy().astype('uint8')
    im=np.transpose(im,(1,2,0))
    im = im[:,:,0]
    im = cv2.resize(im, (ow, oh), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('test.png',im,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    im_pfm=disp.data.cpu().numpy().astype('float32')
    im_pfm=np.transpose(im_pfm,(1,2,0))
    im_pfm = im_pfm[:,:,0]
    im_pfm = cv2.resize(im_pfm, (ow, oh), interpolation=cv2.INTER_CUBIC)
    return im_pfm

#result = prepareGCnet(cv2.imread(dir_syn_l),cv2.imread(dir_syn_r),flag)