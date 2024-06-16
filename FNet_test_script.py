from models import FNet
from warpingOperator import *
import torch
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


imgs = np.load("C:\\Users\\Jatin\\Desktop\\Engineering\\ISRO Internship\\WORK\\NEW_ARCHITECTURE\\HDR-DSP-SR-main\\Dataset\\test\\15\\testLR.npy").astype(np.float32)
ratios = np.load("C:\\Users\\Jatin\\Desktop\\Engineering\\ISRO Internship\\WORK\\NEW_ARCHITECTURE\\HDR-DSP-SR-main\\Dataset\\test\\15\\testRatio.npy").astype(np.float32)


imgs = torch.from_numpy(imgs).to(device)
ratios = torch.from_numpy(ratios).to(device)


print(imgs.shape, ratios.shape)

i = 0
j = 2

img_ind = 4

sample1 = imgs[img_ind,i:i+1,:,:]/ratios[img_ind,i:i+1]
sample2 = imgs[img_ind,j:j+1,:,:]/ratios[img_ind,j:j+1]
coupleImages = torch.cat((sample1, sample2), axis = 0)
coupleImages = coupleImages[None, : , : , :]

gaussian_filter = GaussianBlur(11, sigma=1).to(device)

b , _, h, w = coupleImages.shape

coupleImagesblur = gaussian_filter(coupleImages)


file_path = './TrainHistory/FNet_Pretrain_20000epochs_06-11-2024/'

checkpoints_files = os.listdir(file_path)
checkpoints_files = ['pretrained_Fnet.pth.tar'] + [file_path + i for i in checkpoints_files]
print(checkpoints_files)

FNet = FNet().float().to(device)

for checkpoint_path in checkpoints_files:  
    checkpoint = torch.load(checkpoint_path)

    FNet.load_state_dict(checkpoint['state_dictFnet']) #Load the pretrained Fnet

    # coupleImages = coupleImages.float().to(device)
    # print(coupleImages.shape)

    flow = FNet(coupleImagesblur)

    p=1 
    interpolation = 'bicubicTorch'
    TVLoss_weight = 0.003 #0.1 and 0.003 work 0.01 and 0.03 doesn't 

    warping = WarpedLoss(p, interpolation = interpolation)
    TVLoss = TVL1(TVLoss_weight=1)
    losstype = 'Detail'

    trainwarploss, _ = warping(coupleImagesblur[:,:1], coupleImagesblur[:,1:], flow, losstype)


    traintvloss = TVLoss(flow[...,2:-2,2:-2])
    trainloss = trainwarploss + TVLoss_weight*traintvloss   

    print(checkpoint_path + ' - TrainLoss = WarpLoss + TVFlow: {:.5f} = {:.5f} + {}*{:.5f}'.format(
                    trainloss, trainwarploss, TVLoss_weight, traintvloss))

print("Flow Shape : " , flow.shape)

# fig,ax = plt.subplots(1,2, figsize=(10,10))
# ax[0].imshow(sample1.detach().cpu().numpy()[0,:,:])
# ax[1].imshow(sample2.detach().cpu().numpy()[0,:,:])
# plt.show()

# fig,ax = plt.subplots(1,2, figsize=(10,10))
# ax[0].imshow(flow.detach().cpu().numpy()[0,0,:,:])
# ax[1].imshow (flow.detach().cpu().numpy()[0,1,:,:])
# plt.show()