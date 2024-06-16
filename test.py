""" Python script to train option J """
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from time import time


from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from models import EncoderNet, DecoderNet, FNet
from shiftandadd import shiftAndAdd, featureAdd, featureWeight

from warpingOperator import WarpedLoss, TVL1, base_detail_decomp, BlurLayer
import os
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
from do_shift_interp import do_shift_interp


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass



def flowEstimation(samplesLR, ME, device, gaussian_filter , warping, sr_ratio = 2):
    """
    Compute the optical flows from the other frames to the reference:
    samplesLR: Tensor b, num_im, h, w
    ME: Motion Estimator
    """

    b, num_im, h, w = samplesLR.shape

    samplesLRblur = gaussian_filter(samplesLR)


    samplesLR_0 = samplesLRblur[:,:1,...] #b, 1, h, w


    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1,1)  #b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  #b*num_im, h, w
    concat = torch.cat((samplesLRblur.unsqueeze(1), samplesLR_0.unsqueeze(1)), axis = 1) #b*(num_im), 2, h, w 
    flow = ME(concat.to(device)) #b*(num_im), 2, h, w 

    flow[::num_im] = 0

    warploss, _ = warping(samplesLRblur.unsqueeze(1).to(device),samplesLR_0.unsqueeze(1).to(device), flow, losstype = 'Detail')

    return flow.reshape(b, num_im, 2, h, w), warploss



def DeepSaaSuperresolve_weighted_base(samplesLR, flow, base, Encoder, Decoder, device, feature_mode, num_features = 64, sr_ratio=2, phase = 'training'):
    """
    samplesLR: b, num_im, h, w
    flow: b*(num_im-1), 2, h, w
    """
    nb_mode = len(feature_mode)

    #base, detail = base_detail_decomp(samplesLR[:,:1], gaussian_filter) #b, 1, h, w

    if phase == 'training':        
        samplesLR = samplesLR[:,1:,...].contiguous() #b, (num_im-1), h, w
        flow = flow[:,1:].contiguous()#.view(-1, 1, 2, h, w)
        base = base[:,1:].contiguous()
    b, num_im, h, w = samplesLR.shape

    #base = base.repeat(1,num_im, 1,1).view(-1,1,h,w)
    #base = warping.warp(base, flow.view(-1,2,h,w)) #b*num_im, 1, h, w

    samplesLR = samplesLR.view(-1,1,h,w)
    base = base.view(-1,1,h,w)

    
    inputEncoder = torch.cat((samplesLR, base), dim = 1)#samplesLR_detail.view(-1, 1, h, w) #b*(num_im-1), 1, h, w
    features = Encoder(inputEncoder) #b * (num_im-1), num_features, h, w
    features = features.view(-1, h, w) # b * num_im-1 *num_features, h, w

    dacc = featureWeight(flow.view(-1,2,h,w),sr_ratio=sr_ratio, device = device)
    flow = flow.contiguous().view(-1, 1, 2, h, w).repeat(1,num_features,1,1,1).view(-1,2, h, w) #b * num_im-1 * num_features, 2, h, w
    dadd = featureAdd(features, flow, sr_ratio=sr_ratio, device = device) #b * num_im * num_features, 2h, 2w

    dadd = dadd.view(b, num_im, num_features, sr_ratio*h, sr_ratio*w)
    dacc = dacc.view(b, num_im, 1, sr_ratio*h, sr_ratio*w)

    SR = torch.empty(b, 1+nb_mode*num_features, sr_ratio*h, sr_ratio*w)
    for i in range(nb_mode):
        if feature_mode[i] == 'Max':
            SR[:, i*num_features:(i+1)*num_features], _ = torch.max(dadd, dim = 1, keepdim = False)
        elif feature_mode[i] == 'Std':
            SR[:, i*num_features:(i+1)*num_features] = torch.std(dadd, dim = 1, keepdim = False)
        elif feature_mode[i] == 'Avg':
            #dadd = torch.sum(dadd, 1) #b, num_features, sr_ratioh, sr_ratiow
            dacc = torch.sum(dacc, 1)
            dacc[dacc == 0] = 1
            SR[:, i*num_features:(i+1)*num_features] = torch.sum(dadd, 1)/dacc
            SR[:, -1:] = dacc/15. #normalization/nb of frames
    SR = Decoder(SR.to(device)) #b, 1, sr_ration*h, sr_ratio*w
    #SR = torch.squeeze(SR, 1)

    return SR




class SkySatRealDataset_ME(Dataset):
    def __init__(self, path, augmentation = False,  phase = 'train', normalization = 3400., num_images = 15):
        # self.expotime = torch.from_numpy(np.load(os.path.join(path, '{}'.format(phase), str(num_images), '{}Ratio.npy'.format(phase)))[...,None,None])
        # self.data = torch.from_numpy(np.load(os.path.join(path, '{}'.format(phase), str(num_images), '{}LR.npy'.format(phase)))/normalization)


        self.expotime = torch.from_numpy(np.load(os.path.join(path, '{}'.format(phase), str(4), '{}Ratio.npy'.format(phase)))[...,None,None])
        self.data = torch.from_numpy(np.load(os.path.join(path, '{}'.format(phase), str(4), '{}LR.npy'.format(phase)))/normalization)

        self.data = self.data[:,:,:num_images]
        self.expotime = self.expotime[:num_images, :,:]


        self.len = self.expotime.size()[0]
        self.augmentation = augmentation
        self.num_images = num_images
    def transform(self, data):
        # Random crop
        h, w = data.shape[-2:]
        new_h, new_w = (64,64)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[:, top: top + new_h, left: left + new_w]

        if random.random() < 0.5:
            data = torch.flip(data, [-1])

        # Random vertical flipping
        if random.random() < 0.5:
            data = torch.flip(data, [-2])
            
        if random.random() < 0.5:
            k = random.sample([1,3], 1)[0]
            data = torch.rot90(data, k, [-2, -1])
        
        return data

    def __getitem__(self, idx):
        # data = self.data[idx]
        # expotime = self.expotime[idx]

        data = self.data
        expotime = self.expotime

        if self.augmentation:
            data = self.transform(data)
        return data, expotime

    def __len__(self):
        return self.len


def zoombase_weighted(LR_base, expotime, flow, device, warping):
    b, num_im, h, w = LR_base.shape

    LR_base = LR_base.view(-1,1,h,w)
    LR_base = warping.warp(LR_base, -flow.view(-1,2,h,w))
    LR_base = LR_base.view(b,num_im, h, w)
    LR_base = torch.mean(LR_base*expotime, 1, keepdim = True)/torch.mean(expotime, 1, keepdim = True)

    SR_base = torch.nn.functional.interpolate(LR_base, size = [2*h-1, 2*w-1], mode = 'bilinear', align_corners = True)
    SR_base = torch.cat((SR_base, torch.zeros(b,1,1,2*w-1).to(device)), dim = 2)
    SR_base = torch.cat((SR_base, torch.zeros(b,1,2*h,1).to(device)), dim = 3)
    return SR_base


def test(args):  
    criterion = nn.L1Loss()
    train_bs, val_bs, lr_fnet, factor_fnet, patience_fnet, lr_decoder, factor_decoder, patience_decoder, lr_encoder, factor_encoder, patience_encoder, num_epochs, warp_weight, TVflow_weight= args.train_bs, args.val_bs, args.lr_fnet, args.factor_fnet, args.patience_fnet,  args.lr_decoder, args.factor_decoder, args.patience_decoder, args.lr_encoder, args.factor_encoder, args.patience_encoder, args.num_epochs, args.warp_weight, args.TVflow_weight
    num_features, num_blocks = args.num_features, args.num_blocks
    sigma = args.sigma
    sr_ratio = args.sr_ratio
    feature_mode = args.feature_mode
    nb_mode = len(feature_mode)
    print(feature_mode)

    # For testing, we are going to be taking the pretrained FNet given by authors and pretrained FNet by us, to see how it performs for both. This is to compare if our FNet is being trained correctly.
    # Fnet_checkpoint = torch.load("pretrained_Fnet.pth.tar", map_location=torch.device('cpu')) # Authors pretrained checkpoint
    # Fnet_checkpoint = torch.load("./TrainHistory/FNet_time_05-17-02-25-07/checkpoint_4800.pth.tar", map_location=torch.device('cpu')) # Our pretrained Fnet checkpoint
    

    # Author provided Checkpoint
    # checkpoint = torch.load("checkpoint.pth.tar", map_location=torch.device('cpu')) 


    # Our Trained HDR-DSP with our pre-trained Fnet(Old)
    # checkpoint = torch.load("./TrainHistory/Real_['Avg', 'Max', 'Std']_N2N_FNet_ME_deconv_DetaAtte_W_JS_V_noisy_valvar_time_06-07-07-20-58/checkpoint_1700.pth.tar", map_location=torch.device('cpu'))

    # Our Trained HDR-DSP with our pre-trained Fnet(New) 
    # checkpoint = torch.load("./TrainHistory/Real_['Avg','Max','Std']_Our_Pretrained_Fnet/checkpoint_300.pth.tar", map_location=torch.device('cpu'))


    # Our Trained HDR-DSP with our pre-trained Fnet(Old) 
    # checkpoint = torch.load("./TrainHistory/Real_['Avg','Max','Std']/checkpoint_300.pth.tar", map_location=torch.device('cpu'))


    # Our trained HDR-DSP with Pretrained FNet by Authors
    checkpoint = torch.load("./TrainHistory/Real_['Avg','Max','Std']_Pretrained_Fnet/checkpoint_300.pth.tar", map_location=torch.device('cpu'))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    Decoder = DecoderNet(in_dim=1+nb_mode*num_features).float().to(device)
    Encoder = EncoderNet(in_dim=2,conv_dim=64, out_dim=num_features, num_blocks=num_blocks).float().to(device)
    Fnet = FNet().float().to(device)

    Decoder.load_state_dict(checkpoint['state_dictDecoder']) 
    Encoder.load_state_dict(checkpoint['state_dictEncoder']) 
    Fnet.load_state_dict(checkpoint['state_dictFnet']) # Normally
    # Fnet.load_state_dict(Fnet_checkpoint['state_dictFnet']) # For FNet pretrained checkpoint check

    gaussian_filter = GaussianBlur(11, sigma=1).to(device)

    TVLoss = TVL1(TVLoss_weight=1)
    warping = WarpedLoss(interpolation = 'bicubicTorch') 
    ##################
    
    # Dataset_path = 'SkySat_ME_noSaturation/'
    # try:
    #     Dataset_path = "/mnt/c/Users/Jatin/Desktop/Engineering/ISRO Internship/WORK/HDR-DSP-SR-main (1)/HDR-DSP-SR-main/data/hdr-dsp-real-dataset/"# Linux based path
    # except Exception as e:
    Dataset_path = "C:/Users/Jatin/Desktop/Engineering/ISRO Internship/WORK/HDR-DSP-SR-main (1)/HDR-DSP-SR-main/data/hdr-dsp-real-dataset/"
    test_loader = {}
         
    # for i in range(4,16):
    for i in range(2,3):
        transformedDataset = SkySatRealDataset_ME(Dataset_path, augmentation = False, phase = 'test', num_images = i)
        test_loader[str(i)] = torch.utils.data.DataLoader(transformedDataset, batch_size=val_bs, 
                                           num_workers=1, shuffle=False)

    
    Fnet.eval()
    Decoder.eval()
    Encoder.eval()

    with torch.no_grad():
        # for n in range(4,16):
        for n in range(2,3):
            # savepath = "Results/{}".format(n)
            savepath = "Results/{}".format(4)
            safe_mkdir(savepath)
            for k, (samplesLR, expotime) in enumerate(test_loader[str(n)]):
                plt.imshow(samplesLR[0,:,:,0])
                plt.savefig("./Results/sampleLR.png")
                # plt.show()
                samplesLR = samplesLR.float().to(device)
                # samplesLR = samplesLR.view()
                samplesLR = samplesLR.permute(0, 3, 1, 2)
                b, num_im, h, w = samplesLR.shape
                expotime = expotime.float().to(device)

                #######Flow
                flow, valwarploss = flowEstimation(samplesLR/expotime*3.4, ME=Fnet, gaussian_filter = gaussian_filter, warping = warping, device=device) 

                base, detail = base_detail_decomp(samplesLR/expotime, gaussian_filter) 

                # SR for the detail
                SR_detail = DeepSaaSuperresolve_weighted_base(detail, flow=flow, base = samplesLR, Encoder=Encoder, Decoder=Decoder,
                                        device = device, feature_mode= feature_mode, num_features = num_features, sr_ratio=sr_ratio, phase = 'validation')

                # SR for the base
                SR_base = zoombase_weighted(base, expotime, flow, device, warping)

                SR = SR_base + SR_detail

                SR = SR.detach().cpu().numpy().squeeze()
                np.save(os.path.join(savepath,"SR_{:02d}.npy".format(k)), SR)

                out_bic = do_shift_interp(samplesLR[0,0,:,:].cpu().detach().numpy(), None, None, sr_ratio)

                fig, ax = plt.subplots(1,2,sharex = True, sharey = True)
                ax[0].imshow(out_bic, cmap = "gray")
                ax[0].set_title("Bicubic Interpolation")
                
                ax[1].imshow(SR, cmap = "gray")
                ax[1].set_title("Super Resolved")
            
                plt.savefig(os.path.join("./Results/plot_pre_trained_fnet_trained_checkpoint.png"))
                plt.show()
                np.save(os.path.join(savepath,"bicubic.npy"), out_bic)

                break


def check(args):
    feature_mode = args.feature_mode
    print(feature_mode)
    print(len(feature_mode))

def main(args):
    """
    Given a configuration, trains Encoder, Decoder and fnet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """
    torch.cuda.empty_cache()
    test(args)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-on","--option_name", help="Option id", default='J_selfSR')
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader",type=int, default=10)
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader",type=int, default=1)
    parser.add_argument("-lrf", "--lr_fnet", help="Learning rate of fnet",type=float, default=1e-5)
    parser.add_argument("-lre", "--lr_encoder", help="Learning rate of Encoder",type=float, default=1e-4)
    parser.add_argument("-lrd", "--lr_decoder", help="Learning rate of Decoder",type=float, default=1e-4)
    parser.add_argument("-ff",  "--factor_fnet", help="Learning rate decay factor of fnet",type=float, default=0.3)
    parser.add_argument("-fe",  "--factor_encoder", help="Learning rate decay factor of Encoder",type=float, default=0.3)
    parser.add_argument("-fd",  "--factor_decoder", help="Learning rate decay factor of Decoder",type=float, default=0.3)
    parser.add_argument("-pf",  "--patience_fnet", help="Step size for learning rate of fnet",type=int, default=300)
    parser.add_argument("-pe",  "--patience_encoder", help="Step size for learning rate of Encoder",type=int, default=400)
    parser.add_argument("-pd",  "--patience_decoder", help="Step size for learning rate of Decoder",type=int, default=400)
    parser.add_argument("-ne",  "--num_epochs", help="Num_epochs",type=int, default=1800)
    parser.add_argument("-nf",  "--num_features", help="Num of features for each frame", type=int, default=64)
    parser.add_argument("-nb",  "--num_blocks", help="Number of residual blocks in encoder",type=int, default=4)
    parser.add_argument("-ww",  "--warp_weight", help="Weight for the warping loss",type=float, default=3)
    parser.add_argument("-tvw",  "--TVflow_weight", help="Weight for the TV flow loss",type=float, default=0.01)
    parser.add_argument("-s",  "--sigma", help="Std for SR filtering",type=float, default=1)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor",type=int, default=2)
    parser.add_argument('-fm','--feature_mode', nargs='+', help="feature mode (Avg, Max, Std)", default=['Avg', 'Max', 'Std'])


    args = parser.parse_args()

    main(args)
