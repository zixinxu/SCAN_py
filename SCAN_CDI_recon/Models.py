import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from torchmetrics.image import PeakSignalNoiseRatio, TotalVariation
from Formation import forward2D, evaluate
import wandb
from numpy import fft
from fasta import fasta, plots, Convergence
from fasta.linalg import LinearMap
import scipy.fftpack as sf
import math as m
from skimage.restoration import denoise_tv_chambolle
import time
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale, Pad, Resize
from PIL import Image

from common import skip
import Tools as T

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_new_intensity(phase,old_intensity,probe):
    Combined_Detector=torch.complex(old_intensity*torch.cos(phase),old_intensity*torch.sin(phase))
    Combined_Detector = torch.fft.ifftshift(Combined_Detector, dim=(-1, -2, -3))
    Original_Object = torch.fft.ifftn(Combined_Detector, dim=(-1, -2, -3))
    Original_Object_density=torch.clamp(Original_Object.abs(),0,1)
    Original_Object= torch.complex(Original_Object_density*torch.cos(Original_Object.angle()),Original_Object_density*torch.sin(Original_Object.angle()))
    Original_Object=torch.mul(Original_Object,probe)
    Original_Object = torch.fft.fftn(Original_Object, dim=(-3, -2, -1))
    updated_Intensity = torch.fft.fftshift(Original_Object, dim=(-3, -2, -1))
    updated_Intensity = updated_Intensity.abs()
    return updated_Intensity

def SIREN(y, probe, probesize, image_size, real_density, real_phase, amp_name, phase_name, criterion=nn.MSELoss(), coordpara=1e-1,
          epoches=50000, LR=1e-4, c=False, mode=1, noise=0, lambd=[1,0,1e-4], log=True):
    c, h, w = y.shape
    probe = probe.cuda()

    # logging
    if log:
        t0 = time.time()
        if mode==2:
            wandb.init(project="Phase Retrival", name="SIREN_noabs")
        elif mode==3:
            wandb.init(project="Phase Retrival", name="SIREN_notanh")
        elif mode==4:
            wandb.init(project="Phase Retrival", name="SIREN_noabsclamp")
        elif mode==5:
            wandb.init(project="Phase Retrival", name="SIREN_noabssig")
        elif noise:
            wandb.init(project="Phase Retrival", name="SIREN"+str(noise)+str(lambd))    
        else:
            wandb.init(project="Phase Retrival", name="SIRENl1"+str(lambd))   
        t1 = time.time()
        twandb = t1-t0
    else:
        twandb = 0

    class SineLayer(nn.Module):

        def __init__(self, in_features, out_features, bias=True,
                     is_first=False, omega_0=30):
            super().__init__()
            self.omega_0 = omega_0
            self.is_first = is_first

            self.in_features = in_features
            self.linear = nn.Linear(in_features, out_features, bias=bias)

            self.init_weights()

        def init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features,
                                                1 / self.in_features)
                else:
                    self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                                np.sqrt(6 / self.in_features) / self.omega_0)

        def forward(self, input):
            return torch.sin(self.omega_0 * self.linear(input))

    class Siren(nn.Module):

        def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                     first_omega_0=30, hidden_omega_0=30.):

            super().__init__()

            self.net2 = []
            self.net2.append(SineLayer(in_features, hidden_features,
                                       is_first=True, omega_0=first_omega_0))

            for i in range(hidden_layers):
                self.net2.append(SineLayer(hidden_features, hidden_features,
                                           is_first=False, omega_0=hidden_omega_0))

            if outermost_linear:
                final_linear = nn.Linear(hidden_features, out_features)

                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                 np.sqrt(6 / hidden_features) / hidden_omega_0)

                self.net2.append(final_linear)
            else:
                self.net2.append(SineLayer(hidden_features, out_features,
                                           is_first=False, omega_0=hidden_omega_0))
            if (mode!=3) and (mode!=5):
                self.net2.append(nn.Tanh())

            self.net2 = nn.Sequential(*self.net2)

        def forward(self, coords):
            coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
            model_output = self.net2(coords)
            if mode == 3:
                model_output = torch.clamp(model_output, min=-1.0, max=1.0)
            return model_output

    min_loss = 1000
    size = y.shape[-1]
    coords = get_mgrid(probesize, dim=2)*coordpara
    model = Siren(in_features=2,
                  hidden_features=256,
                  hidden_layers=3,
                  out_features=2,
                  outermost_linear=True)
    model = model.cuda()
    coords = coords.cuda()
    optim = torch.optim.Adam(lr=LR, params=model.parameters())
    total_steps = epoches

    psnr = PeakSignalNoiseRatio().cuda()
    tv = TotalVariation().cuda()
    for step in range(total_steps):
        model_output = model(coords)

        density = model_output[:, 0]
        phase = model_output[:, 1]
        if (mode != 2)and(mode!=4)and(mode!=5):
            density = abs(density)
        elif mode==2:
            density = (density+1)/2
        elif mode==5:
            density = nn.Sigmoid(density)
            phase = nn.Tanh(phase)
        else:
            density = torch.clamp(density, min=0, max=1)
        density = density.view(1, probesize, probesize)
        phase = phase.view(1, probesize, probesize)
        transform = Compose([
        Pad(int(size/2-probesize/2)),
        ])
        density = transform(density)
        # plt.imshow(density.cpu().detach().numpy()[0])
        # plt.show()
        phase = transform(phase)
        Predicted_Intensity, P_phase = forward2D(probe, density, phase)

        Updated_Intensity=get_new_intensity(P_phase,y.cuda(),probe)
        loss = lambd[0]*criterion(Predicted_Intensity, y.cuda())+ lambd[1]*criterion(Updated_Intensity,y.cuda())+ lambd[2]*(noise/255)**2*tv(phase.view(1,1, h, w)) # 
        t0=time.time()
        psnr_amp, psnr_pha, ssim_amp, ssim_pha, reg_amp_best, reg_pha_best = evaluate(density.cpu(),
                                                                                      phase.cpu(),
                                                                                      real_density,
                                                                                      real_phase, amp_name,
                                                                                      phase_name,
                                                                                      image_size, probesize,
                                                                                      registration)
        if log:
            wandb.log({"loss": loss, "psnr_amp": np.asarray(psnr_amp.detach().numpy()),
                    "psnr_pha": np.asarray(psnr_pha.detach().numpy()),
                    "ssim_amp": np.asarray(ssim_amp.detach().numpy()),
                    "ssim_pha": np.asarray(ssim_pha.detach().numpy())})
        t1=time.time()
        twandb = twandb + t1-t0

        if (not step % 1000) or (step==total_steps-1):
            print("Step %d, Total loss %0.6f" % (step, loss))
            
        if loss <= min_loss:   
            minstep = step
            min_loss = loss
            t0=time.time()
            psnr_amp, psnr_pha, ssim_amp, ssim_pha, reg_amp_best, reg_pha_best = evaluate(density.cpu(),
                                                                                      phase.cpu(),
                                                                                      real_density,
                                                                                      real_phase, amp_name,
                                                                                      phase_name,
                                                                                      image_size, probesize,
                                                                                      registration)
            t1=time.time()
            twandb = twandb + t1-t0
            best_density = density
            best_phase = phase
            best_regamp = reg_amp_best
            best_regpha = reg_pha_best
            bestmodel = model

        optim.zero_grad()
        loss.backward()
        optim.step()
    if log:
        t0=time.time()
        images = wandb.Image(
            best_regamp,
        )
        wandb.log({"Amplitude": images})

        images = wandb.Image(
            best_regpha,
        )
        wandb.log({"Phase": images})
        wandb.finish()
        if mode==2:
            torch.save(bestmodel.state_dict(), './model/SIREN_noabs'+amp_name+str(lambd)+'.pt')
        elif mode==3:
            torch.save(bestmodel.state_dict(), './model/SIREN_notanh'+amp_name+str(lambd)+'.pt')
        elif mode==4:
            torch.save(bestmodel.state_dict(), './model/SIREN_noabsclamp'+amp_name+str(lambd)+'.pt')
        elif mode==5:
            torch.save(bestmodel.state_dict(), './model/SIREN_noabssig'+amp_name+str(lambd)+'.pt')
        elif noise:
            torch.save(bestmodel.state_dict(), './model/SIRENn'+str(noise)+amp_name+str(lambd)+'.pt')  
        else:
            torch.save(bestmodel.state_dict(), './model/SIREN_'+amp_name+str(lambd)+'.pt') 
        t1=time.time()
        twandb = twandb + t1-t0
    return best_density, best_phase, twandb




