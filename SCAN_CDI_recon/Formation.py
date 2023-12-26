import torch
from torch import nn
import numpy as np
import skimage
from PIL import Image,ImageChops
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale, Pad
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import ms_ssim, ssim
from scipy.io import loadmat

def forward2D(probe,amplitude,phase):
    Predicted_Object=torch.complex(amplitude*torch.cos(phase*np.pi),amplitude*torch.sin(phase*np.pi))
    Prediction=torch.mul(Predicted_Object,probe)
    Prediction = torch.fft.fftn(Prediction, dim=(-2, -1))
    P_Intensity = torch.fft.fftshift(Prediction, dim=(-2, -1))
    P_phase = P_Intensity.angle()
    P_Intensity = torch.abs(P_Intensity)
    return P_Intensity, P_phase

def gen_probe(probesize, imagesize):
    probe = torch.tensor(np.ones((1, probesize, probesize)))
    pad = nn.ZeroPad2d(round(imagesize/2-probesize/2))
    probe = pad(probe)
    return probe

def gen_image(imagesize, pad, imagename = "cameraman", needgrayscale=True):
    if imagename == 'cameraman':
        img = Image.fromarray(skimage.data.camera())
    elif imagename == 'brick':
        img = Image.fromarray(skimage.data.brick())
    elif imagename == 'identity':
        img = Image.fromarray(np.ones((imagesize, imagesize)))
    else:
        img = Image.open(imagename)
    if needgrayscale:
        img = Grayscale(1)(img)
    transform = Compose([
        Resize([imagesize,imagesize]),
        ToTensor(),
        Pad(pad),
    ])
    img = transform(img)
    return img

        
def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

def simulation(imagesize=128, probesize = 64, pad = 0, amp_name = 'cameraman', phase_name = 'brick', amp_needgrayscale=True, pha_needgrayscale=True, noise=False):
    size = imagesize + 2*pad
    probe = gen_probe(probesize, size)
    amplitude = gen_image(imagesize, pad, amp_name, needgrayscale=amp_needgrayscale)
    phase = gen_image(imagesize,pad, phase_name, needgrayscale=pha_needgrayscale)
    intensity, _ = forward2D(probe, amplitude, phase)
    if noise:
        intensity = add_gaussian_noise(intensity, mean=0, std=noise/255)
        # intensity_ = abs(intensity_)

    amplitude_pad = amplitude
    amplitude = amplitude[:,round(size/2-probesize/2):round(size/2+probesize/2),round(size/2-probesize/2):round(size/2+probesize/2)]
    phase = phase[:, round(size / 2 - probesize / 2):round(size / 2 + probesize / 2),
                round(size / 2 - probesize / 2):round(size / 2 + probesize / 2)]

    return intensity, probe, amplitude, phase, amplitude_pad

def registration(name, image_size, probesize, img, amp_needgrayscale=True, pha_needgrayscale=True):
    if torch.is_tensor(img):
        img = img.detach().numpy()
    if len(img.shape)==3:
        img = img[0]
    size = img.shape[0]
    pad = round((size - image_size) / 2)
    size = img.shape[0]
    referimage = simulation(imagesize=image_size, probesize = probesize, pad = pad, amp_name = name, phase_name = 'identity', amp_needgrayscale=amp_needgrayscale, pha_needgrayscale=pha_needgrayscale)[-1].detach().numpy()[-1]
    probe = gen_probe(probesize, size)
    referimage = referimage*probe[0].detach().numpy()
    shift, error, diffphase = phase_cross_correlation(referimage, img)
    img = fourier_shift(np.fft.fftn(img), shift)
    img = np.fft.ifftn(img)
    img = img[round(size/2-probesize/2):round(size/2+probesize/2),round(size/2-probesize/2):round(size/2+probesize/2)]
    return img.real


def evaluate(rec_amp, rec_pha, gt_amp, gt_pha, amp_name, pha_name, image_size, probesize, registra=True, amp_needgrayscale=True, pha_needgrayscale=True):
    psnr = PeakSignalNoiseRatio()
    size = rec_amp.shape[-1]
    if registra:
        reg_amp = registration(amp_name, image_size, probesize, rec_amp, amp_needgrayscale=amp_needgrayscale, pha_needgrayscale=pha_needgrayscale)
        reg_amp_rot = np.rot90(reg_amp, k=2)
        reg_pha = registration(pha_name, image_size, probesize, rec_pha, amp_needgrayscale=amp_needgrayscale, pha_needgrayscale=pha_needgrayscale)
        reg_pha_rot = np.rot90(reg_pha, k=2)
    else:
        if torch.is_tensor(rec_amp):
            rec_amp = rec_amp.detach().numpy()
        if len(rec_amp.shape) == 3:
            rec_amp = rec_amp[0]
        if torch.is_tensor(rec_pha):
            rec_pha = rec_pha.detach().numpy()
        if len(rec_pha.shape) == 3:
            rec_pha = rec_pha[0]
        reg_amp = rec_amp[round(size / 2 - probesize / 2):round(size / 2 + probesize / 2),
                  round(size / 2 - probesize / 2):round(size / 2 + probesize / 2)]
        reg_amp_rot = np.rot90(reg_amp, k=2)
        reg_pha = rec_pha[round(size / 2 - probesize / 2):round(size / 2 + probesize / 2),
                  round(size / 2 - probesize / 2):round(size / 2 + probesize / 2)]
        reg_pha_rot = np.rot90(reg_pha, k=2)
    gt_amp = (gt_amp - gt_amp.min()) / (gt_amp.max() - gt_amp.min())
    gt_pha = (gt_pha - gt_pha.min()) / (gt_pha.max() - gt_pha.min())    
    if reg_amp.max() == reg_amp.min():
        reg_amp = (reg_amp - reg_amp.min()) / 10
    else:
        reg_amp = (reg_amp - reg_amp.min()) / (reg_amp.max() - reg_amp.min())
    if reg_amp_rot.max() ==reg_amp_rot.min():
        reg_amp = (reg_amp - reg_amp.min()) / 10
    else:
        reg_amp_rot = (reg_amp_rot - reg_amp_rot.min()) / (reg_amp_rot.max() - reg_amp_rot.min())
    if reg_pha.max() == reg_pha.min():
        reg_amp = (reg_amp - reg_amp.min()) / 10
    else:
        reg_pha = (reg_pha - reg_pha.min()) / (reg_pha.max() - reg_pha.min())
    if reg_pha_rot.max() == reg_pha_rot.min():
        reg_amp = (reg_amp - reg_amp.min()) / 10
    else:
        reg_pha_rot = (reg_pha_rot - reg_pha_rot.min()) / (reg_pha_rot.max() - reg_pha_rot.min())
        

    ac, ah, aw = gt_amp.shape
    pc, ph, pw = gt_pha.shape

    ssim_amp = ssim(torch.from_numpy(reg_amp.copy()).float().reshape(1, ac, ah, aw),
                    gt_amp.float().reshape(1, ac, ah, aw), data_range=1., nonnegative_ssim=True)
    ssim_amp_rot = ssim(torch.from_numpy(reg_amp_rot.copy()).float().reshape(1, ac, ah, aw),
                        gt_amp.float().reshape(1, ac, ah, aw), data_range=1., nonnegative_ssim=True)
    ssim_amp_rev = ssim(torch.from_numpy(1 - reg_amp.copy()).float().reshape(1, ac, ah, aw),
                        gt_amp.float().reshape(1, ac, ah, aw), data_range=1., nonnegative_ssim=True)
    ssim_amp_rev_rot = ssim(torch.from_numpy(np.rot90(1 - reg_amp.copy(), k=2).copy()).float().reshape(1, ac, ah, aw),
                            gt_amp.float().reshape(1, ac, ah, aw), data_range=1., nonnegative_ssim=True)

    ssim_pha = ssim(torch.from_numpy(reg_pha.copy()).float().reshape(1, pc, ph, pw),
                    gt_pha.float().reshape(1, pc, ph, pw), data_range=1., nonnegative_ssim=True)
    ssim_pha_rot = ssim(torch.from_numpy(reg_pha_rot.copy()).float().reshape(1, pc, ph, pw),
                        gt_pha.float().reshape(1, pc, ph, pw), data_range=1., nonnegative_ssim=True)
    ssim_pha_rev = ssim(torch.from_numpy(1 - reg_pha.copy()).float().reshape(1, pc, ph, pw),
                        gt_pha.float().reshape(1, pc, ph, pw), data_range=1., nonnegative_ssim=True)
    ssim_pha_rev_rot = ssim(torch.from_numpy(np.rot90(1 - reg_pha.copy(), k=2).copy()).float().reshape(1, pc, ph, pw),
                            gt_pha.float().reshape(1, pc, ph, pw), data_range=1., nonnegative_ssim=True)

    if ssim_pha >= max(ssim_pha_rot, ssim_pha_rev, ssim_pha_rev_rot):
        reg_pha_best = reg_pha
    elif ssim_pha_rot >= max(ssim_pha, ssim_pha_rev, ssim_pha_rev_rot):
        reg_pha_best = reg_pha_rot
        ssim_pha = ssim_pha_rot
    elif ssim_pha_rev >= max(ssim_pha, ssim_pha_rot, ssim_pha_rev_rot):
        reg_pha_best = 1 - reg_pha.copy()
        ssim_pha = ssim_pha_rev
    elif ssim_pha_rev_rot >= max(ssim_pha, ssim_pha_rot, ssim_pha_rev):
        reg_pha_best = np.rot90(1 - reg_pha.copy(), k=2)
        ssim_pha = ssim_pha_rev_rot
    else:
        print('None')
        print(ssim_pha, ssim_pha_rot, ssim_pha_rev, ssim_pha_rev_rot)

    if ssim_amp >= max(ssim_amp_rot, ssim_amp_rev, ssim_amp_rev_rot):
        reg_amp_best = reg_amp
    elif ssim_amp_rot >= max(ssim_amp, ssim_amp_rev, ssim_amp_rev_rot):
        reg_amp_best = reg_amp_rot
        ssim_amp = ssim_amp_rot
    elif ssim_amp_rev >= max(ssim_amp, ssim_amp_rot, ssim_amp_rev_rot):
        reg_amp_best = 1 - reg_amp.copy()
        ssim_amp = ssim_amp_rev
    elif ssim_amp_rev_rot >= max(ssim_amp, ssim_amp_rot, ssim_amp_rev):
        reg_amp_best = np.rot90(1 - reg_amp.copy(), k=2)
        ssim_amp = ssim_amp_rev_rot
    else:
        print('None')
        print(ssim_amp, ssim_amp_rot, ssim_amp_rev, ssim_amp_rev_rot)

    psnr_amp = psnr(gt_amp, torch.from_numpy(reg_amp_best.copy()).reshape(gt_amp.shape))
    psnr_pha = psnr(gt_pha, torch.from_numpy(reg_pha_best.copy()).reshape(gt_pha.shape))

    return psnr_amp, psnr_pha, ssim_amp, ssim_pha, reg_amp_best, reg_pha_best