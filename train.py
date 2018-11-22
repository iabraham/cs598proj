import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optimizers
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainFromFolder, ValidateFromFolder, display
from loss import GenLoss
from model import Generator, Discrim

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=42, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE, UPSCALE_FACTOR, NUM_EPOCHS = opt.crop_size, opt.upscale_factor, opt.num_epochs

train_set = TrainFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValidateFromFolder('data/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG, netD, gen_crit = Generator(UPSCALE_FACTOR), Discrim(), GenLoss()

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    gen_crit.cuda()

optG = optimizers.Adam(netG.parameters())
optD = optimizers.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        b_size = data.size(0)
        running_results['batch_sizes'] += b_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = gen_crit(fake_out, fake_img, real_img)
        g_loss.backward()
        optG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        g_loss = gen_crit(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.item() * b_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.item() * b_size
        running_results['d_score'] += real_out.item() * b_size
        running_results['g_score'] += fake_out.item() * b_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    netG.eval()
    out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    validation_bar = tqdm(val_loader)
    validation_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    validation_images = []
    for val_lr, val_hr_restore, val_hr in validation_bar:
        b_size = val_lr.size(0)
        validation_results['batch_sizes'] += b_size
        with torch.no_grad():
            lr, hr = Variable(val_lr), Variable(val_hr)
        if torch.cuda.is_available():
            lr, hr = lr.cuda(), hr.cuda()
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        validation_results['mse'] += batch_mse * b_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).item()
        validation_results['ssims'] += batch_ssim * b_size
        validation_results['psnr'] = 10 * log10(1 / (validation_results['mse'] / validation_results['batch_sizes']))
        validation_results['ssim'] = validation_results['ssims'] / validation_results['batch_sizes']
        validation_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                validation_results['psnr'], validation_results['ssim']))

        validation_images.extend(
            [display()(val_hr_restore.squeeze(0)), display()(hr.data.cpu().squeeze(0)),
             display()(sr.data.cpu().squeeze(0))])
    validation_images = torch.stack(validation_images)
    validation_images = torch.chunk(validation_images, validation_images.size(0) // 15)
    validation_save_bar = tqdm(validation_images, desc='[saving training results]')
    index = 1
    for image in validation_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(validation_results['psnr'])
    results['ssim'].append(validation_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
