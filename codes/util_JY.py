import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn


## 네트워크 grad 설정하기
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


## 네트워크 weights 초기화 하기
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>




## 네트워크 저장하기
def save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    print(ckpt_lst[-1])

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, netD, optimG, optimD, epoch

## Add Sampling
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(img.shape)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk

    elif type == "random":
        prob = opts[0]

        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < prob).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float)

        dst = img * msk

    elif type == "gaussian":
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(-((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk

    return dst

## 

def add_noise(img, type="gaussian", sgm=None):
    sz = img.shape
    # print('img shape')
    # print(img.shape)

    if type == "random":

        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], 1)

        dst = img + noise

    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    elif type == "gaussian":

        noise = np.random.normal(0,sgm,(sz[0], sz[1],1))
        dst = noise + img

        # noise = torch.tensor(np.random.normal(0,sgm,(sz[0], sz[2], sz[3]))).to('cuda')
        # # print('noise shape')
        # # print(noise.shape)
        
        # img_sq = torch.squeeze(img)
        # # print('img_sq shape')
        # # print(img_sq.shape)
        
        # dst = torch.add(img_sq, noise)
        # # print('dst shape')
        # # print(dst.shape)
        
        # dst = torch.unsqueeze(dst,1)
        # # print('dst_unsq shape')
        # # print(dst.shape)

    return dst

def add_clip(img, maxv=None, minv=None):
    dst = img.clip(minv,maxv)
    return dst

def add_blur(img, type="bilinear", rescale_factor=None):
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    sz = img.shape
    # if len(opts) == 1:
    #     keepdim = True
    # else:
    #     keepdim = opts[1]

    # dw = 1.0 / opts[0]
    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0]//rescale_factor[0], sz[1]//rescale_factor[1]), order=order)

    # if keepdim:
    #     # dst = rescale(dst, scale=(1 / dw, 1 / dw, 1), order=order)
    dst = resize(dst, output_shape=(sz[0], sz[1]), order=order)

    return dst

##

def image2patch(src, nimg, npatch, nmargin, datatype="tensor"):
    src = src.to('cpu').detach().numpy()

    nimg_zp = np.zeros(4, np.int32)
    ncrop = np.zeros(4, np.int32)
    nset = np.zeros(4, np.int32)

    for id in range(0, 4):
        nimg_zp[id] = int(nimg[id] + 2 * nmargin[id])
        ncrop[id] = int(npatch[id] - 2 * nmargin[id])
        nset[id] = np.ceil(nimg_zp[id] / ncrop[id]).astype(np.int32)

    nsmp = np.prod(nset)

    iset = [(np.linspace(0, nimg_zp[0] - npatch[0], nset[0])).astype(np.int32),
            (np.linspace(0, nimg_zp[1] - npatch[1], nset[1])).astype(np.int32),
            (np.linspace(0, nimg_zp[2] - npatch[2], nset[2])).astype(np.int32),
            (np.linspace(0, nimg_zp[3] - npatch[3], nset[3])).astype(np.int32)]

    patch = [np.arange(0, npatch[0])[:, np.newaxis, np.newaxis, np.newaxis],
             np.arange(0, npatch[1])[:, np.newaxis, np.newaxis],
             np.arange(0, npatch[2])[:, np.newaxis],
             np.arange(0, npatch[3])]

    src = np.pad(src, ((nmargin[0], nmargin[0]), (nmargin[1], nmargin[1]), (nmargin[2], nmargin[2]), (nmargin[3], nmargin[3])), 'reflect')
    dst = np.zeros((nsmp * npatch[0], npatch[1], npatch[2], npatch[3]), dtype=np.float32)

    for i in range(0, nset[0]):
        for j in range(0, nset[1]):
            for k in range(0, nset[2]):
                for q in range(0, nset[3]):

                    pos = [nset[3] * nset[2] * nset[1] * i + nset[2] * nset[1] * j + nset[1] * k + q]

                    i_ = iset[0][i] + patch[0]
                    j_ = iset[1][j] + patch[1]
                    k_ = iset[2][k] + patch[2]
                    q_ = iset[3][q] + patch[3]

                    dst[pos, :, :, :] = src[i_, j_, k_, q_]

    if datatype == "tensor":
        dst = torch.from_numpy(dst)

    return dst


def patch2image(src, nimg, npatch, nmargin, datatype="tensor", type="count"):
    src = src.to('cpu').detach().numpy()

    nimg_zp = np.zeros(4, np.int32)
    ncrop = np.zeros(4, np.int32)
    nset = np.zeros(4, np.int32)

    for id in range(0, 4):
        nimg_zp[id] = int(nimg[id] + 2 * nmargin[id])
        ncrop[id] = int(npatch[id] - 2 * nmargin[id])
        nset[id] = np.ceil(nimg_zp[id] / ncrop[id]).astype(np.int32)

    nsmp = np.prod(nset)

    iset = [(np.linspace(0, nimg_zp[0] - npatch[0], nset[0])).astype(np.int32),
             (np.linspace(0, nimg_zp[1] - npatch[1], nset[1])).astype(np.int32),
             (np.linspace(0, nimg_zp[2] - npatch[2], nset[2])).astype(np.int32),
             (np.linspace(0, nimg_zp[3] - npatch[3], nset[3])).astype(np.int32)]

    crop = [nmargin[0] + np.arange(0, ncrop[0])[:, np.newaxis, np.newaxis, np.newaxis],
            nmargin[1] + np.arange(0, ncrop[1])[:, np.newaxis, np.newaxis],
            nmargin[2] + np.arange(0, ncrop[2])[:, np.newaxis],
            nmargin[3] + np.arange(0, ncrop[3])]

    dst = np.zeros([nimg_zp[0], nimg_zp[1], nimg_zp[2], nimg_zp[3]], dtype=np.float32)
    wgt = np.zeros([nimg_zp[0], nimg_zp[1], nimg_zp[2], nimg_zp[3]], dtype=np.float32)

    i_img = [np.arange(nmargin[0], nimg_zp[0] - nmargin[0]).astype(np.int32)[:, np.newaxis, np.newaxis, np.newaxis],
             np.arange(nmargin[1], nimg_zp[1] - nmargin[1]).astype(np.int32)[:, np.newaxis, np.newaxis],
             np.arange(nmargin[2], nimg_zp[2] - nmargin[2]).astype(np.int32)[:, np.newaxis],
             np.arange(nmargin[3], nimg_zp[3] - nmargin[3]).astype(np.int32)]

    bnd = [ncrop[0] - iset[0][1] if not len(iset[0]) == 1 else 0,
           ncrop[1] - iset[1][1] if not len(iset[1]) == 1 else 0,
           ncrop[2] - iset[2][1] if not len(iset[2]) == 1 else 0,
           ncrop[3] - iset[3][1] if not len(iset[3]) == 1 else 0]

    if type == 'cos':
        wgt_bnd = [None for _ in range(4)]

        for id in range(1, 4):
            t = np.linspace(np.pi, 2 * np.pi, bnd[id])
            wgt_ = np.ones((ncrop[id]), np.float32)
            wgt_[0:bnd[id]] = (np.cos(t) + 1.0)/2.0

            axis_ = [f for f in range(0, 4)]
            axis_.remove(id)
            wgt_ = np.expand_dims(wgt_, axis=axis_)

            ncrop_ = [ncrop[f] for f in range(0, 4)]
            ncrop_[id] = 1

            wgt_bnd[id] = np.tile(wgt_, ncrop_)

    for i in range(0, nset[0]):
        for j in range(0, nset[1]):
            for k in range(0, nset[2]):
                for q in range(0, nset[3]):

                    wgt_ = np.ones(ncrop, np.float32)

                    if type == 'cos':
                        for id in range(1, 4):
                            if id == 1:
                                axs = j
                            elif id == 2:
                                axs = k
                            elif id == 3:
                                axs = q

                            if axs == 0:
                                wgt_ *= np.flip(wgt_bnd[id], id)
                            elif axs == nset[id] - 1:
                                wgt_ *= wgt_bnd[id]
                            else:
                                wgt_ *= np.flip(wgt_bnd[id], id) * wgt_bnd[id]

                    pos = [nset[3] * nset[2] * nset[1] * i + nset[2] * nset[1] * j + nset[1] * k + q]

                    i_ = iset[0][i] + crop[0]
                    j_ = iset[1][j] + crop[1]
                    k_ = iset[2][k] + crop[2]
                    q_ = iset[3][q] + crop[3]

                    src_ = src[pos, :, :, :]
                    dst[i_, j_, k_, q_] = dst[i_, j_, k_, q_] + src_[crop[0], crop[1], crop[2], crop[3]] * wgt_
                    wgt[i_, j_, k_, q_] = wgt[i_, j_, k_, q_] + wgt_

    if type == 'count':
        dst = dst/wgt

    dst = dst[i_img[0], i_img[1], i_img[2], i_img[3]]
    wgt = wgt[i_img[0], i_img[1], i_img[2], i_img[3]]

    if datatype == "tensor":
        dst = torch.from_numpy(dst)
        wgt = torch.from_numpy(wgt)

    return dst