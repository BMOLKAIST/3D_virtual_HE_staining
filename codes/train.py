
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset_h5_aug import *
#from dataset import *
from util_JY import *
from scnas import ScNas


import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms

torch.cuda.empty_cache()


def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    #task = args.task
    #opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_ids = args.gpu_ids

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    #print("task: %s" % task)
    #print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    print("nch: %d" % nch)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))

    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'png'))

    ## 네트워크 학습하기
    if mode == 'train':
        #transform_train = transforms.Compose([Normalization(mean=0.5, std=0.5)])
        transform_train = transforms.Compose([AddNoise(noise_type='gaussian', sgm=0.2), AddBlur(type='bilinear',rescale_factor=[3,3]), AddClip(maxv=0.85, minv = 0)])

        dataset_train = Dataset(yaml_dir=os.path.join(data_dir, 'train.yaml'), data_dir=os.path.join(data_dir, 'train'), transform=transform_train)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
        print(len(loader_train))


        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)


        #transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5)])
        transform_val = transforms.Compose([AddNoise(noise_type='gaussian', sgm=0.2), AddBlur(type='bilinear',rescale_factor=[3,3]), AddClip(maxv=0.85, minv = 0)])

        dataset_val = Dataset(yaml_dir=os.path.join(data_dir, 'val.yaml'), data_dir=os.path.join(data_dir, 'val'), transform=transform_val)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

        num_data_val = len(dataset_val)
        num_batch_val = np.ceil(num_data_val / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix":
        netG = Pix2Pix(in_channels=nch*3, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=4, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "scnas":
        netG = ScNas(input_channel=nch, num_feature=12, num_layers=8, num_multiplier=2, num_classes=3).to(device)
        netD = Discriminator(in_channels=4, out_channels=1, nker=nker, norm=norm).to(device)

        netG.to(device)
        netD.to(device)
        # netG = nn.DataParallel(netG, device_ids = gpu_ids)
        # netD = nn.DataParallel(netD, device_ids = gpu_ids)

        # netG.to(f'cuda:{netG.device_ids[0]}')
        # netD.to(f'cuda:{netD.device_ids[0]}')

        # print(netG)

        # init_weights(netG, init_type='normal', init_gain=0.02)
        # init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    # fn_l1 = nn.L1Loss().to(device)
    # fn_l2 = nn.MSELoss().to(device)
    # fn_gan = nn.BCELoss(reduction='mean').to(device)
    fn_gan = nn.BCELoss(reduction='mean').to(device)

    # import pytorch_ssim
    # fn_ssim = pytorch_ssim.SSIM(window_size = nx).to(device)

    # from ssim import SSIM
    # fn_ssim = SSIM().to(device)

    # def fn_pcc(y_pred, y_true):
    #     # x = torch.Tensor(y_pred)
    #     # y = torch.Tensor(y_true)
    #     x = y_pred
    #     y = y_true
    #     vx = x - torch.mean(x)
    #     vy = y - torch.mean(y)
    #     cov = torch.sum(vx * vy)
    #     corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
    #     return 1-corr

    # def norm(x):
    #     _min = 14450
    #     _max = 15614
    #     x = x.clip(_min, _max)
    #     # x = x.clip(x.min(), x.max())
    #     # x = (x - x.min()) / (x.max() - x.min())
    #     x = (x - _min) / (_max - _min)
    #     return x

    def fn_pcc(stain, ri):
        std_x = stain.std()
        std_y = ri.std()

        mean_x = stain.mean()
        mean_y = ri.mean()

        vx = (stain - mean_x) / std_x
        vy = (ri - mean_y) / std_y

        pcc = (vx * vy).mean()

        return 1-pcc

    # fn_pcc = fn_pcc.to(f'cuda:{netG.device_ids[0]}')

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    #fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = 'gray'

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                        netG=netG, netD=netD,
                                                        optimG=optimG, optimD=optimD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG.train()
            netD.train()

            # loss_G_l2_train = []
            loss_G_pcc_train = []
            # loss_G_l1_train = []
            # loss_G_ssim_train = []
            loss_G_gan_train = []
            loss_D_real_train = []
            loss_D_fake_train = []
            acc_D_fake_train = []
            acc_D_real_train = []

            for batch, data in enumerate(loader_train, 1):
                # forward pass

                label = data['label'].to(device)
                input = data['input'].to(device)

                # label = data['label'].to(f'cuda:{netG.device_ids[0]}')
                # input = data['input'].to(f'cuda:{netG.device_ids[0]}')

                # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)
                # print(netG)
                # print('label')
                # print(label)
                # print('input')
                # print(input)
                output = netG(input)

                # print(output.shape)

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                real = torch.cat([input, label], dim=1)
                # print(real.shape)
                fake = torch.cat([input, output], dim=1)

                pred_real = netD(real)
                # pred_real = torch.ones_like(pred_real)


                pred_fake = netD(fake.detach()) #detach is to eliminate backward propagation to the generator, let it stay within the discriminator
                # pred_fake = torch.ones_like(pred_fake)
                # print(pred_fake[0,0,:,:])

                if batch == 1 and epoch == 1:
                    print(pred_real.shape)
                    print(pred_fake.shape)
                    # print(output.shape)

                acc_D_real = torch.mean(pred_real) # 숫자 클수록 잘 맞춘 것
                acc_D_fake = 1 - torch.mean(pred_fake) # 숫자 작을수록 잘 맞춘 것

                loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad(netD, False)
                optimG.zero_grad()

                fake = torch.cat([input, output], dim=1)
                pred_fake = netD(fake)

                loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                # print(output.dtype)
                # print(label.dtype)
                # loss_G_l1 = fn_l1(output, label)
                # loss_G_l1 = fn_l2(output, label)
                # print(output.shape)
                # print(label.shape)
                # loss_G_pcc_r = fn_pcc(output[:,0,:,:], label[:,0,:,:])
                # print(loss_G_pcc_r)
                # loss_G_pcc_g = fn_pcc(output[:,1,:,:], label[:,1,:,:])
                # print(loss_G_pcc_r + loss_G_pcc_g)
                # loss_G_pcc_b = fn_pcc(output[:,2,:,:], label[:,2,:,:])
                # loss_G_pcc = (loss_G_pcc_r + loss_G_pcc_g + loss_G_pcc_b)/3

                loss_G_pcc = fn_pcc(output, label)

                # loss_G_l2 = fn_l2(output, label)
                # loss_G_l1 = 0.1 * loss_G_pcc + 0.9 * loss_G_l2
                # loss_G_ssim = fn_ssim(output, label)
                loss_G = loss_G_gan + wgt * loss_G_pcc

                # loss_G.backward()
                loss_G.backward()
                optimG.step()

                # 손실함수 계산
                # loss_G_l1_train += [loss_G_l1.item()]
                loss_G_pcc_train += [loss_G_pcc.item()]
                # loss_G_l2_train += [loss_G_l2.item()]
                # loss_G_ssim_train += [loss_G_ssim.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]
                acc_D_fake_train += [acc_D_fake.item()]
                acc_D_real_train += [acc_D_real.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN L1 %.4f | GEN GAN %.4f | "
                      "DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_pcc_train), np.mean(loss_G_gan_train),
                       np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))
                # print(loss_D_real_train)


                if batch % 100 == 0:
                  # Tensorboard 저장하기
#                  input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
#                  label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
#                  output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                  input = fn_tonumpy(input).squeeze()
                  label = fn_tonumpy(label).squeeze()
                  output = fn_tonumpy(output).squeeze()

                #   input = norm(input)
                #   label = np.clip(label, a_min=0, a_max=1)
                #   output = np.clip(output, a_min=0, a_max=1)

                #   label =  label / 255

                  input = np.clip(input, a_min=0, a_max=1)
                  label = np.clip(label, a_min=0, a_max=1)
                  output = np.clip(output, a_min=0, a_max=1)

                  id = num_batch_train * (epoch - 1) + batch

                  plt.imsave(os.path.join(result_dir_train, '%03d_input.png' % id), input, cmap=cmap) #input[0]
                  plt.imsave(os.path.join(result_dir_train, '%03d_label.png' % id), label, cmap=cmap) #label[0]
                  plt.imsave(os.path.join(result_dir_train, '%03d_output.png' % id), output, cmap=cmap) #output[0]

#                  writer_train.add_image('input', input, id, dataformats='NHWC')
#                  writer_train.add_image('label', label, id, dataformats='NHWC')
#                  writer_train.add_image('output', output, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G_pcc', np.mean(loss_G_pcc_train), epoch)
            # writer_train.add_scalar('loss_G_l2', np.mean(loss_G_l2_train), epoch)
            # writer_train.add_scalar('loss_G_l1', np.mean(loss_G_l1_train), epoch)
            # writer_train.add_scalar('loss_G_ssim', np.mean(loss_G_ssim_train), epoch)
            writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)
            writer_train.add_scalar('acc_D_fake', np.mean(acc_D_fake_train), epoch)
            writer_train.add_scalar('acc_D_real', np.mean(acc_D_real_train), epoch)

            with torch.no_grad():
                netG.eval()
                netD.eval()

                # loss_G_l1_val = []
                # loss_G_l2_val = []
                loss_G_pcc_val = []
                # loss_G_ssim_val = []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = []
                acc_D_fake_val = []
                acc_D_real_val = []

                for batch, data in enumerate(loader_val, 1):
                    # forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)

                    # label = data['label'].to(f'cuda:{netG.device_ids[0]}')
                    # input = data['input'].to(f'cuda:{netG.device_ids[0]}')

                    # input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                    output = netG(input)

                    # backward netD
                    # set_requires_grad(netD, True)
                    # optimD.zero_grad()

                    real = torch.cat([input, label], dim=1)
                    fake = torch.cat([input, output], dim=1)

                    pred_real = netD(real)
                    # print(pred_real.shape)
                    pred_fake = netD(fake.detach())
                    # print(pred_fake.shape)

                    acc_D_real = torch.mean(pred_real) # 숫자 클수록 잘 맞춘 것
                    acc_D_fake = 1 - torch.mean(pred_fake) # 숫자 작을수록 잘 맞춘 것

                    loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    # loss_D.backward()
                    # optimD.step()

                    # backward netG
                    # set_requires_grad(netD, False)
                    # optimG.zero_grad()

                    fake = torch.cat([input, output], dim=1)
                    pred_fake = netD(fake)

                    loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                    # loss_G_l1 = fn_l1(output, label)
                    # loss_G_l2 = fn_l2(output, label)

                    # loss_G_pcc_r = fn_pcc(output[:,0,:,:], label[:,0,:,:])
                    # loss_G_pcc_g = fn_pcc(output[:,1,:,:], label[:,1,:,:])
                    # loss_G_pcc_b = fn_pcc(output[:,2,:,:], label[:,2,:,:])
                    # loss_G_pcc = (loss_G_pcc_r + loss_G_pcc_g + loss_G_pcc_b)/3

                    loss_G_pcc = fn_pcc(output, label)
                    # loss_G_l1 = 0.1 * loss_G_pcc + 0.9 * loss_G_l2
                    # loss_G_ssim = fn_ssim(output, label)
                    loss_G = loss_G_gan + wgt * loss_G_pcc

                    # loss_G.backward()
                    # optimG.step()

                    # 손실함수 계산
                    # loss_G_l1_val += [loss_G_l1.item()]
                    # loss_G_l2_val += [loss_G_l2.item()]
                    loss_G_pcc_val += [loss_G_pcc.item()]
                    # loss_G_ssim_val += [loss_G_ssim.item()]
                    loss_G_gan_val += [loss_G_gan.item()]
                    loss_D_real_val += [loss_D_real.item()]
                    loss_D_fake_val += [loss_D_fake.item()]
                    acc_D_fake_val += [acc_D_fake.item()]
                    acc_D_real_val += [acc_D_real.item()]

                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | "
                          "GEN L1 %.4f | GEN GAN %.4f | "
                          "DISC REAL: %.4f | DISC FAKE: %.4f" %
                          (epoch, num_epoch, batch, num_batch_val,
                           np.mean(loss_G_pcc_val), np.mean(loss_G_gan_val),
                           np.mean(loss_D_real_val), np.mean(loss_D_fake_val)))

                    if batch % 50 == 0:
                        # Tensorboard 저장하기
                        #input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                        #label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                        #output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                        input = fn_tonumpy(input).squeeze()
                        label = fn_tonumpy(label).squeeze()
                        output = fn_tonumpy(output).squeeze()

                        # label =  label / 255

                        input = np.clip(input, a_min=0, a_max=1)
                        label = np.clip(label, a_min=0, a_max=1)
                        output = np.clip(output, a_min=0, a_max=1)

                        id = num_batch_train * (epoch - 1) + batch

                        plt.imsave(os.path.join(result_dir_val,  '%03d_input.png' % id), input, cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val,  '%03d_label.png' % id), label, cmap=cmap)
                        plt.imsave(os.path.join(result_dir_val,  '%03d_output.png' % id), output, cmap=cmap)
                        # input_i = Image.fromarray(input)
                        # label_i = Image.fromarray(label)
                        # output_i = Image.fromarray(output)

                        # input_i.save(os.path.join(result_dir_val,  '%03d_input.jpg' % id))
                        # label_i.save(os.path.join(result_dir_val,  '%03d_label.jpg' % id))
                        # output_i.save(os.path.join(result_dir_val,  '%03d_output.jpg' % id))


#                        writer_val.add_image('input', input, id, dataformats='NHWC')
#                        writer_val.add_image('label', label, id, dataformats='NHWC')
#                        writer_val.add_image('output', output, id, dataformats='NHWC')

                # writer_val.add_scalar('loss_G_l1', np.mean(loss_G_l1_val), epoch)
                # writer_val.add_scalar('loss_G_l2', np.mean(loss_G_l2_val), epoch)
                writer_val.add_scalar('loss_G_pcc', np.mean(loss_G_pcc_val), epoch)
                # writer_val.add_scalar('loss_G_ssim', np.mean(loss_G_ssim_val), epoch)
                writer_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
                writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch)
                writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch)
                writer_val.add_scalar('acc_D_fake', np.mean(acc_D_fake_val), epoch)
                writer_val.add_scalar('acc_D_real', np.mean(acc_D_real_val), epoch)


            if epoch % 10 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()
        writer_val.close()

def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    #task = args.task
    #opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt = args.wgt
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

   # print("task: %s" % task)
   # print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    print("nch: %d" % nch)

    print("device: %s" % device)
    print("mode: test")

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        # os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == "test":
        #transform_test = transforms.Compose([Normalization(mean=0.5, std=0.5)])

        dataset_test = Dataset_test(yaml_dir=os.path.join(data_dir, 'test.yaml'), data_dir=os.path.join(data_dir, 'test'))#, transform=transform_test)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        print(num_data_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix":
        netG = Pix2Pix(in_channels=nch / 3, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels= 4, out_channels=1, nker=nker, norm=norm).to(device) #in_channels = 2*nch nch + nch/3

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "scnas":
        netG = ScNas(input_channel=nch, num_feature=12, num_layers=8, num_multiplier=2, num_classes=3).to(device)
        netD = Discriminator(in_channels=4, out_channels=1, nker=nker, norm=norm).to(device)

        # print(netG)

        # init_weights(netG, init_type='normal', init_gain=0.02)
        # init_weights(netD, init_type='normal', init_gain=0.02)


    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    # fn_l1 = nn.L1Loss().to(device)
    # fn_gan = nn.BCELoss().to(device)



    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    #fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = 'gray'

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == "test":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG.eval()

            # loss_G_l1_test = []

            for batch, data in enumerate(loader_test, 1):

                
                # forward pass
                
                input = data['input'].to(device)
                savename = data['filename']
                savename = savename[0]
                savename = savename.split('.h5')
                savename = savename[0]

                output = netG(input)

                print("TEST: BATCH %04d / %04d " %
                    (batch, num_batch_test))

                input = fn_tonumpy(input).squeeze()
                output = fn_tonumpy(output).squeeze()

#save as h5
                # for j in range(input.shape[0]):

                #     id = batch_size * (batch - 1) + j

                #     input_ = input[j]
                #     output_ = output[j].squeeze()

                #     with h5py.File(os.path.join(result_dir_test, '%03d.h5' % id), 'a') as f:
                #         f.create_dataset('input', data=input_, compression='gzip')
                #         f.create_dataset('output', data=output_, compression='gzip')

#save as png

                # for j in range(input.shape[0]):

                #     id = batch_size * (batch - 1) + j

                #     input_ = input[j]
                #     output_ = output[j].squeeze()


                #     input_ = np.clip(input_, a_min=0, a_max=1)
                #     output_ = np.clip(output_, a_min=0, a_max=1)


                #     plt.imsave(os.path.join(result_dir_test, '%03d_input.png' % id), input_, cmap=cmap)
                #     plt.imsave(os.path.join(result_dir_test, '%03d_output.png' % id), output_)

#save using PIL

                input_ = input
                output_ = output

                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                input_PIL = input_*255
                input_PIL = input_PIL.astype(np.uint8)
                input_PIL = Image.fromarray(input_PIL)
                input_savepath = f'{result_dir_test}/{savename}_input.png'
                input_PIL.save(input_savepath)

                output_PIL = output_*255
                output_PIL = output_PIL.astype(np.uint8)
                output_PIL = Image.fromarray(output_PIL)
                output_savepath = f'{result_dir_test}/{savename}_output.png'
                output_PIL.save(output_savepath)
