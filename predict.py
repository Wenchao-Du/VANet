#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models.AG import *
#Metrics lib
from metrics import calc_psnr, calc_ssim
import matplotlib.pyplot as plt
import h5py
import matplotlib.image as ims


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='demo')
    parser.add_argument("--input_dir", type=str, default='demo/input/')
    parser.add_argument("--output_dir", type=str, default='demo/output/')
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args


def align_to_four(img):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(image):
    image = torch.from_numpy(image).float()
    image = Variable(image).cuda()
    masklist, s1out, s2out, out = model(image)  #
    out = out.cpu().data
    out = out.numpy()
    out = np.squeeze(out)

    # mask = torch.cat((masklist[0], masklist[1], masklist[2], masklist[3]), 1)
    # mask = mask.cpu().data
    # mask = mask.numpy()
    # mask = np.squeeze(mask)
    return out


def test():
    fp = h5py.File('your dataset', 'r')
    Hdata = fp['HData']
    Ldata = fp['LData']
    # label = fp['lable']

    hdmat = Hdata[1, :, :]
    hdmat = np.transpose(hdmat)
    hdmat = hdmat.astype(np.float32)
    ldmat = Ldata[1, :, :]
    ldmat = np.transpose(ldmat)
    ldmat = ldmat.astype(np.float32)
    ldmat1 = np.expand_dims(ldmat, 0)
    image = np.expand_dims(ldmat1, 0)
    outimage = predict(image)

    print(calc_psnr(outimage, hdmat))
    print(calc_ssim(outimage, hdmat))

    print(calc_psnr(ldmat, hdmat))
    print(calc_ssim(ldmat, hdmat))

    plt.figure('src_image')
    plt.imshow(hdmat[:, :], cmap='gray', vmin=860 / 3000, vmax=1260 / 3000)
    plt.figure('denoise_ret')
    plt.imshow(outimage[:, :], cmap='gray', vmin=860 / 3000,
               vmax=1260 / 3000)  #, vmin=860/3000, vmax=1260/3000
    # plt.savefig('denoise.png')
    plt.show()


def normlize(input, nrange=[0, 1]):
    minvalue = np.min(input)
    maxvalue = np.max(input)
    scale = (nrange[1] - nrange[0]) / (maxvalue - minvalue)
    newinput = (input - minvalue) * scale

    return newinput


def eval(file):

    fp = h5py.File('your dataset', 'r')
    Hdata = fp['HData']
    Ldata = fp['LData']
    label = fp['lable']
    data_Count = label.shape[1]
    index_list = np.zeros((1, label.shape[1]), dtype=np.int32)
    fp = open(file, 'r')
    for line in fp:
        index = int(line)
        index_list[0, index] = 1
    fp.close()

    llist = np.zeros((1936, 512, 512), dtype=np.float32)
    hlist = np.zeros((1936, 512, 512), dtype=np.float32)
    plist = np.zeros((1936, 512, 512), dtype=np.float32)

    print('data all number: {}'.format(data_Count))

    index_count = 0
    for i in range(data_Count):
        if index_list[0, i] != 0:
            continue
        hdmat = Hdata[i, :, :]
        hdmat = np.transpose(hdmat)
        hdmat = hdmat.astype(np.float32)
        ldmat = Ldata[i, :, :]
        ldmat = np.transpose(ldmat)
        ldmat = ldmat.astype(np.float32)
        ldmat1 = np.expand_dims(ldmat, 0)
        image = np.expand_dims(ldmat1, 0)
        preimg = predict(image)
        llist[index_count, :, :] = ldmat
        hlist[index_count, :, :] = hdmat
        plist[index_count, :, :] = preimg
        index_count = index_count + 1
    print('index_sum:{}'.format(index_count))

    if len(llist) != len(hlist) or len(hlist) != len(plist):
        raise Exception('size does not match!')

    # np.save('K:\\DeRaindrop\\DeNoisedrop\\LDCT.npy', llist)
    # np.save('K:\\DeRaindrop\\DeNoisedrop\\HDCT.npy', hlist)
    np.save('', plist)

    print('data save finish !')


def evaldata():
    a = np.load('K:\\DeNoisedrop\\LDCT.npy')
    b = np.load('K:\\DeNoisedrop\\HDCT.npy')
    c = np.load('K:\\DeNoisedrop\\GD.npy')
    sum_ = a.shape[0]
    sum_psnr = 0
    sum_ssim = 0
    ld_sum_psnr = 0
    ld_sum_ssim = 0
    count = 0
    b = np.where(b > 0, b, 0)
    b = np.where(b < 1, b, 1)
    c = np.where(c > 0, c, 0)
    c = np.where(c < 1, c, 1)
    for i in range(sum_):
        psnr = calc_psnr(b[i, :, :], c[i, :, :])
        ssim = calc_ssim(b[i, :, :], c[i, :, :])
        sum_psnr = sum_psnr + psnr
        sum_ssim = sum_ssim + ssim
        count = count + 1
        lpsnr = calc_psnr(b[i, :, :], a[i, :, :])
        lssim = calc_ssim(b[i, :, :], a[i, :, :])
        ld_sum_psnr = ld_sum_psnr + lpsnr
        ld_sum_ssim = ld_sum_ssim + lssim
    print('count:{}'.format(count))
    pp_mean = sum_psnr / count
    ps_mean = sum_ssim / count

    lp_mean = ld_sum_psnr / count
    ls_mean = ld_sum_ssim / count

    print('pre:psnr.{}\t ssim.{}\n ld:psnr.{}\t ssim.{}'.format(
        pp_mean, ps_mean, lp_mean, ls_mean))


def getMask(file):
    fp = h5py.File('your dataset', 'r')
    Hdata = fp['HData']
    Ldata = fp['LData']
    label = fp['lable']
    data_Count = label.shape[1]
    index_list = np.zeros((1, label.shape[1]), dtype=np.int32)
    fp = open(file, 'r')
    for line in fp:
        index = int(line)
        index_list[0, index] = 1
    fp.close()

    masklist = np.zeros((1936, 4, 512, 512), dtype=np.float32)

    print('data all number: {}'.format(data_Count))

    index_count = 0
    for i in range(data_Count):
        if index_list[0, i] != 0:
            continue
        hdmat = Hdata[i, :, :]
        hdmat = np.transpose(hdmat)
        hdmat = hdmat.astype(np.float32)
        ldmat = Ldata[i, :, :]
        ldmat = np.transpose(ldmat)
        ldmat = ldmat.astype(np.float32)
        ldmat1 = np.expand_dims(ldmat, 0)
        image = np.expand_dims(ldmat1, 0)
        preimg = predict(image)
        masklist[index_count, :, :, :] = preimg
        index_count = index_count + 1
    print('index_sum:{}'.format(index_count))
    np.save('K:\\DeRaindrop\\DeNoisedrop\\Masklist.npy', masklist)

    print('data save finish !')


if __name__ == '__main__':

    args = get_args()

    model = Generator().cuda()
    model.load_state_dict(torch.load('pretrainedmodel.pt'))  # AG_LSTM_865363
    model.eval()
    if args.mode == 'demo':
        file = 'K:\\DeNoise\\sample_list2.txt'
        eval(file)
        # evaldata()
        # test()
        # getMask(file)
    elif args.mode == 'test':
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            print('Processing image: %s' % (input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            gt = cv2.imread(args.gt_dir + gt_list[i])
            img = align_to_four(img)
            gt = align_to_four(gt)
            result = predict(img)
            result = np.array(result, dtype='uint8')
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            print('PSNR is %.6f and SSIM is %.6f' % (cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.6f and SSIM is %.6f' %
              (cumulative_psnr / num, cumulative_ssim / num))
    else:
        print('Mode Invalid!')
