import torch
import torchvision
import os
import random
import logging
import time
from torch import optim
from torch import nn
from torch.autograd import Variable
from models.AG import Generator
from models.AD import Discriminator
from torch.nn.functional import upsample_bilinear
from models.vgg_init import WGAN_VGG_FeatureExtractor
from models.dataLoader import dataset
import numpy as np
from metrics import compare_psnr, compare_ssim


class Solver(object):
    def __init__(self, config):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.VGG_model = None
        self.beta1 = config.TRAIN.beta1
        self.beta2 = config.TRAIN.beta2
        self.data_loader = dataset(config.TRAIN.root)
        self.num_epochs = config.TRAIN.epochs
        self.batch_size = config.TRAIN.batch_size
        self.sample_size = config.TRAIN.sample_size
        self.g_lr = config.TRAIN.g_lr
        self.d_lr = config.TRAIN.d_lr
        self.bceloss = nn.BCELoss()
        self.p_loss = nn.L1Loss()
        self.log_step = config.TRAIN.log_step
        self.sample_step = config.TRAIN.sample_step
        self.sample_path = config.TRAIN.sample_path
        self.model_path = config.TRAIN.model_path
        self.test_step = config.TRAIN.test_step
        self.test_batchsize = config.TRAIN.test_batchsize
        self.build_model(config)
        self.logging = logging
        self._print_log_()

    def _print_log_(self):
        times = time.localtime(time.time())
        filename = str(times.tm_year)+ "-" +str(times.tm_mon) + "-" + str(times.tm_mday) + "-" \
                + str(times.tm_hour) + "-" + str(times.tm_min) + "-" + str(times.tm_sec) + ".log"
        logging.basicConfig(
            filename=filename,
            filemode="w",
            format=
            '%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self.logging.basicConfig()
        self.logging.info("print train param Log :")
        self.logging.info("generator learning_rate:{}".format(self.g_lr))
        self.logging.info("discriminator learning_rate:{}".format(self.d_lr))
        self.logging.info("training batch size:{}".format(self.batch_size))
        self.logging.info("training epoches:{}".format(self.num_epochs))
        self.logging.info("training log step:{}".format(self.log_step))
        self.logging.info("test log step:{}".format(self.test_step))
        self.logging.info("test batchsize:{}".format(self.test_batchsize))

    def build_model(self, config):

        self.VGG_model = WGAN_VGG_FeatureExtractor()
        self.generator = Generator()
        # self.discriminator = Discriminator()
        if torch.cuda.is_available():
            self.generator.cuda()
            # self.discriminator.cuda()
            # self.bceloss.cuda()
            # self.VGG_model.cuda()
            self.p_loss.cuda()
        if config.TRAIN.weights is not None:
            if not os.path.exists(config.TRAIN.weights):
                raise Exception('fine tune weights is not exist !')
            self.generator.load_state_dict(torch.load(config.TRAIN.weights))
            # self.discriminator.load_state_dict(torch.load(config.TRAIN.weights.replace('generator', 'discriminator')))
            print('pretrained model load finish!!')
        # ======================================================
        #  define optimizer
        # ======================================================
        self.g_optimizer = optim.Adam(self.generator.parameters(), self.g_lr,
                                      [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.d_lr, [self.beta1, self.beta2])
        self.decayG = optim.lr_scheduler.MultiStepLR(self.g_optimizer,
                                                     [2, 5, 10],
                                                     gamma=0.5)  # step lr
        self.decayD = optim.lr_scheduler.MultiStepLR(self.d_optimizer,
                                                     [2, 5, 10],
                                                     gamma=0.5)  # step lr

    def to_variable(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        # self.discriminator.zero_grad()
        self.generator.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def cal_attention_loss(self, input, label):
        loss = torch.tensor(0.0).cuda()
        lambda_ = [0.125, 0.25, 0.5, 0.65, 0.8, 0.9, 1.0]
        for index, mask_map in enumerate(input):
            l1_loss = lambda_[index] * torch.mean(
                (label - mask_map)**2)  # L2 loss
            loss = loss + l1_loss
        return loss

    def cal_autoencoder_loss(self, input, label):
        H = label.size(2)
        W = label.size(3)
        srcmap = label
        scalemap2 = upsample_bilinear(label, size=[int(H / 2), int(W / 2)])
        scalemap4 = upsample_bilinear(label, size=[int(H / 4), int(W / 4)])
        label_list = list([scalemap4, scalemap2, srcmap])
        lambda_i = [0.25, 0.5, 1.0]
        if len(input) != len(label_list):
            raise Exception('input feature map num does not match!')
        loss = 0
        for index, in_map in enumerate(input):
            mse_loss = torch.mean(
                (label_list[index] - in_map)**2) * lambda_i[index]
            loss = loss + mse_loss
        return loss

    def cal_vgg_loss(self, input_tensor, label_tensor):
        if len(input_tensor) != len(label_tensor):
            raise Exception('input feature map does not match')
        per_loss = torch.zeros([1, len(input_tensor)])
        for index, feas in enumerate(label_tensor):
            per_loss[0, index] = torch.mean(
                (label_tensor[index] - input_tensor[index])**2)
        loss = torch.mean(per_loss)
        return loss

    def cal_Lmap_loss(self, input_tensor, attention_map, isLabel=True):
        if input_tensor.size() != attention_map.size():
            raise Exception('input map shape does not match!')
        loss = 0
        if isLabel:
            attmap = torch.zeros(attention_map.size(),
                                 dtype=torch.float32).cuda()
            loss = torch.mean((input_tensor - attmap)**2)
        else:
            loss = torch.mean((input_tensor - attention_map)**2)
        return loss

    def calc_gradient_penalty(self, Net, real_data, fake_data):
        alpha = torch.rand(
            self.batch_size,
            1,
        )
        alpha = alpha.expand(self.batch_size,
                             int(real_data.nelement() /
                                 self.batch_size)).contiguous().view(
                                     self.batch_size, 1, 64, 64)
        if torch.cuda.is_available():
            alpha = alpha.cuda()
        else:
            alpha = alpha
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        if torch.cuda.is_available():
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = Net(interpolates)[-1]

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda()
            if torch.cuda.is_available() else torch.ones(
                disc_interpolates.size()),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        gradient_penalty = (
            (gradients.view(gradients.size()[0], -1).norm(2, dim=1) - 1)**
            2).mean() * LAMBDA

        return gradient_penalty

    def train(self):
        Data_Num = self.data_loader._count_
        count = 0
        testpairs = list(
            zip(self.data_loader._testld_, self.data_loader._testhd_))
        random.shuffle(testpairs)
        self.data_loader._testld_, self.data_loader._testhd_ = zip(*testpairs)
        testindex = 0
        for epoch in range(self.num_epochs):
            pairs = list(
                zip(self.data_loader._gt_data_, self.data_loader._label_data_,
                    self.data_loader._mask_data_))
            random.shuffle(pairs)
            self.logging.info('input data shuffle finish !')
            self.data_loader._gt_data_, self.data_loader._label_data_, self.data_loader._mask_data_ = zip(
                *pairs)
            for index in range(0, Data_Num, self.batch_size):
                gt_data_list = self.data_loader._gt_data_[index:index +
                                                          self.batch_size]
                label_data_list = self.data_loader._label_data_[index:index +
                                                                self.
                                                                batch_size]
                mask_data_list = self.data_loader._mask_data_[index:index +
                                                              self.batch_size]
                gt_imgs = np.zeros(
                    (self.batch_size, 1, gt_data_list[0].shape[0],
                     gt_data_list[0].shape[1]))
                label_imgs = np.zeros(
                    (self.batch_size, 1, label_data_list[0].shape[0],
                     label_data_list[0].shape[1]))
                mask_imgs = np.zeros(
                    (self.batch_size, 1, mask_data_list[0].shape[0],
                     mask_data_list[0].shape[1]))
                for k, tmp in enumerate(gt_data_list):
                    gt_imgs[k, 0, :, :] = tmp

                for k, tmp in enumerate(label_data_list):
                    label_imgs[k, 0, :, :] = tmp

                for k, tmp in enumerate(mask_data_list):
                    mask_imgs[k, 0, :, :] = tmp

                gt_imgs = torch.from_numpy(gt_imgs).float()
                label_imgs = torch.from_numpy(label_imgs).float()
                mask_imgs = torch.from_numpy(mask_imgs).float()

                ldata = self.to_variable(gt_imgs)
                hdata = self.to_variable(label_imgs)
                maskdata = self.to_variable(mask_imgs)

                #======================train D=============================#

                mask_list, scaleframe1, scaleframe2, fake_images = self.generator(
                    ldata)

                src_outputs = self.discriminator(hdata)
                pre_outputs = self.discriminator(fake_images)

                Lmap_loss = self.cal_Lmap_loss(src_outputs[0], mask_list[-1], isLabel = True) + \
                    self.cal_Lmap_loss(pre_outputs[0], mask_list[-1], isLabel= False)
                pre_real = torch.ones_like(src_outputs[-1]).cuda()
                pre_fake = torch.zeros_like(src_outputs[-1]).cuda()
                entropy_loss = self.bceloss(src_outputs[-1],
                                            pre_real) + self.bceloss(
                                                pre_outputs[-1], pre_fake)
                # # # # ============================================================
                # # # #  define Discripter loss : d_loss = dgan + Lmap_loss
                # # # # ============================================================

                d_loss = entropy_loss + 0.05 * Lmap_loss
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                # #======================train G================================#

                mask_list, scaleframe1, scaleframe2, fake_images = self.generator(
                    ldata)  # mask_list
                generator_output = [scaleframe1, scaleframe2, fake_images]

                vgg_pre_out = self.VGG_model(hdata)
                vgg_src_out = self.VGG_model(generator_output[-1])

                attention_rnn_loss = self.cal_attention_loss(
                    mask_list, maskdata)
                autocode_loss = self.cal_autoencoder_loss(
                    generator_output, hdata)
                vgg_loss = self.p_loss(vgg_pre_out, vgg_src_out)
                pre_outputs = self.discriminator(generator_output[-1])

                l_gan_loss = self.bceloss(pre_outputs[-1], pre_real)
                #=============================================================
                #     Lg = Lgan + Latt + Lm + Lp
                #=============================================================
                g_loss = 0.01 * autocode_loss.cuda() + 0.5 * vgg_loss.cuda(
                ) + 0.005 * l_gan_loss.cuda() + attention_rnn_loss
                # g_loss = autocode_loss.cuda() + attention_rnn_loss
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                count = count + 1
                # print log info
                if (
                        count
                ) % self.log_step == 0:  # src_outputs[-1].data[0], pre_outputs[-1].data[0],   d_real_loss:%.4f,' 'd_fake_loss:%.4f
                    self.logging.info(
                        'Epoch[%d/%d], step[%d/%d], g_loss:%.4f' %
                        (epoch + 1, self.num_epochs, index, Data_Num,
                         g_loss.item()))
                # test data error  print
                if (count) % self.test_step == 0:
                    if testindex + self.test_batchsize < self.data_loader._testNum_:
                        testindex = testindex
                    else:
                        testindex = 0
                    testdata_list = self.data_loader._testld_[
                        testindex:testindex + self.test_batchsize]
                    testlabel_list = self.data_loader._testhd_[
                        testindex:testindex + self.test_batchsize]
                    test_imgs = np.zeros(
                        (self.test_batchsize, 1, testdata_list[0].shape[0],
                         testdata_list[0].shape[1]))
                    test_labels = np.zeros(
                        (self.test_batchsize, testlabel_list[0].shape[0],
                         testlabel_list[0].shape[1]))
                    for k, tmp in enumerate(testdata_list):
                        test_imgs[k, 0, :, :] = tmp
                    for k, tmp in enumerate(testdata_list):
                        test_labels[k, :, :] = tmp
                    inputs = torch.from_numpy(test_imgs).float()
                    testldata = self.to_variable(inputs)
                    testfake_images = self.generator(testldata)[-1]
                    testfake_images = testfake_images.cpu().data
                    testfake_images = testfake_images.numpy()
                    test_batch = testfake_images.shape[0]
                    test_labels = test_labels.astype(np.float32)
                    psnr = 0
                    ssim = 0
                    for tb in range(test_batch):
                        psnr = psnr + compare_psnr(
                            test_labels[tb, :, :], testfake_images[tb,
                                                                   0, :, :])
                        ssim = ssim + compare_ssim(
                            test_labels[tb, :, :], testfake_images[tb,
                                                                   0, :, :])
                    self.logging.info("[{} / {}]  psnr:{}, ssim:{}".format(
                        testindex * self.test_batchsize,
                        self.data_loader._testNum_, psnr / test_batch,
                        ssim / test_batch))
                    testindex = testindex + 1

                # save train model
                if (count) % self.sample_step == 0:
                    if not os.path.exists(self.model_path):
                        os.makedirs(self.model_path)
                    g_path = os.path.join(
                        self.model_path,
                        'AG_LSTM_%d.pkl' % (epoch * Data_Num + count + 1))
                    # d_path = os.path.join(
                    #     self.model_path, 'AD_%d.pkl' % (epoch * Data_Num + count + 1))
                    self.logging.info('start save model to :{}'.format(g_path))
                    torch.save(self.generator.state_dict(), g_path)
                    torch.save(self.discriminator.state_dict(), d_path)
            self.decayD.step(epoch)
            self.decayG.step(epoch)
        self.logging.info('start save final model to model path !')
        torch.save(self.generator.state_dict(),
                   os.path.join(self.model_path, 'AG_LSTM_final.pkl'))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(self.model_path, 'AD_final.pkl'))
