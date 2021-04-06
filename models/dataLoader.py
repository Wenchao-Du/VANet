import os
from torch.utils import data
from torchvision import  transforms
import h5py
import random
import numpy as np
import cv2
import scipy.io as sio

class dataset(data.Dataset):
    def __init__(self, root):
        if not os.path.exists(root):
            raise Exception('input root exist!')
        # filepath = os.path.join(root, 'mayoData.mat')
        self._gt_data_, self._label_data_, self._mask_data_, self._count_, self._testld_, self._testhd_, self._testNum_ = self.__loadimglist__(root)
        # self.__loadimglist__(root)
    def __load_matdata__(self, file):
        fp = h5py.File(file, 'r')
        hdata = fp['HData']
        ldata = fp['LData']
        label = fp['lable']
        gt_imgs = []
        gt_labels = []
        mask_labels = [] 
        test_ld = []
        test_hd = []           
        count = label.shape[1]
        sample_sum = 4000
        sample_list = random.sample(range(0, count), sample_sum)
        f2 = open('sample_list2.txt', 'w')
        indexlist = np.zeros([1, count], dtype=np.int32)
        print("start to create train dataset")
        for i, index in enumerate(sample_list):
            sample_index = int(index)
            indexlist[0, sample_index] = 1
            f2.write('{}\n'.format(sample_index))            
            hdmat = hdata[sample_index, :, :]
            hdmat = np.transpose(hdmat)
            ldmat = ldata[sample_index, :, :]
            ldmat = np.transpose(ldmat)
            diff_image = np.subtract(hdmat, ldmat)
            diff_image = np.abs(diff_image)
            mask_image = np.zeros(diff_image.shape, np.float32)
            meanvalue = np.mean(diff_image)
            mask_image[np.where(diff_image >= meanvalue)] = 1
            gtimgs, label_imgs, mask_images = self.__gen_patchdata__(ldmat, hdmat, mask_image)
            gt_imgs.extend(gtimgs)
            gt_labels.extend(label_imgs)
            mask_labels.extend(mask_images)
        f2.close()
        print("start to create test dataset")
        print('create test dataset')

        for k in range(indexlist.shape[1]):
            if indexlist[0, k] == 1:
                continue
            hdmat = hdata[k, :, :]
            hdmat = np.transpose(hdmat)
            ldmat = ldata[k, :, :]
            ldmat = np.transpose(ldmat)
            testimgs, testlabels, testmasks = self.__gen_patchdata__(ldmat, hdmat, None, patch_num = 2)
            test_ld.extend(testimgs)
            test_hd.extend(testlabels)
        # fp.close()
        print('test dataset create finish!')    
        Num = len(gt_imgs)
        test_Num = len(test_ld)
        print('test_num:{}'.format(test_Num))
        
        return gt_imgs, gt_labels, mask_labels, Num, test_ld, test_hd, test_Num


    def __gen_patchdata__(self, gt_img, label_img, mask_img, patch_num = 35, patch_size=64):
        # if gt_img.shape != label_img.shape or label_img.shape != mask_img.shape:
        #     raise Exception("input process data shape does not match!!\n")
        gt_img_patchs = []
        label_img_patchs = []
        mask_img_patchs = []
        img_h = gt_img.shape[0]
        img_w = gt_img.shape[1]
        np.random.seed(1234)
        for i in range(patch_num):
            seed_y = np.random.randint(int(patch_size / 2 + 1), img_w - int(patch_size / 2) - 1)
            seed_x = np.random.randint(int(patch_size / 2 + 1), img_h - int(patch_size / 2) - 1)
            s_img = gt_img[seed_x - int(patch_size / 2) : seed_x + int(patch_size / 2), seed_y - int(patch_size / 2) : seed_y + int(patch_size / 2)]
            s_label = label_img[seed_x - int(patch_size / 2) : seed_x + int(patch_size / 2), seed_y - int(patch_size / 2) : seed_y + int(patch_size / 2)]
            # ==================================================== #
            #  data selecting for training and testing
            # ==================================================== #
            ld_minval = np.min(s_img)
            ld_maxval = np.max(s_img)
            hd_minval = np.min(s_label)
            hd_maxval = np.max(s_label)
            if (ld_minval < -1) or (ld_maxval > 1) or (hd_minval) < -1 or (hd_maxval > 1):
                continue
            if mask_img is not None:
                s_mask = mask_img[seed_x - int(patch_size / 2) : seed_x + int(patch_size / 2), seed_y - int(patch_size / 2) : seed_y + int(patch_size / 2)]
                mask_img_patchs.append(s_mask)
            gt_img_patchs.append(s_img)
            label_img_patchs.append(s_label)
            
        
        return gt_img_patchs, label_img_patchs, mask_img_patchs

    def __load_image_(self, root):
        if not os.path.exists(root):
            raise Exception('input root does not exist !')
        l_path = os.path.join(root, 'Ldata')
        h_path = os.path.join(root, 'Hdata')

        gt_imgs = []
        gt_labels = []
        mask_labels = []
        for file in os.listdir(l_path):
            l_fpath = os.path.join(l_path, file)
            if not os.path.exists(l_fpath):
                continue
            
            ld_mat = cv2.imread(l_fpath, 0)
            ld_mat = np.array(ld_mat, dtype = np.float32) / 255
            
            h_fpath = os.path.join(h_path, file.replace('noise', 'clean'))
            if not os.path.exists(h_fpath):
                continue
            
            hd_mat = cv2.imread(h_fpath, 0)
            hd_mat = np.array(hd_mat, dtype = np.float32) / 255

            diff_image = np.subtract(hd_mat, ld_mat)
            mask_image = np.zeros(diff_image.shape, np.float32)
            meanvalue = np.mean(diff_image)
            mask_image[np.where(diff_image >= meanvalue)] = 1
            gtimgs, label_imgs, mask_images = self.__gen_patchdata__(ld_mat, hd_mat, mask_image)
            gt_imgs.extend(gtimgs)
            gt_labels.extend(label_imgs)
            mask_labels.extend(mask_images)
        print('crop traindata finish!')    
        Num = len(gt_imgs)
        return gt_imgs, gt_labels, mask_labels, Num

    def __len__(self):
        return len(self._count_)

    def __loadimglist__(self, file):
        fp = open(file, 'r')
        linedata = []
        for line in fp.readlines():
            line = line.strip()
            linedata.append(line)
        fp.close()

        fhy = h5py.File('F:\\Medical_Img\\net\\mayoData.mat')
        hdata = fhy['HData']
        ldata = fhy['LData']
        label = fhy['lable']
        gt_imgs = []
        gt_labels = []
        gt_masks = []
        indexlist = np.zeros((1, label.shape[1]), dtype = np.int32)
        for _, index in enumerate(linedata):
            index = int(index)
            indexlist[0, index] = 1
            hdmat = hdata[index, :, :]
            hdmat = np.transpose(hdmat)
            ldmat = ldata[index, :, :]
            ldmat = np.transpose(ldmat)
            hdmat = hdmat.astype(np.float32)
            ldmat = ldmat.astype(np.float32)
            
            diff_image = np.subtract(hdmat, ldmat)
            mask_image = np.zeros(diff_image.shape, np.float32)
            meanvalue = np.mean(diff_image)
            mask_image[np.where(diff_image >= meanvalue)] = 1

            ldmats, hdmats, maskmats = self.__gen_patchdata__(ldmat, hdmat, mask_image, patch_num=35, patch_size=64)
            gt_imgs.extend(ldmats)
            gt_labels.extend(hdmats)
            gt_masks.extend(maskmats)

        pairs = list(zip(gt_imgs, gt_labels, gt_masks))
        random.shuffle(pairs)
        gt_imgs, gt_labels, gt_masks = zip(*pairs)
        size = len(gt_imgs)
        

        testdata = []
        testlabel = []
        for index in range(indexlist.shape[1]):
            if indexlist[0, index] != 0:
                continue
            thdmat = hdata[index, :, :]
            thdmat = np.transpose(thdmat)
            tldmat = ldata[index, :, :]
            tldmat = np.transpose(tldmat)
            thdmat = thdmat.astype(np.float32)
            tldmat = tldmat.astype(np.float32)
            tldmats, thdmats, tmaskmats = self.__gen_patchdata__(tldmat, thdmat, None, patch_num= 4, patch_size=64)
            testdata.extend(tldmats)
            testlabel.extend(thdmats)
            # testmask.extend(tmaskmats)
        fhy.close()
        print(len(testdata))
        testpairs = list(zip(testdata, testlabel))
        random.shuffle(testpairs)
        testdata, testlabel = zip(*testpairs)
        testsize = len(testdata)
        
        return  gt_imgs, gt_labels, gt_masks, size, testdata, testlabel, testsize

    def __readh5__(self, file):
        file = h5py.File(file, 'r+')
        data = file['data']
        label = file['label']
        print(data.shape)
        print(label.shape)
        mat = data[100, 0, :,:]
        cv2.imshow("t", mat)
        cv2.waitKey(0)
            
if __name__ == '__main__':
    file = 'K:\\DeRaindrop\\DeNoisedrop\\sample_list2.txt'
    # file = 'K:\\DeRaindrop\\traindata-0.h5'
    dataloader = dataset(file)
    

        