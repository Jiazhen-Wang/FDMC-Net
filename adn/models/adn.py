import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .base import Base, BaseTrain
from ..networks import ADN, NLayerDiscriminator, add_gan_loss
from ..utils import print_model, get_device
from skimage.restoration import denoise_tv_bregman
import math
import matplotlib.pyplot as plt
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)
from skimage.metrics import structural_similarity
from torchvision.utils import save_image


def apply_atv(img, weight=0.1):
    # 将图像转换为 float 类型（如果图像是 torch tensor，需要转换为 numpy 数组）
    img_float = img.squeeze().cpu().detach().numpy()
    denoised_img = denoise_tv_bregman(img_float, weight=weight)

    # 将去噪后的图像转换回 tensor 并保持原始维度
    denoised_img_tensor = torch.tensor(denoised_img).unsqueeze(0).unsqueeze(0)
    return denoised_img_tensor


# 2. 块稀疏正则化（Block Sparse Regularization）
def block_sparse_regularization(img, weight=0.1, block_size=(8, 8)):
    # 将图像转换为 numpy 数组（假设输入是 torch tensor）
    img_float = img.squeeze().cpu().detach().numpy()

    # 获取图像的尺寸
    h, w = img_float.shape
    bh, bw = block_size

    # 存储每个处理后的块
    denoised_blocks = []

    # 遍历图像中的每个块
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = img_float[i:i + bh, j:j + bw]
            # 对每个块应用 L1 正则化（简单地减去一个稀疏项）
            block = block - weight * np.sign(block)
            denoised_blocks.append(block)

    # 将处理后的块重新组合成图像
    denoised_img = np.zeros_like(img_float)
    block_idx = 0
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            denoised_img[i:i + bh, j:j + bw] = denoised_blocks[block_idx]
            block_idx += 1

    # 返回去噪后的图像作为 tensor
    denoised_img_tensor = torch.tensor(denoised_img).unsqueeze(0).unsqueeze(0)
    return denoised_img_tensor


class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, p, n, m):
        d_ap = self.l1(a, p.detach())
        d_an = self.l1(a, n.detach())
        d_am = self.l1(a, m.detach())
        loss = d_ap / (d_an + d_am + 1e-7)
        return loss
import copy
import scipy.io as sio
class ADNTrain(BaseTrain):
    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(ADNTrain, self).__init__(learn_opts, loss_opts)
        g_opts, d_opts = model_opts[g_type], model_opts[d_type]

        model_dict = dict(
            adn=lambda: ADN(**g_opts),
            nlayer=lambda: NLayerDiscriminator(**d_opts))

        self.model_g = self._get_trainer(model_dict, g_type)  # ADN generators
        self.model_dl = add_gan_loss(
            self._get_trainer(model_dict, d_type))  # discriminator for low quality image (with artifact)
        self.model_dh = add_gan_loss(
            self._get_trainer(model_dict, d_type))  # discriminator for high quality image (without artifact)

        loss_dict = dict(
            l1=nn.L1Loss,
            con=ContrastLoss,
            gl=(self.model_dl.get_g_loss, self.model_dl.get_d_loss),  # GAN loss for low quality image.
            gh=(self.model_dh.get_g_loss, self.model_dh.get_d_loss))  # GAN loss for high quality image

        # Create criterion for different loss types
        self.model_g._criterion["ll"] = self._get_criterion(loss_dict, self.wgts["ll"], "ll_")
        self.model_g._criterion["lh"] = self._get_criterion(loss_dict, self.wgts["lh"], "lh_")
        self.model_g._criterion["hh"] = self._get_criterion(loss_dict, self.wgts["hh"], "hh_")
        self.model_g._criterion["lhl"] = self._get_criterion(loss_dict, self.wgts["lhl"], "lhl_")
        self.model_g._criterion["hlh"] = self._get_criterion(loss_dict, self.wgts["hlh"], "hlh_")
        self.model_g._criterion["art"] = self._get_criterion(loss_dict, self.wgts["art"], "art_")
        self.model_g._criterion["gl"] = self._get_criterion(loss_dict, self.wgts["gl"])
        self.model_g._criterion["gh"] = self._get_criterion(loss_dict, self.wgts["gh"])
        self.model_g._criterion["cont"] = self._get_criterion(loss_dict, self.wgts["cont"], "cont")

        print_model(self)

    def _nonzero_weight(self, *names):
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if type(w[0]) is str: w = [w]
            for p in w: wgt = wgt + p[1]
        return wgt

    def _get_target_network(self):
        netT = copy.deepcopy(self.model_g)
        return netT

    def optimize(self, img_low, img_high):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        self.model_g._clear()
        # print(self.img_low.shape)
        self.pred_ll, self.pred_lh, self.pred_hl, self.pred_hh, self.pred_hlh, self.pred_lhl = self.model_g.forward1(self.img_low, self.img_high)

        # low -> low_l, low -> low_h
        if self._nonzero_weight("gl", "lh", "ll"):
            self.model_dl._clear()
            self.model_g._criterion["gl"](self.pred_lh, self.img_high)
            self.model_g._criterion["lh"](self.pred_lh, self.img_high)
            self.model_g._criterion["ll"](self.pred_ll, self.img_low)
            # self.model_g._criterion["cont"](self.content_low, self.content_high, self.artifacts_low, self.artifacts_high)
            # self.model_dl._update()

        # high -> high_l, high -> high_h
        if self._nonzero_weight("gh", "hh"):
            self.model_dh._clear()
            # self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
            self.model_g._criterion["gh"](self.pred_hl, self.img_low)
            self.model_g._criterion["hh"](self.pred_hh, self.img_high)
            # self.model_dh._update()

        # low_h -> low_h_l
        if self._nonzero_weight("lhl"):
            # self.pred_lhl = self.model_g.forward_hl(self.pred_hl, self.pred_lh)
            self.model_g._criterion["lhl"](self.pred_lhl, self.img_low)

        # high_l -> high_l_h
        if self._nonzero_weight("hlh"):
            # self.pred_hlh = self.model_g.forward_lh(self.pred_hl)
            self.model_g._criterion["hlh"](self.pred_hlh, self.img_high)

        # artifact
        if self._nonzero_weight("art"):
            ll = self.img_low if self.gt_art else self.pred_ll
            hh = self.img_high if self.gt_art else self.pred_hh
            self.model_g._criterion["art"](
                ll - self.pred_lh, self.pred_hl - hh)

        self.model_g._update()

        # print('111', self.model_dl.out1.weight.grad)
        # print(self.model_dl.out1.weight[0][0][0])

        self.model_dl._update()


        self.model_dh._update()

        # merge losses for printing
        self.loss = self._merge_loss(
            self.model_dl._loss, self.model_dh._loss, self.model_g._loss)

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"), ("lhl", "pred_lhl"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh"), ("hlh", "pred_hlh")]

        return self._get_visuals(lookup, n)

    def evaluate(self, loader, metrics):
        progress = tqdm(loader)
        res = defaultdict(lambda: defaultdict(float))
        cmt = 0
        p = 0
        p2 = 0
        s = 0
        s2 = 0
        psnr = []
        ssim = []
        save_root = './Baseline_FDM_CL_DMoEG_Supervised/motion_image/test_image'
        import os
        os.makedirs(save_root, exist_ok=True)

        for img_low, img_high, name in progress:
            # print(name)
            img_low, img_high= self._match_device(img_low, img_high)

            def to_numpy(*data):
                data = [loader.dataset.to_numpy(d, False) for d in data]
                return data[0] if len(data) == 1 else data

            pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh, content_low, artifacts_low, content_high, artifacts_high = self.model_g.forward1(img_low, img_high)
            pred_lh = torch.clamp(pred_lh, min=0.0, max=1.0)

            # pred_lh = apply_atv(pred_lh, weight=55)

            img_low, img_high, pred_ll, pred_lh, pred_hl = to_numpy(
                img_low, img_high, pred_ll, pred_lh, pred_hl)

            im_output = pred_lh
            im_gt = img_high
            im_input = img_low
            # sio.savemat(
            #     '/media/wangjiazhen/My Passport/MR-ART-Dataset/train/relabel_motion2_adn2/' + str(
            #         name[0]), {'img': im_output})
            # sio.savemat(
            #     '/media/wangjiazhen/My Passport/MR-ART-Dataset/train/remotion2_adn2/' + str(
            #         name[0]), {'img': pred_hl})
            pp = PSNR(im_output, im_gt)
            psnr.append(pp)
            # print(pp)
            pp2 = PSNR(im_input, im_gt)
            ppp2 = PSNR(im_input, pred_hl)
            # print(pp2)
            p += pp
            p2 += pp2

            ss = structural_similarity(im_output, im_gt,
                                       data_range=im_gt.max())
            ssim.append(ss)
            ss2 = structural_similarity(im_input, im_gt,
                                        data_range=im_gt.max())
            sss2 = structural_similarity(pred_hl, im_input,
                                                data_range=im_gt.max())
            s += ss
            s2 += ss2
            # HR_4x = HR[:,:,:,:,0].cpu()
            im_output = im_output
            # save_image(im_output.data,'6.png')
            # save_image(im_output.data,opt.save+'/'+name[-1])
            cmt += 1
            with open(save_root + "result.txt", "a") as f:
                f.write("\n")
                f.writelines(
                    'Data:{:s},\tPSNR:{:.4f},\tSSIM:{:.4f},\tPSNR_input:{:.4f},\tSSIM_input:{:.4f}, Sim_Motion_PSNR:{:.4f},\tSim_Motion_SSIM:{:.4f}'.format(
                        str(cmt), pp, ss, pp2, ss2, ppp2, sss2))

            # save_image(im_input, save_root + '/' + str(cmt) + '_motion.png')
            # save_image(im_output, save_root + '/' + str(cmt) + '_rec.png')
            # save_image(im_gt, save_root + '/' + str(cmt) + '_gt.png')
            plt.imsave(save_root + '/' + str(cmt) + '_motion.png', im_input, cmap='gray')
            plt.imsave(save_root + '/' + str(cmt) + '_rec.png', im_output, cmap='gray')
            plt.imsave(save_root + '/' + str(cmt) + '_gt.png', im_gt, cmap='gray')
            plt.imsave(save_root + '/' + str(cmt) + '_sim_motion.png', pred_hl, cmap='gray')
            plt.imsave(save_root + '/' + str(cmt) + '_res.png', np.abs(im_gt - im_output), cmap='gray')
            plt.imsave(save_root + '/' + str(cmt) + '_sim_res.png', np.abs(im_input - pred_hl), cmap='gray')

        # ssim=calculate_ssim_floder(dataset,opt.save)
        # ssim_input=calculate_ssim_floder(dataset,opt.input,mode='input')
        print("Average PSNR:", p / cmt)
        print("Average input PSNR:", p2 / cmt)
        print("Average SSIM:", s / cmt)
        print("Average Input SSIM:", s2 / cmt)

        print('Mean_cor_ssim:', np.average(ssim))
        print('Mean_cor_psnr:', np.average(psnr))

        print('Std_cor_ssim:', np.std(ssim))
        print('Std_cor_psnr:', np.std(psnr))


class ADNTest(Base):
    def __init__(self, g_type, **model_opts):
        super(ADNTest, self).__init__()

        g_opts = model_opts[g_type]
        model_dict = dict(adn=lambda: ADN(**g_opts))
        self.model_g = model_dict[g_type]()
        print_model(self)

    def forward(self, img_low):
        self.img_low = self._match_device(img_low)
        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)

        return self.pred_ll, self.pred_lh

    def evaluate(self, img_low, img_high, name=None):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        self.name = name

        self.pred_ll, self.pred_lh = self.model_g.forward1(self.img_low)
        self.pred_hl, self.pred_hh = self.model_g.forward2(self.img_low, self.img_high)
        self.pred_hlh = self.model_g.forward_lh(self.pred_hl)

    def get_pairs(self):
        return [
                   ("before", (self.img_low, self.img_high)),
                   ("after", (self.pred_lh, self.img_high))], self.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh")]
        func = lambda x: x * 0.5 + 0.5
        return self._get_visuals(lookup, n, func, False)