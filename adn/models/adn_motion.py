import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .base import Base, BaseTrain
from ..networks import ADN, NLayerDiscriminator, add_gan_loss, ADN1
from ..utils import print_model, get_device
import math



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


class ADNTrain(BaseTrain):
    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(ADNTrain, self).__init__(learn_opts, loss_opts)
        g_opts, d_opts = model_opts[g_type], model_opts[d_type]

        model_dict = dict(
            adn=lambda: ADN1(**g_opts),
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

        # print_model(self)

    def _nonzero_weight(self, *names):
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if type(w[0]) is str: w = [w]
            for p in w: wgt = wgt + p[1]
        return wgt

    def produce_pseudo_label(self, x, y, teacher_model):
        with torch.no_grad():
            pslabel_list = []

            for i in range(4):
                x1, x2, y1, y2, y2_rec, lhl = teacher_model.forward1(x, y)
                pslabel_list.append(x2)

            stacked_pslabel = torch.stack(pslabel_list)
            mea = torch.mean(stacked_pslabel, dim=0)
            var = torch.var(stacked_pslabel, unbiased=False, dim=0)
            confidence = 1 - torch.sigmoid(var / 0.0001)
            return mea, confidence

    def optimize(self, img_low, img_high, teacher_model):
        self.img_low, self.img_high = self._match_device(img_low, img_high)
        self.model_g._clear()
        self.pred_ll, self.pred_lh, self.pred_hl, self.pred_hh, self.pred_hlh, self.pred_lhl = self.model_g.forward1(
            self.img_low, self.img_high)
        self.pseudo_label, self.confidence = self.produce_pseudo_label(
            self.img_low, self.img_high, teacher_model)

        # low -> low_l, low -> low_h
        if self._nonzero_weight("gl", "lh", "ll"):
            self.model_dl._clear()
            self.model_g._criterion["gl"](self.pred_lh, self.img_high)
            # self.model_g._criterion["lh"](self.pred_lh, self.img_high)
            self.model_g._criterion["lh"](self.pred_lh * self.confidence, self.pseudo_label * self.confidence)
            self.model_g._criterion["ll"](self.pred_ll, self.img_low)


        # high -> high_l, high -> high_h
        if self._nonzero_weight("gh", "hh"):
            self.model_dh._clear()
            self.model_g._criterion["gh"](self.pred_hl, self.img_low)
            self.model_g._criterion["hh"](self.pred_hh, self.img_high)
            # self.model_dh._update()

        # low_h -> low_h_l
        if self._nonzero_weight("lhl"):
            self.model_g._criterion["lhl"](self.pred_lhl, self.img_low)

        # high_l -> high_l_h
        if self._nonzero_weight("hlh"):
            self.model_g._criterion["hlh"](self.pred_hlh, self.img_high)

        # artifact
        if self._nonzero_weight("art"):
            ll = self.img_low if self.gt_art else self.pred_ll
            hh = self.img_high if self.gt_art else self.pred_hh
            self.model_g._criterion["art"](
                ll - self.pred_lh, self.pred_hl - hh)

        self.model_g._update()

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


        psnrs = []
        ssims = []
        psnrs_in = []
        ssims_in = []


        for img_low, img_high, name in progress:
            # print(name)
            img_low, img_high = self._match_device(img_low, img_high)

            def to_numpy(*data):
                data = [loader.dataset.to_numpy(d, False) for d in data]
                return data[0] if len(data) == 1 else data


            pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh, lhl = self.model_g.forward1(
                img_low, img_high)

            img_low, img_high, pred_ll, pred_lh, pred_hl = to_numpy(
                img_low, img_high, pred_ll, pred_lh, pred_hl)


            im_output = pred_lh
            im_gt = img_high
            im_input = img_low

            pp = PSNR(im_output, im_gt)
            psnrs.append(pp)
            # print(pp)
            pp2 = PSNR(im_input, im_gt)
            psnrs_in.append(pp2)
            ppp2 = PSNR(im_input, pred_hl)

            ss = structural_similarity(im_output, im_gt,
                                       data_range=im_gt.max())
            ssims.append(ss)
            ss2 = structural_similarity(im_input, im_gt,
                                        data_range=im_gt.max())
            ssims_in.append(ss2)
            sss2 = structural_similarity(pred_hl, im_input,
                                         data_range=im_gt.max())


        print('Mean_cor_ssim:', np.average(ssims))
        print('Mean_cor_psnr:', np.average(psnrs))

        print('Std_cor_ssim:', np.std(ssims))
        print('Std_cor_psnr:', np.std(psnrs))

        print('Mean_cor_ssim_input:', np.average(ssims_in))
        print('Mean_cor_psnr_input:', np.average(psnrs_in))

        print('Std_cor_ssim_input:', np.std(ssims_in))
        print('Std_cor_psnr_input:', np.std(psnrs_in))





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
