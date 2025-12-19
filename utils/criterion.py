#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import torch
from torch.nn.modules.loss import _WeightedLoss

import math as mt




def get_loss_func(name, component, normalizer):
    if name == 'rel2':
        return RelLpLoss(p=2,component=component, normalizer=normalizer)
    elif name == "rel1":
        return RelLpLoss(p=1,component=component, normalizer=normalizer)
    elif name == 'l2':
        return LpLoss(p=2, component=component, normalizer=normalizer)
    elif name == "l1":
        return LpLoss(p=1, component=component, normalizer=normalizer)
    else:
        raise NotImplementedError

class SimpleLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True,return_comps = False):
        super(SimpleLpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.return_comps = return_comps



    def forward(self, x, y, mask=None):
        num_examples = x.size()[0]

        # Lp loss 1
        if mask is not None:##TODO: will be meaned by n_channels for single channel data
            x = x * mask
            y = y * mask

            ## compute effective channels
            # msk_channels = mask.sum(dim=(1,2,3),keepdim=False).count_nonzero(dim=-1) # B, 1
            msk_channels = mask.sum(dim=list(range(1, mask.ndim-1)),keepdim=False).count_nonzero(dim=-1) # B, 1
        else:
            msk_channels = x.shape[-1]

        diff_norms = torch.norm(x.reshape(num_examples,-1, x.shape[-1]) - y.reshape(num_examples,-1,x.shape[-1]), self.p, dim=1)    ##N, C
        y_norms = torch.norm(y.reshape(num_examples,-1, y.shape[-1]), self.p, dim=1) + 1e-8

        if self.reduction:
            if self.size_average:
                    return torch.mean(diff_norms/y_norms)          ## deprecated
            else:
                return torch.sum(torch.sum(diff_norms/y_norms, dim=-1) / msk_channels)    #### go this branch
        else:
            return torch.sum(diff_norms/y_norms, dim=-1) / msk_channels
        ## Lp loss 2, channel average
        # diff_norms = torch.norm(x.reshape(num_examples, -1, x.shape[-1]) - y.reshape(num_examples, -1, x.shape[-1]),self.p, 1)
        # y_norms = torch.norm(y.reshape(num_examples, -1, x.shape[-1]), self.p, 1)
        # if self.reduction:
        #     if self.size_average:
        #         return torch.mean(diff_norms / y_norms)
        #     else:
        #         return torch.sum(torch.mean(diff_norms / y_norms,dim=1))  #### go this branch
        if self.return_comps:
            if self.size_average:
                return torch.mean(diff_norms/y_norms,dim=0)
            else:
                return torch.sum(diff_norms/y_norms)



class LpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(LpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)

        self.regularizer = regularizer
        self.normalizer = normalizer

    def _lp_losses(self, pred, target):
        if self.component == 'all':
            losses = ((pred - target).view(pred.shape[0],-1,pred.shape[-1]).abs() ** self.p).mean(dim=1) ** (1 / self.p)
            metrics = losses.mean(dim=0).clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            losses = ((pred - target).view(pred.shape[0],-1,pred.shape[-1]).abs() ** self.p).mean(dim=1) ** (1 / self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, pred, target):

        #### only for computing metrics

        loss, metrics = self._lp_losses(pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component, inverse=True), self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics

class RelLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(RelLpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component in ['all' , 'all-reduce'] else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer

    ### all reduce is used in temporal cases, use only one metric for all components
    def _lp_losses(self, pred, target):
        if (self.component == 'all') or (self.component == 'all-reduce'):
            err_pool = ((pred - target).view(pred.shape[0], -1, pred.shape[-1]).abs()**self.p).sum(dim=1,keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1]).abs()**self.p).sum(dim=1,keepdim=False)
            losses = (err_pool / target_pool)**(1/ self.p)
            if self.component == 'all':
                # metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
                metrics = losses.mean(dim=0).unsqueeze(0).clone().cpu().detach().numpy()  # 1, n

            else:
                # metrics = losses.mean().clone().detach().cpu().numpy()
                metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()   # 1, 1

        else:
            assert self.component <= target.shape[1]
            err_pool = ((pred - target[...,self.component]).view(pred.shape[0], -1, pred.shape[-1]).abs() ** self.p).sum(dim=1,keepdim=False)
            target_pool = (target.view(target.shape[0], -1, target.shape[-1])[...,self.component].abs() ** self.p).sum(dim=1, keepdim=False)
            losses = (err_pool / target_pool)**(1/ self.p)
            # metrics = losses.mean().clone().detach().cpu().numpy()
            metrics = losses.mean().unsqueeze(0).clone().cpu().detach().numpy()


        loss = losses.mean()

        return loss, metrics


    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):
        loss, metrics = self._lp_losses(pred, target)

        ### only for computing metrics
        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)
            _ , metrics = self._lp_losses(ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)


        return loss, reg, metrics


class RFNELoss(_WeightedLoss):
    '''
    RFNE(y, y_hat) = Frobenius_norm(y-y_hat) / Frobenius_norm(y)
    y: target, (batch, nx^i..., timesteps, nc)
    y_hat: prediction, (batch, nx^i..., timesteps, nc)
    '''
    def forward(self, pred, target):
        dims = target.size()
        error_norm = torch.norm(pred - target, dim=dims[1:-2])
        target_norm = torch.norm(target, dim=dims[1:-2])
        return torch.mean(error_norm / target_norm)


class Evaluator(_WeightedLoss):
    def __init__(self, temporal=False, griddata=False, component=0,  normalizer=None, ilow=4, ihigh=12):
        super(Evaluator, self).__init__()


        self.component = component if component in ['all', 'all-reduce'] else int(component)
        self.normalizer = normalizer
        self.temporal = temporal
        self.griddata = griddata
        self.ilow = ilow
        self.ihigh = ihigh





    ### pred, target: B, N1, N2..., Nm, C-> B, C
    def forward(self, pred, target):


        with torch.no_grad():
            ### only for computing metrics
            if self.normalizer is not None:
                pred, target = self.normalizer.transform(pred, component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)

            if self.component not in ['all', 'all-reduce']:
                target = target[..., self.component]
                pred, target = pred.unsqueeze(-1), target.unsqueeze(-1)

            metrics = {}
            ## 1, C
            _pred, _target = pred.view(pred.shape[0], -1, pred.shape[-1]), target.view(target.shape[0], -1, target.shape[-1])
            nmae = ((_pred - _target).abs().sum(dim=1, keepdim=False) / (_target.abs().sum(dim=1, keepdim=False))).mean(dim=0, keepdim=True)
            nmse = torch.sqrt(((_pred - _target) ** 2).sum(dim=1, keepdim=False) / ((_target) ** 2).sum(dim=1, keepdim=False)).mean(dim=0, keepdim=True)
            nmxe = (torch.amax((_pred - _target).abs(), dim=1, keepdim=False) / torch.amax(_target.abs(), dim=1, keepdim=False)).mean(dim=0, keepdim=True)

            metrics.update({'nmae': nmae, 'nmse': nmse, 'nmxe': nmxe})
            if self.temporal:
                _pred, _target = pred.view(pred.shape[0], -1, pred.shape[-2], pred.shape[-1]), target.view(target.shape[0], -1, target.shape[-2], target.shape[-1])

                nmae_t = ((_pred - _target).abs().sum(dim=1, keepdim=False) / (_target.abs().sum(dim=1, keepdim=False))).mean(dim=0, keepdim=True)
                nmse_t = torch.sqrt(((_pred - _target) ** 2).sum(dim=1, keepdim=False) / (_target ** 2).sum(dim=1, keepdim=False)).mean(dim=0, keepdim=True)
                nmxe_t = (torch.amax((_pred - _target).abs(), dim=1, keepdim=False) / torch.amax(_target.abs(), dim=1, keepdim=False)).mean(dim=0, keepdim=True)

                metrics.update({'nmae_t': nmae_t, 'nmse_t': nmse_t, 'nmxe_t': nmxe_t})
            if self.griddata:
                bdmse, fmse_low, fmse_mid, fmse_high = compute_fourier_error(pred, target, self.ilow, self.ihigh)
                metrics.update({'bdmse': bdmse, 'fmse_low': fmse_low, 'fmse_mid': fmse_mid, 'fmse_high': fmse_high})

            metrics = {key: value.cpu().numpy() for key, value in metrics.items()}
        return metrics




_SPECTRAL_BIN_CACHE = {}


def _get_radial_bins_2d(nx2, ny2, kmax, device):
    key = ('2d', nx2, ny2, kmax, device.type, device.index if device.index is not None else -1)
    cached = _SPECTRAL_BIN_CACHE.get(key, None)
    if cached is not None:
        return cached

    ii = torch.arange(nx2, device=device, dtype=torch.float32).view(nx2, 1)
    jj = torch.arange(ny2, device=device, dtype=torch.float32).view(1, ny2)
    bin_idx = torch.floor(torch.sqrt(ii * ii + jj * jj)).to(torch.long)
    mask_flat = (bin_idx < kmax).reshape(-1)
    bin_idx_flat = bin_idx.reshape(-1)[mask_flat]

    _SPECTRAL_BIN_CACHE[key] = (bin_idx_flat, mask_flat)
    return bin_idx_flat, mask_flat


def _get_radial_bins_3d(nx2, ny2, nz2, kmax, device):
    key = ('3d', nx2, ny2, nz2, kmax, device.type, device.index if device.index is not None else -1)
    cached = _SPECTRAL_BIN_CACHE.get(key, None)
    if cached is not None:
        return cached

    ii = torch.arange(nx2, device=device, dtype=torch.float32)
    jj = torch.arange(ny2, device=device, dtype=torch.float32)
    kk = torch.arange(nz2, device=device, dtype=torch.float32)
    I, J, K = torch.meshgrid(ii, jj, kk)
    bin_idx = torch.floor(torch.sqrt(I * I + J * J + K * K)).to(torch.long)
    mask_flat = (bin_idx < kmax).reshape(-1)
    bin_idx_flat = bin_idx.reshape(-1)[mask_flat]

    _SPECTRAL_BIN_CACHE[key] = (bin_idx_flat, mask_flat)
    return bin_idx_flat, mask_flat


def compute_fourier_error(pred, target, iLow, iHigh, if_mean=False):
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE
    err_mean = torch.sqrt(torch.mean((pred.reshape(nb, nc, -1, nt) - target.reshape(nb, nc, -1, nt)) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.reshape(nb, nc, -1, nt) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.reshape(nb, nc, -1, nt), dim=2) - torch.sum(target.reshape(nb, nc, -1, nt), dim=2)) ** 2,
        dim=0))
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(torch.max(
        torch.abs(pred.reshape(nb, nc, -1, nt) - target.reshape(nb, nc, -1, nt)), dim=2)[0], dim=0)[0]

    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        err_BD = torch.sum(err_BD_x.reshape(nb, -1, nt), dim=-2) \
                 + torch.sum(err_BD_y.reshape(nb, -1, nt), dim=-2) \
                 + torch.sum(err_BD_z.reshape(nb, -1, nt), dim=-2)
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)

    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx
    elif len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        err_sq = torch.abs(pred_F - target_F) ** 2

        nx2, ny2 = nx // 2, ny // 2
        kmax = min(nx2, ny2)
        bin_idx_flat, mask_flat = _get_radial_bins_2d(nx2, ny2, kmax, pred.device)

        err_region = err_sq[:, :, :nx2, :ny2, :]
        src = err_region.permute(0, 1, 4, 2, 3).reshape(nb, nc, nt, -1)
        src = src[..., mask_flat]

        out = torch.zeros(nb, nc, nt, kmax, device=pred.device, dtype=src.dtype)
        out.index_add_(3, bin_idx_flat, src)

        err_F = out.permute(0, 1, 3, 2)
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny)
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        err_sq = torch.abs(pred_F - target_F) ** 2

        nx2, ny2, nz2 = nx // 2, ny // 2, nz // 2
        kmax = min(nx2, ny2, nz2)
        bin_idx_flat, mask_flat = _get_radial_bins_3d(nx2, ny2, nz2, kmax, pred.device)

        err_region = err_sq[:, :, :nx2, :ny2, :nz2, :]
        src = err_region.permute(0, 1, 5, 2, 3, 4).reshape(nb, nc, nt, -1)
        src = src[..., mask_flat]

        out = torch.zeros(nb, nc, nt, kmax, device=pred.device, dtype=src.dtype)
        out.index_add_(3, bin_idx_flat, src)

        err_F = out.permute(0, 1, 3, 2)
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz)

    fmse_low = torch.mean(_err_F[:, :iLow], dim=1).T  # low freq
    fmse_mid = torch.mean(_err_F[:, iLow:iHigh], dim=1).T  # middle freq
    fmse_high = torch.mean(_err_F[:, iHigh:], dim=1).T

    # err_F = torch.zeros([nc, 3, nt]).to(pred.device)
    # err_F[:, 0] += torch.mean(_err_F[:, :iLow], dim=1)  # low freq
    # err_F[:, 1] += torch.mean(_err_F[:, iLow:iHigh], dim=1)  # middle freq
    # err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)  # high freq

    # if if_mean:
    #     return torch.mean(err_RMSE, dim=[0, -1]), \
    #            torch.mean(err_nRMSE, dim=[0, -1]), \
    #            torch.mean(err_CSV, dim=[0, -1]), \
    #            torch.mean(err_Max, dim=[0, -1]), \
    #            torch.mean(err_BD, dim=[0, -1]), \
    #            torch.mean(err_F, dim=[0, -1])
    # else:
    #     return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F
    return err_BD, fmse_low, fmse_mid, fmse_high    ## T, C, ### T, C


class SpectralBinnedLoss(_WeightedLoss):
    def __init__(self, iLow=4, iHigh=12, weight_mid=1.0, weight_high=0.5):
        super(SpectralBinnedLoss, self).__init__()
        self.iLow = iLow
        self.iHigh = iHigh
        self.weight_mid = weight_mid
        self.weight_high = weight_high

    def forward(self, pred, target):
        err_BD, fmse_low, fmse_mid, fmse_high = compute_fourier_error(pred, target, self.iLow, self.iHigh)
        loss_mid = fmse_mid.mean()
        loss_high = fmse_high.mean()
        return self.weight_mid * loss_mid + self.weight_high * loss_high


if __name__ == "__main__":
    x = torch.randn([8, 11, 12, 4, 3])
    y = torch.randn([8, 11, 12, 4, 3])

    evaluator = Evaluator(temporal=True, griddata=True, component='all', normalizer=None)
    metrics = evaluator(x, y)
    print(metrics)
    # for key, value in metrics.items():
    #     print(key, value.shape)
