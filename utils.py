import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import scipy.io as sio
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr

def generate_Phi(mask_path, batch_size, ch):
    mask = sio.loadmat(mask_path)
    mask = torch.from_numpy(mask['mask'])
    H, W = mask.shape
    C = ch
    step = 2
    Phi = torch.zeros([H, W + (C - 1) * step, C])
    for i in range(C):
        Phi[:, i*step:i*step+W, i] = mask
    Phi = Phi.permute(2, 0, 1) # C, H, W
    nC, H, W = Phi.shape
    if batch_size > 1:
        Phi_batch = Phi.expand([batch_size, nC, H, W]).cuda().float()
    else:
        Phi_batch = Phi.unsqueeze(0).cuda().float()
    return Phi_batch

def crop_mask(Phi, input_mask, opt, ch):
    _, _, H, W = Phi.shape
    mx = random.randint(0, H-1-256)
    my = random.randint(0, W-1-256-(ch-1)*2)
    Phi_syn = Phi[:,:,mx:mx+256,my:my+256+(ch-1)*2]
    if opt.input_mask is None:
        mask_in = None
    elif opt.input_mask == "Mask":
        mask_in = input_mask[:,:,mx:mx+256,my:my+256]
    elif opt.input_mask == "Phi":
        mask_in = input_mask[:,:,mx:mx+256,my:my+256+(ch-1)*2]
    elif opt.input_mask == "PhiPhi_T":
        mask_in = (input_mask[0][:,:,mx:mx+256,my:my+256+(ch-1)*2], input_mask[1][:,mx:mx+256,my:my+256+(ch-1)*2])
    elif opt.input_mask == "SSR":
        mask_in = input_mask[0][:,:,mx:mx+256,my:my+256], input_mask[1][:,:,mx:mx+256,my:my+256+(ch-1)*2], input_mask[2][:,:,mx:mx+256,my:my+256+(ch-1)*2]
    elif opt.input_mask == "DPU":
        mask_in = input_mask[0][:,:,mx:mx+256,my:my+256], input_mask[1][:,:,mx:mx+256,my:my+256+(ch-1)*2]

    return Phi_syn, mask_in

def gen_meas_torch(data_batch, Phi_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = Phi_batch * shift(data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas, nC)
        if mul_mask:
            HM = torch.mul(H, Phi_batch)
            return HM
        return H
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step, device="cuda", dtype=torch.float32)
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, nC, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step, device="cuda", dtype=torch.float32)
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def shift_back_mask(Phi, step=2):
    B, C, H, W = Phi.shape # B, 28, 256, 310
    input_mask = torch.zeros(B, C, H, W-(C-1)*step, device=Phi.device).float()
    for i in range(C):
        input_mask[:,i,:,:] = Phi[:,i,:,i*step:i*step+W-(C-1)*step]

    return input_mask

def init_mask(mask_path, mask_type, batch_size, ch):
    Phi = generate_Phi(mask_path, batch_size, ch)
    if mask_type == 'Phi':
        input_mask = Phi
    elif mask_type == 'Phi_PhiPhiT':
        Phi_s = torch.sum(Phi**2, 1)
        Phi_s[Phi_s==0] = 1
        input_mask = (Phi, Phi_s)
    elif mask_type == 'Mask':
        input_mask = shift_back_mask(Phi)
    elif mask_type == 'DPU':
        Phi_s = torch.sum(Phi**2, 1)
        Phi_s[Phi_s==0] = 1
        Phi_s = Phi_s.unsqueeze(1)
        input_mask = shift_back_mask(Phi), Phi_s
    elif mask_type == "SSR":
        Phi_s = torch.sum(Phi**2, 1)
        Phi_s[Phi_s==0] = 1
        Phi_s = Phi_s.unsqueeze(1)
        input_mask = shift_back_mask(Phi), Phi, Phi_s
    elif mask_type == 'SGID':
        input_mask = Phi, shift_back_mask(Phi)
    elif mask_type == None:
        input_mask = None
    return Phi, input_mask

def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=False)
    return input_meas

class AverageMeter(object):
    def __init__(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsBase(object):
    def __init__(self, num_classes, names):
        pass

    def pixel_accuracy(self):
        raise NotImplementedError

    def pixel_accuracy_class(self):
        raise NotImplementedError

    def mean_intersection_over_union(self):
        raise NotImplementedError

    def frequency_weighted_intersection_over_union(self):
        raise NotImplementedError

    def _generate_matrix(self):
        raise NotImplementedError

    def get_table(self):
        raise NotImplementedError

    def add_batch(self, gt, pred):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class Metrics_Seg(MetricsBase):
    def __init__(self, num_classes, names):
        super(Metrics_Seg, self).__init__(num_classes, names)
        assert num_classes == len(names)
        self.num_classes = num_classes
        self.names = names
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mean_intersection_over_union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pred_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pred_image[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def get_table(self):
        eps = 1e-4
        total_elem = np.sum(self.confusion_matrix, axis=None)
        tp = np.diag(self.confusion_matrix)
        fp_plus_tp = np.sum(self.confusion_matrix, axis=0)
        fn_plus_tp = np.sum(self.confusion_matrix, axis=1)

        A = (total_elem - (fp_plus_tp + fn_plus_tp - 2 * tp)) / total_elem
        R = tp / (eps + fn_plus_tp)
        P = tp / (eps + fp_plus_tp)
        F1 = 2 * P * R / (eps + P + R)
        IOU = tp / (eps + fp_plus_tp + fn_plus_tp - tp)

        df = pd.DataFrame(data=np.column_stack([IOU, F1, P, R, A]),
                          columns=['IoU', 'F1', 'Prec', 'recall', 'Acc'])

        df = df.round(4)
        df.index = self.names
        total = df.iloc[:, :].mean()
        total_bg = df.iloc[1:, :].mean()
        df.loc['total'] = total
        df.loc['total(-bg)'] = total_bg

        return df

    def add_batch(self, gt_image, pred_image):
        assert gt_image.shape == pred_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pred_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)


class Metrics_Rec:
    def __init__(self):
        super(Metrics_Rec, self).__init__()
        self.psnr = 0
        self.ssim = 0
        self.count = 0

    def get_table(self):
        df = pd.DataFrame(data=np.column_stack([self.psnr/self.count, self.ssim/self.count]),
                          columns=['PSNR', 'SSIM'])

        df = df.round(4)
        return df

    def add_batch(self, gt_image, pred_image):
        assert gt_image.shape == pred_image.shape
        self.psnr += psnr(pred_image, gt_image, data_range=1.0).item()
        self.ssim += ssim(pred_image, gt_image, data_range=1.0).item()
        self.count += 1

    def reset(self):
        self.psnr = 0
        self.ssim = 0
        self.count = 0


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pkl".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
