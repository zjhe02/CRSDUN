import os
from opt import opt
print(opt)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import math
import gc
import torch
from torch import nn
from utils import *
from dataset import *
from models import *
scaler = torch.GradScaler(device="cuda")

import warnings
warnings.filterwarnings("ignore")

set_seed(3407)

# dataset
train_loader, valid_loader = prep_loaders(
    root_dir=opt.data_root, 
    batch_size=opt.batch_size, 
    workers=4
)

# mask
ch = 28
Phi_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size, ch)
Phi_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 1, ch)

# model
n_classes = train_loader.dataset.num_classes

model = model_generator(opt.method, ch, n_classes, opt.pretrained_model_path).cuda()

# optimizer
optimizer = torch.optim.Adam(params= model.parameters(),lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, 1e-6)

# loss
loss_fn_seg = nn.CrossEntropyLoss().cuda()
loss_fn_rec = nn.MSELoss().cuda()

# metrics
metrics_rec = Metrics_Rec()
metrics_seg = Metrics_Seg(train_loader.dataset.num_classes, train_loader.dataset.class_names)

# path
result_path = opt.outf + opt.name + '/result/'
model_path = opt.outf + opt.name + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

logger = gen_log(model_path)

def main():
    max_iou = 0
    max_psnr = 0
    lam_rec = 1
    lam_seg = 1e-4
    for epoch in range(opt.max_epoch):
        model.train()

        # Progress reporting
        losses_rec = AverageMeter()
        losses_seg = AverageMeter()

        for i, (sample) in enumerate(train_loader):

            # Load a batch and send it to GPU
            x = sample['image'].float().cuda()
            y = sample['label'].long().cuda()

            Phi_syn, mask_in = crop_mask(Phi_batch_train, input_mask_train, opt, ch)
            mea = init_meas(x, Phi_syn, opt.input_setting)

            with torch.autocast(device_type="cuda", enabled=True):
                # Forward pass: compute predicted y by passing x to the model.
                x_pred_list, y_pred_list = model(mea, mask_in)

                # Compute and print loss.
                loss_rec = 0
                loss_seg = 0
                stage_list = list(range(len(x_pred_list)))
                stage_list.reverse()
                for i, stage in enumerate(stage_list):
                    loss_rec = loss_rec + loss_fn_rec(x_pred_list[stage], x) * math.pow(0.7, i)
                    loss_seg = loss_seg + loss_fn_seg(y_pred_list[stage], y) * math.pow(0.7, i)

                loss = lam_rec * loss_rec + lam_seg * loss_seg

            # Record loss
            losses_rec.update(loss_rec.data.item(), x.size(0))
            losses_seg.update(loss_seg.data.item(), y.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        logger.info(f"Rec loss: {losses_rec.avg}")
        logger.info(f"Seg loss: {losses_seg.avg}")
        logger.info(f"λ_rec: {lam_rec}")
        logger.info(f"λ_seg: {lam_seg}")

        scheduler.step()
        torch.cuda.empty_cache(); del x, y; gc.collect()

        # Validation after each epoch
        model.eval()
        metrics_rec.reset()
        metrics_seg.reset()
        name_list = []
        seg_map_list = []
        for i, (sample) in enumerate(valid_loader):
            x, y = sample['image'].float().cuda(), sample['label'].numpy()
            mea = init_meas(x, Phi_batch_test, opt.input_setting)
            with torch.no_grad():
                x_pred_list, y_pred_list = model(mea, input_mask_test)
                x_pred = x_pred_list[-1]
                y_pred = y_pred_list[-1]
                y_pred = torch.argmax(y_pred, dim=1) # get the most likely prediction
                seg_map_list.append(decode_segmap(y_pred.cpu()).astype(np.uint8))
                name_list.append(valid_loader.dataset.names[i])

            metrics_rec.add_batch(x.cpu(), x_pred.detach().cpu())
            metrics_seg.add_batch(y, y_pred.detach().cpu().numpy())
        metrics_rec_table = metrics_rec.get_table()
        metrics_seg_table = metrics_seg.get_table()
        logger.info(f"-------------------Epoch: {epoch+1}------------------------")
        logger.info(f'\nValidation stats:\n{metrics_rec_table}')
        logger.info(f'\nValidation stats:\n{metrics_seg_table}')
        # Save model
        test_iou = metrics_seg_table.at["total(-bg)", "IoU"]
        test_psnr = metrics_rec_table.at[0, "PSNR"]
        if test_iou > max_iou or test_psnr > max_psnr:
            checkpoint(model, epoch+1, model_path, logger)

    print("Done")

if __name__ == "__main__":
    "------------------start training-------------------------"
    main()

