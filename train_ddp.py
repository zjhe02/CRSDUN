import os
from opt_ddp import opt
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import torch
from torch import nn
from utils import *
from dataset import *
from models import *
import math
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import warnings
warnings.filterwarnings("ignore")

def main_worker(rank, world_size):
    # -------------------- 初始化 DDP --------------------
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    set_seed(42 + rank)

    # -------------------- 环境与路径 --------------------
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    if rank == 0:
        print(opt)
    result_path = opt.outf + opt.name + '/result/'
    model_path = opt.outf + opt.name + '/model/'
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # -------------------- dataset --------------------
    train_loader, valid_loader = prep_loaders_ddp(
        root_dir=opt.data_root,
        batch_size=opt.batch_size,
        workers=4,
        rank=rank,
        world_size=world_size
    )

    # -------------------- mask --------------------
    ch = 28
    Phi_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size, ch)
    Phi_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 1, ch)

    # -------------------- model --------------------
    n_classes = train_loader.dataset.num_classes
    model = model_generator(opt.method, ch, n_classes, opt.pretrained_model_path).cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # -------------------- optimizer / loss --------------------
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, 1e-6)
    scaler = torch.GradScaler(device="cuda")

    loss_fn_seg = nn.CrossEntropyLoss().cuda(rank)
    loss_fn_rec = nn.MSELoss().cuda(rank)

    # -------------------- metrics / logger --------------------
    metrics_rec = Metrics_Rec()
    metrics_seg = Metrics_Seg(n_classes, train_loader.dataset.class_names)
    logger = None
    if rank == 0:
        logger = gen_log(model_path)

    # -------------------- training --------------------
    max_iou = 0
    max_psnr = 0
    lam_rec = 1
    lam_seg = 1e-4

    for epoch in range(opt.max_epoch):
        model.train()
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        losses_rec = AverageMeter()
        losses_seg = AverageMeter()

        for i, (sample) in enumerate(train_loader):
            x = sample['image'].float().cuda(rank, non_blocking=True)
            y = sample['label'].long().cuda(rank, non_blocking=True)

            Phi_syn, mask_in = crop_mask(Phi_batch_train, input_mask_train, opt, ch)
            mea = init_meas(x, Phi_syn, opt.input_setting)

            with torch.autocast(device_type="cuda", enabled=True):
                x_pred_list, y_pred_list = model(mea, mask_in)

                loss_rec = 0
                loss_seg = 0
                stage_list = list(range(len(x_pred_list)))
                stage_list.reverse()
                for i, stage in enumerate(stage_list):
                    loss_rec = loss_rec + loss_fn_rec(x_pred_list[stage], x) * math.pow(0.7, i)
                    loss_seg = loss_seg + loss_fn_seg(y_pred_list[stage], y) * math.pow(0.7, i)

                loss = lam_rec * loss_rec + lam_seg * loss_seg

            losses_rec.update(loss_rec.item(), x.size(0))
            losses_seg.update(loss_seg.item(), y.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        torch.cuda.empty_cache()

        # -------------------- validation --------------------
        if rank == 0:
            model.eval()
            metrics_rec.reset()
            metrics_seg.reset()
            name_list = []
            seg_map_list = []

            for i, (sample) in enumerate(valid_loader):
                x = sample['image'].float().cuda(rank)
                y = sample['label'].numpy()
                mea = init_meas(x, Phi_batch_test, opt.input_setting)
                with torch.no_grad():
                    x_pred_list, y_pred_list = model(mea, input_mask_test)
                    x_pred = x_pred_list[-1]
                    y_pred = y_pred_list[-1]
                    y_pred = torch.argmax(y_pred, dim=1)
                    seg_map_list.append(decode_segmap(y_pred.cpu()).astype(np.uint8))
                    name_list.append(valid_loader.dataset.names[i])
                metrics_rec.add_batch(x.cpu(), x_pred.detach().cpu())
                metrics_seg.add_batch(y, y_pred.detach().cpu().numpy())

            metrics_rec_table = metrics_rec.get_table()
            metrics_seg_table = metrics_seg.get_table()
            logger.info(f"-------------------Epoch: {epoch+1}------------------------")
            logger.info(f'\nValidation stats:\n{metrics_rec_table}')
            logger.info(f'\nValidation stats:\n{metrics_seg_table}')

            test_iou = metrics_seg_table.at["total(-bg)", "IoU"]
            test_psnr = metrics_rec_table.at[0, "PSNR"]
            if test_iou > max_iou or test_psnr > max_psnr:
                checkpoint(model, epoch+1, model_path, logger)
            
    if rank == 0:
        print("Done")
    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    print("------------------start training-------------------------")
    main()
