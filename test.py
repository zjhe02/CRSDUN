import os
from opt import opt
print(opt)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils import *
from dataset import *
from models import *
scaler = torch.GradScaler(device="cuda")
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

set_seed(42)

# dataset
_, valid_loader = prep_loaders(
    root_dir=opt.data_root,
    batch_size=opt.batch_size, 
    workers=4
)

# mask
ch = 28

Phi_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 1, ch)

# model
n_classes = valid_loader.dataset.num_classes

model = model_generator(opt.method, ch, n_classes, opt.pretrained_model_path).cuda()

# metrics
metrics_rec = Metrics_Rec()
metrics_seg = Metrics_Seg(valid_loader.dataset.num_classes, valid_loader.dataset.class_names)

# path
result_path = opt.outf + opt.name + '/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

logger = gen_log(result_path)

def main():
    # Validation after each epoch
    model.eval()
    metrics_seg.reset()
    for i, (sample) in tqdm(enumerate(valid_loader)):
        x = sample['image'].float().cuda()
        label = sample['label'].numpy()
        mea = init_meas(x, Phi_batch_test, opt.input_setting)

        with torch.no_grad():
            rec_pred, seg_pred = model(mea, input_mask_test)
            rec_pred = rec_pred[-1]
            seg_pred = seg_pred[-1]
            seg_pred = torch.argmax(seg_pred, dim=1) # get the most likely prediction
            
            # add metrics batch
            rec_pred = rec_pred.cpu()
            metrics_rec.add_batch(x.cpu(), rec_pred)
            metrics_seg.add_batch(label, seg_pred.cpu().numpy())
            
            # save_hsi
            np.save(os.path.join(result_path, f"scene{i+1:02d}_hsi.npy"), rec_pred.squeeze(0).permute(1,2,0).numpy())

            # save_seg_map
            seg_map_rgb = decode_segmap(seg_pred.cpu()).astype(np.uint8)
            plt.imsave(os.path.join(result_path, f"scene{i+1:02d}.png"), seg_map_rgb)

    metrics_rec_table = metrics_rec.get_table()
    metrics_seg_table = metrics_seg.get_table()
    logger.info(f'\nTest stats:\n{metrics_rec_table}')
    logger.info(f'\nTest stats:\n{metrics_seg_table}')

    print("Done")

if __name__ == "__main__":
    "------------------start testing-------------------------"
    main()

