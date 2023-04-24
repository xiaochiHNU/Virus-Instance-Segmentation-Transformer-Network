import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data
from tensorboardX import SummaryWriter
import argparse
import datetime
import re

from utils_yolcat import timer
from modules.yolact import Yolact
from config import get_config
from utils_yolcat.coco import COCODetection, train_collate
from utils_yolcat.common_utils import save_best, save_latest
from eval import evaluate
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--cfg', default='swin_transformer', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
parser.add_argument('--img_size', default=896, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')  # 'weights/best_77.39_swin_tiny_coco_23500.pth'
parser.add_argument('--val_interval', default=10, type=int,
                    help='The validation interval during training, pass -1 to disable.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

loss_min = 5

args = parser.parse_args()
cfg = get_config(args, mode='train')
cfg_name = cfg.__class__.__name__

net = Yolact(cfg)
net.train()

if args.resume:
    net.load_weights(cfg.weight, cfg.cuda)
    start_step = int(cfg.weight.split('.pth')[0].split('_')[-1])
else:
    net.backbone.init_backbone(cfg.weight)
    start_step = 0

dataset = COCODetection(cfg, mode='train')

if 'res' in cfg.__class__.__name__:
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
elif cfg.__class__.__name__ == 'swin_transformer':
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.05)
else:
    raise ValueError('Unrecognized cfg.')

train_sampler = None
main_gpu = False
num_gpu = 0
if cfg.cuda:
    cudnn.benchmark = True
    cudnn.fastest = True
    main_gpu = True
    num_gpu = 1

    net = net.cuda()

# shuffle must be False if sampler is specified
data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=0, shuffle=False,
                               collate_fn=train_collate, pin_memory=True)

epoch_seed = 0
map_tables = []
training = True
timer.reset()
step = start_step
val_step = start_step
writer = SummaryWriter(f'tensorboard_log/{cfg_name}')

if main_gpu:
    print(f'Number of all parameters: {sum([p.numel() for p in net.parameters()])}\n')

try:  # try-except can shut down all processes after Ctrl + C.
    while training:
        if train_sampler:
            epoch_seed += 1
            train_sampler.set_epoch(epoch_seed)

        for images, targets, masks in data_loader:
            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init

            if step in cfg.lr_steps:  # learning rate decay.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * 0.1 ** cfg.lr_steps.index(step)

            if cfg.cuda:
                images = images.cuda().detach()
                targets = [ann.cuda().detach() for ann in targets]
                masks = [mask.cuda().detach() for mask in masks]

            with timer.counter('forloss'):
                loss_c, loss_b, loss_m, loss_s = net(images, targets, masks)

                if cfg.cuda:
                    # use .all_reduce() to get the summed loss from all GPUs
                    all_loss = torch.stack([loss_c, loss_b, loss_m, loss_s], dim=0)
                    #dist.all_reduce(all_loss)

            with timer.counter('backward'):
                loss_total = loss_c + loss_b + loss_m + loss_s
                optimizer.zero_grad()
                loss_total.backward()

            with timer.counter('update'):
                optimizer.step()

            time_this = time.time()
            if step > start_step:
                batch_time = time_this - time_last
                timer.add_batch_time(batch_time)
            time_last = time_this

            if step % 10 == 0 and step != start_step:
                if (not cfg.cuda) or main_gpu:
                    cur_lr = optimizer.param_groups[0]['lr']
                    time_name = ['batch', 'data', 'forloss', 'backward', 'update']
                    t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                    seconds = (cfg.lr_steps[-1] - step) * t_t
                    eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

                    # Get the mean loss across all GPUS for printing, seems need to call .item(), not sure
                    l_c = all_loss[0].item() / num_gpu if main_gpu else loss_c.item()
                    l_b = all_loss[1].item() / num_gpu if main_gpu else loss_b.item()
                    l_m = all_loss[2].item() / num_gpu if main_gpu else loss_m.item()
                    l_s = all_loss[3].item() / num_gpu if main_gpu else loss_s.item()

                    writer.add_scalar('loss/class', l_c, global_step=step)
                    writer.add_scalar('loss/box', l_b, global_step=step)
                    writer.add_scalar('loss/mask', l_m, global_step=step)
                    writer.add_scalar('loss/semantic', l_s, global_step=step)
                    writer.add_scalar('loss/total', loss_total, global_step=step)

                    print(f'step: {step} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | '
                          f'l_mask: {l_m:.3f} | l_semantic: {l_s:.3f} | loss_total: {loss_total:.3f}  ETA: {eta}')

                    if loss_total <= loss_min:
                        loss_min = loss_total
                        torch.save(net.state_dict(), f'weights/loss_{loss_total:.2f}_{cfg_name}_{step}.pth')

            if ((not cfg.cuda) or main_gpu) and step == val_step + 1:
                timer.start()  # the first iteration after validation should not be included

            step += 1
            if step >= cfg.lr_steps[-1]:
                training = False

                if (not cfg.cuda) or main_gpu:
                    save_latest(net.module if cfg.cuda else net, cfg_name, step)

                    print('\nValidation results during training:\n')
                    for table in map_tables:
                        print(table, '\n')

                    print(f'Training completed.')

                break

except KeyboardInterrupt:
    if (not cfg.cuda) or main_gpu:
        save_latest(net.module if cfg.cuda else net, cfg_name, step)

        print('\nValidation results during training:\n')
        for table in map_tables:
            print(table, '\n')
