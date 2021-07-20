import time
import random
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
from PIL import Image
import wandb
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

class Logging(Callback):

    def __init__(self, tb_logger):
        super().__init__()
        self.start_time = None
        self.tb_logger = tb_logger

    def on_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        if pl_module.args.log_weights_change:
            self.old_weights = save_weights(pl_module.img_gen)

    def on_fit_start(self, trainer, pl_module):
        pass
        #if not trainer.fast_dev_run:
            #trainer.logger.watch(trainer.model)

    def on_epoch_end(self, trainer, pl_module):
        pl_module.log(f'{pl_module.phase}/Epoch Elapsed Time', time.time() - self.start_time)
        if pl_module.args.log_weights_change:
            new_weights = save_weights(pl_module.img_gen)
            # weights_changed(pl_module.current_epoch, self.old_weights, new_weights)
            ratios = compare_weights(pl_module.current_epoch, self.old_weights, new_weights)
            pl_module.log('update-weight-ratio', ratios.mean())

        # if pl_module.current_epoch == 0:
        #     # Log just once
        #     self.tb_logger.add_hparams(dict(pl_module.hparams), dict())

def img_to_PIL(img):

    '''
    img: shape [C, H, W]
    '''

    if len(img.shape) > 4:
        raise Exception("Too many dimensions in image to show")
    if len(img.shape) == 4: 
        # Get a random sample from the batch
        idx = random.randint(0,img.shape[0] - 1) 
        img = img[idx]

    if type(img) == torch.Tensor:
        img = img.cpu().detach()

    n_ch = img.shape[0]
    if n_ch == 1:
        img_array = img.squeeze(0).numpy()
        arr_to_pil = lambda x: Image.fromarray(x.astype('uint8'), 'L')

    elif n_ch == 3:
        img_array = np.transpose(img.numpy(), (1,2,0))
        arr_to_pil = lambda x: Image.fromarray(x.astype('uint8'), 'RGB')

    else:
        raise Exception(f'Unsupported number of channels ({n_ch})')

        
    norm_img = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    PIL_image = arr_to_pil(norm_img)
    return norm_img, PIL_image

def save_weights(model):
    state_dict = {}
    for key in model.state_dict():
        state_dict[key] = model.state_dict()[key].clone()
    return state_dict

def get_data_stats(data):
    stats = f'Mean:\t{data.mean()}\nStd:\t{data.std()}\nMax:\t{data.max()}\nMin:\t{data.min()}'
    return stats

def check_data_ordering(vocab, src, tgt, seq_len=10):
    for i in range(seq_len):
        print((vocab[src[i].item()] == tgt[i]).all())

def weights_changed(epoch, old_state_dict, new_state_dict):
    changed = {}
    for key in old_state_dict:
        module = key.split('.')[0]
        if (old_state_dict[key].cpu() == new_state_dict[key].cpu()).all():
            if module not in changed:
                changed[module] = np.array([False])
            else:
                changed[module] = np.append(changed[module], False)
        else:
            if module not in changed:
                changed[module] = np.array([True])
            else:
                changed[module] = np.append(changed[module], True)
    with open('weight_change.txt', 'a') as out:
        out.write(f'Epoch: {epoch}\n')
        for module in changed:
            if(changed[module].any()):
                out.write(f'Module {module} changed\n')

def compare_weights(epoch, old_state_dict, new_state_dict):

    ratios = []
    for k in old_state_dict:
        module = k.split('.')[0]
        update = (new_state_dict[k].float() - old_state_dict[k].float()).norm()
        param_scale = old_state_dict[k].float().norm()
        ratio = update / param_scale
        ratios.append(ratio.unsqueeze(0))
    return torch.cat(ratios)


def log_prediction(gt, tgt_box, pred, box, n_channels, seq_bs, logger, title : str = "Logged Image"):

    pred = pred.view(pred.shape[0] * pred.shape[1], n_channels, pred.shape[2], pred.shape[3])
    gt = gt.view(gt.shape[0] * gt.shape[1], n_channels, gt.shape[3], gt.shape[4])

    resize = T.Resize((64,64))
    gt = resize(gt)
    pred = resize(pred)
    gt_grid = make_grid(gt.cpu(), nrow=seq_bs, padding=16)
    p_grid = make_grid(pred.cpu(), nrow=seq_bs, padding=16)

    gt_img, _ = img_to_PIL(gt_grid)
    p_img, _ = img_to_PIL(p_grid)

    wandb.log({title :[
        wandb.Image(p_grid, caption="Prediction"),
        wandb.Image(gt_grid, caption="Ground Truth"),
    ]})

    logger.experiment.add_image("Prediction", p_img, dataformats="HWC")
    logger.experiment.add_image("Ground Truth", gt_img, dataformats="HWC")


def log_metric(pl_module, title, metric, step, prog_bar=False):
    # for phase in metrics:
    #     for metric in phase:
    pl_module.log(title, metric, prog_bar=prog_bar)
    wandb.log({title: metric})


def convert_weights_pl_to_pt(w_path, out_path):
    state_dict = torch.load(w_path)['state_dict']
    new_sd = {}
    for k in state_dict.keys():
        new_k = k[6:] #strip the model string
        new_sd[new_k] = state_dict[k]
    torch.save(new_sd, out_path)