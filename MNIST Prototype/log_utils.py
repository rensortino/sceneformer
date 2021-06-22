import time
import random
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
from PIL import Image
import wandb


class Logging(Callback):

    def __init__(self, tb_logger):
        super().__init__()
        self.start_time = None
        self.tb_logger = tb_logger

    def on_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_fit_start(self, trainer, pl_module):
        pass
        #if not trainer.fast_dev_run:
            #trainer.logger.watch(trainer.model)

    def on_epoch_end(self, trainer, pl_module):
        #trainer.logger.experiment.log({"Reconstructed Image": [wandb.Image(np_image, caption="First of each batch")]})
        # self.log('pl_logger_train_loss', pl_module.step_loss)
        # trainer.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=pl_module.current_epoch)
        self.tb_logger.add_scalar('Epoch Elapsed Time', time.time() - self.start_time)

def img_to_PIL(img):

    if len(img.shape) > 4:
        raise Exception("Too many dimensions in image to show")
    if len(img.shape) == 4: 
        # Get a random sample from the batch
        idx = random.randint(0,img.shape[0] - 1) 
        img = img[idx]

    if type(img) == torch.Tensor:
        img = img.cpu().detach()
        
    img_array = np.transpose(img.numpy(), (1,2,0))
    norm_img = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
    PIL_image = Image.fromarray(norm_img.astype('uint8'), 'RGB')
    return norm_img, PIL_image

def save_weights(model):
    state_dict = {}
    for key in model.state_dict():
        state_dict[key] = model.state_dict()[key].clone()
    return state_dict

def get_data_stats(data):
    stats = f'Mean:\t{data.mean()}\nStd:\t{data.std()}\nMax:\t{data.max()}\nMin:\t{data.min()}'
    return stats

def compare_weights(epoch, old_state_dict, new_state_dict):
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


def log_metric(pl_module, title, metric, step):
    pl_module.log(title, metric, prog_bar=True)
    pl_module.tb_writer.add_scalar(title, metric, step)
    wandb.log({title: metric})


def convert_weights_pl_to_pt(w_path, out_path):
    state_dict = torch.load(w_path)['state_dict']
    new_sd = {}
    for k in state_dict.keys():
        new_k = k[6:] #strip the model string
        new_sd[new_k] = state_dict[k]
    torch.save(new_sd, out_path)