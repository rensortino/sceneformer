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

    def __init__(self):
        super().__init__()
        self.start_time = None

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
        pl_module.logger.experiment.add_scalar('Epoch Elapsed Time', time.time() - self.start_time)

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
    return norm_img
    # plt.imshow(np.transpose(img.numpy(), (1,2,0)), cmap="gray", interpolation='nearest')
    #image = plt.imshow(img_array, interpolation='nearest')
    #plt.show()
    #return PIL_image, img_array


def log_prediction(gt, pred, logger, nrow=16, title : str = "Logged Image"):
    # h, w = hp['img_h'], hp['img_w']

    # gt = gt.view(hp['seq_len'] * hp['t_bs'], 1, h * int(math.sqrt(hp['seq_len'])), w * int(math.sqrt(hp['seq_len']))) # Reshape to [B,C,H,W]
    # pred = pred.view(hp['seq_len'] * hp['t_bs'], 1, h * int(math.sqrt(hp['seq_len'])), w * int(math.sqrt(hp['seq_len']))) # Reshape to [B,C,H,W]

    resize = T.Resize((64,64))
    gt = resize(gt)
    pred = resize(pred)
    gt_grid = make_grid(gt.cpu(), nrow=nrow, padding=16)
    p_grid = make_grid(pred.cpu(), nrow=nrow, padding=16)

    gt_img = img_to_PIL(gt_grid)
    p_img = img_to_PIL(p_grid)

    wandb.log({title :[
        wandb.Image(p_grid, caption="Prediction"),
        wandb.Image(gt_grid, caption="Ground Truth"),
    ]})

    logger.experiment.add_image("Prediction", p_img, dataformats="HWC")
    logger.experiment.add_image("Ground Truth", gt_img, dataformats="HWC")




def convert_weights_pl_to_pt(w_path, out_path):
    state_dict = torch.load('ckpt/resnet18_mnist_unformatted.pt')['state_dict']
    new_sd = {}
    for k in state_dict.keys():
        new_k = k[6:] #strip the model string
        new_sd[new_k] = state_dict[k]
    torch.save(new_sd, 'resnet18_mnist.pt')