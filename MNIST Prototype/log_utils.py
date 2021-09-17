import time
from datetime import datetime
import numpy as np
from torchvision.utils import draw_bounding_boxes
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from torchvision.utils import make_grid

from PIL import Image
import wandb
import os
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from torch.utils.tensorboard.summary import hparams

#TODO Make logger class

# class LoggingCB(Callback):

#     def __init__(self, tb_logger):
#         super().__init__()
#         self.start_time = None
#         self.tb_logger = tb_logger

#     def on_epoch_start(self, trainer, pl_module):
#         self.start_time = time.time()
#         if pl_module.args.log_weights_change:
#             self.old_weights = save_weights(pl_module.img_gen)

#     def on_fit_start(self, trainer, pl_module):
#         pass
#         #if not trainer.fast_dev_run:
#             #trainer.logger.watch(trainer.model)

#     def on_epoch_end(self, trainer, pl_module):
#         pl_module.log(f'{pl_module.phase}/Epoch Elapsed Time', time.time() - self.start_time)
#         if pl_module.args.log_weights_change:
#             new_weights = save_weights(pl_module.img_gen)
#             # weights_changed(pl_module.current_epoch, self.old_weights, new_weights)
#             ratios = compare_weights(pl_module.current_epoch, self.old_weights, new_weights)
#             pl_module.log('update-weight-ratio', ratios.mean())

#         # if pl_module.current_epoch == 0:
#         #     # Log just once
#         #     self.tb_logger.add_hparams(dict(pl_module.hparams), dict())


class Logger:
    
    def __init__(self, args):

        current_date = datetime.now()
        formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'

        self.out_dir = args.output_dir

        subfolder = args.subfolder

        if args.suffix:
            test_name = f'{formatted_date}_{args.suffix}'
        else:
            test_name = f'{formatted_date}'

        self.tb_log_dir = os.path.join('logs', subfolder, test_name)
        self.path_out = os.path.join(self.out_dir, subfolder, test_name)

        self.writer = SummaryWriter(self.tb_log_dir)
        wandb.login()

        hparams = dict(
            name = args.data_module,
            ff_dim = args.ff_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            enc_layers = args.n_enc_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dec_layers = args.n_dec_layers, # the number of nn.TransformerDecoderLayer in nn.TransformerDecoder
            nheads = args.n_heads, # the number of heads in the multiheadattention models
            dropout = args.dropout, # the dropout value
            seq_len = args.seq_len,
            seq_bs = args.seq_bs,
            t_lr = args.t_lr,
            g_lr = args.g_lr,
            d_lr = args.d_lr,
            emb_size = args.emb_size,
            img_size = args.data.img_h,
            one_batch = args.overfit_batches,
            ngf = args.ngf,
            ndf = args.ndf
        )

        run_mode = 'disabled' if args.debug else 'online'        

        wandb.init(
            config=hparams,
            # notes=run_notes,
            name=args.subfolder,
            mode=run_mode
        )

        # self.tb = pl.loggers.TensorBoardLogger(
        #     save_dir=args.tb_dir,
        #     name=args.subfolder,
        #     version=formatted_date,
        #     # log_graph=True,
        #     # default_hp_metric=False,
        # )

    def log_images(self, outputs, step, phase, n_samples=6):

        out_image = outputs[-1]['images']
        gt = outputs[-1]['gt'][1:-1]
        boxes = outputs[-1]['gt_boxes'][:-1]
        pred_boxes = outputs[-1]['pred_boxes'][:-1]

        boxes = rearrange(boxes, 'seq b coord -> (seq b) coord')
        pred_boxes = rearrange(pred_boxes, 'seq b coord -> (seq b) coord')
        gt = rearrange(gt, 'seq b c h w -> (seq b) c h w')

        self.log_combined_images(out_image, pred_boxes, phase, step, 'out_images', n_samples)
        self.log_combined_images(gt, boxes, phase, step, 'gt_images', n_samples)

    def combine_img_w_bbox(self, image, box):
        norm_img = ((image - image.min()) / (image.max() - image.min())) * 255

        x, y, w, h = box.tolist()
        xmax = x + w
        ymax = y + h
        box[2] = xmax
        box[3] = ymax

        # Rescale to image size
        box = box * image.shape[-1]
        box = box.unsqueeze(0)

        norm_img = norm_img.type(torch.ByteTensor)
        box = box.type(torch.ByteTensor)

        return draw_bounding_boxes(norm_img, box)

    def custom_histogram_adder(self, named_parameters, current_epoch):
        for name,params in named_parameters:
            self.writer.add_histogram(name, params, current_epoch)
        

    def log_combined_images(self, images, boxes, phase, step, title, n_samples=6):
        image_list = []

        for i, _ in enumerate(range(len(images))):
            img_w_box = self.combine_img_w_bbox(images[i], boxes[i])
            image_list.append(img_w_box.unsqueeze(0))

        images = torch.cat(image_list)

        # Log generated images, show first batch for both
        grid = make_grid(images)
        grid = torch.nn.functional.interpolate(grid.unsqueeze(0), scale_factor=(4,4)).squeeze(0)
        _, pil_img = self.img_to_PIL(grid)

        # pil_img = pil_img.resize((256,256))
        self.writer.add_image(f'{phase}/{title}', grid, step)
        wandb.log({f'{phase}/{title}': wandb.Image(pil_img)})

    def log_metric(self, title, metric, step):
        self.writer.add_scalar(title, metric, step)
        wandb.log({title: metric})

    def img_to_PIL(self, img):

        '''
        img: shape [C, H, W]
        '''

        if len(img.shape) > 3:
            raise Exception("Too many dimensions in image to show")

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


def convert_weights_pl_to_pt(w_path, out_path):
    state_dict = torch.load(w_path)['state_dict']
    new_sd = {}
    for k in state_dict.keys():
        new_k = k[6:] #strip the model string
        new_sd[new_k] = state_dict[k]
    torch.save(new_sd, out_path)