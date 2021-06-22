from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from data_processing import build_vocab
from feature_extractor import ResNet18
import math
import json
import torch
from ytid import YTID
from layers import ImageGenerator, PositionalEncoding, ImageTransformer
import numpy as np
from attrdict import AttrDict
from log_utils import Logging
from layers import ImageGenerator, Discriminator
import os




from data_modules import MNISTDataModule, CIFAR10DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import wandb

from cgan import weights_init, DCGANDiscriminator

# TODO Input mask should zero attention where there is pad
# Target mask should do this and also the triu mask

tokens = {
    'src': {
        "SOS": 14,
        "EOS": 15,
        "PAD": 16
    },
    'tgt': {
        "SOS": 0.0,
        "EOS": 1.0,
        "PAD": 0.5 # TODO Modify 
    }
}

def main(args):

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data.batch_size % args.model.seq_len == 0, "Batch size not divisible by sequence length"
    
    args.model.seq_bs = args.data.batch_size / args.model.seq_len
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image_size = (args.data.img_w * int(math.sqrt(args.model.seq_len)), args.data.img_h * int(math.sqrt(args.model.seq_len)))
    image_size = (args.data.img_w, args.data.img_h)

    if args.name == "MNIST":
        data_module = MNISTDataModule(args.data, image_size)
    elif args.name == "CIFAR10":
        data_module = CIFAR10DataModule(args.data)

    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss == "mae":
        criterion = torch.nn.L1Loss()
    elif args.loss == "kldiv":
        criterion = torch.nn.KLDivLoss()
        # criterion = torch.nn.KLDivLoss(log_target=True)
    elif args.loss == "xe":
        criterion = torch.nn.CrossEntropyLoss()

    # Config WandB Logging
    wandb.login()

    hparams = dict(
        name = args.name,
        nhid = args.model.ff_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.model.n_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = args.model.n_heads, # the number of heads in the multiheadattention models
        dropout = args.model.dropout, # the dropout value
        criterion = args.loss,
        seq_len = args.model.seq_len,
        seq_bs = args.model.seq_bs,
        t_lr = args.optimizer.t_lr,
        emb_size = args.model.emb_size,
        img_size = args.data.img_h,
        one_batch = args.trainer.debug_one_batch,
        ngf = args.model.ngf,
        ndf = args.model.ndf
    )

    wandb.init(
        config=hparams,
        mode="disabled"
    )

    wandb.run.name = "Vocab + KLDiv and XE on trf_out + fc"

    current_date = datetime.now()
    tb_writer = SummaryWriter(os.path.join('tb_logs', wandb.run.name, f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'))


    #TODO Pass as kwargs
    feature_extractor = ResNet18(args.model.fe_weights_path, 16)
    transformer = ImageTransformer(
            args.model.emb_size,
            args.model.n_heads,
            args.model.n_layers,
            args.model.n_layers,
            args.model.ff_dim,
            args.model.dropout,
            args.data,
            image_size,
            args.device,
            feature_extractor
        ).to(args.device)
    disc = Discriminator(args.model.emb_size, ndf=16, channels=args.data.n_channels)
    img_gen = ImageGenerator(image_size, emb_size=args.model.emb_size, ngf=16, channels=args.data.n_channels).to(args.device)
    if os.path.exists('vocab.pth'):
        vocab = torch.load('vocab.pth')
    else:
        data_module.setup()
        vocab = build_vocab(feature_extractor, data_module.train_dataloader(), tokens)
    model = YTID(transformer, img_gen, disc, feature_extractor, args, criterion).to(args.device)


    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        # overfit_batches=0.000625, # 1% of training set used as batch to make it overfit
        # limit_train_batches=0.25, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        num_sanity_val_steps=0,
        flush_logs_every_n_steps=20,
        progress_bar_refresh_rate=20,
        max_epochs=500,
        #profiler=True,
        callbacks=[
            Logging(tb_writer),
        ]
    )
    
    #TODO Parametrize
    resume = True
    if resume:
        checkpoint = torch.load('weights/checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        trainer.current_epoch = checkpoint['epoch']

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config/mnist.json") as conf:
        args = AttrDict(json.load(conf))

    main(args)