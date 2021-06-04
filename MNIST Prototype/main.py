import math
import json
import torch
from ytid import YTID
from layers import ImageGenerator, PositionalEncoding, ImageTransformer
import numpy as np
from attrdict import AttrDict
from log_utils import Logging


from feature_extractor import ResNet18
from data_modules import MNISTDataModule, CIFAR10DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import wandb

from cgan import weights_init, DCGANDiscriminator

# TODO Input mask should zero attention where there is pad
# Target mask should do this and also the triu mask

TOKENS = {
        "SOS": 0.0,
        "EOS": 1.0,
        "PAD": 0.5
}

def main(args):

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data_loader.batch_size % args.model.seq_len == 0, "Batch size not divisible by sequence length"
    
    args.model.seq_bs = args.data_loader.batch_size / args.model.seq_len
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.name == "MNIST":
        data_module = MNISTDataModule(args.data_loader)
    elif args.name == "CIFAR10":
        data_module = CIFAR10DataModule(args.data_loader)

    if args.model.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.model.loss == "l1":
        criterion = torch.nn.L1Loss()
    elif args.model.loss == "kldiv":
        criterion = torch.nn.KLDivLoss()
        # criterion = torch.nn.KLDivLoss(log_target=True)
    elif args.model.loss == "xe":
        criterion = torch.nn.CrossEntropyLoss()

    tb_logger = TensorBoardLogger("tb_logs", name="YTID")
    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        # overfit_batches=0.01, # 1% of training set used as batch to make it overfit # TODO Use this
        # limit_train_batches=0.1, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        num_sanity_val_steps=0,
        flush_logs_every_n_steps=20,
        progress_bar_refresh_rate=20,
        #max_epochs=epochs,
        #profiler=True,
        logger=tb_logger,
        callbacks=[
            Logging(),
        ]
    )

    wandb.login()

    hparams = dict(
        nhid = args.model.ff_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.model.n_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = args.model.n_heads, # the number of heads in the multiheadattention models
        dropout = args.model.dropout, # the dropout value
        g_lr = args.model.g_lr,
        d_lr = args.model.d_lr,
        t_lr = args.model.t_lr,
        emb_size = args.model.emb_size,
        img_h = args.data_loader.img_h,
        img_w = args.data_loader.img_w,
    )

    wandb.init(
        config=hparams,
        mode="disabled"
    )

    wandb.run.name = "MNIST Without adversarial"

    config = wandb.config

    image_size = (args.data_loader.img_w * int(math.sqrt(args.model.seq_len)), args.data_loader.img_h * int(math.sqrt(args.model.seq_len)))
    
    feature_extractor = ResNet18(args.model.fe_weights_path).to(args.device)
    pos_enc = PositionalEncoding(args.model.emb_size, args.model.dropout).to(args.device)
    embedding = torch.nn.Embedding(args.model.num_classes + len(TOKENS), args.model.emb_size).to(args.device)

    img_gen = ImageGenerator(image_size, emb_size=args.model.emb_size, ngf=16, channels=args.data_loader.n_channels).to(args.device)
    #TODO Pass as kwargs
    transformer = ImageTransformer(
            args.model.emb_size,
            args.model.n_heads,
            args.model.n_layers,
            args.model.n_layers,
            args.model.ff_dim,
            args.model.dropout,
            args.data_loader,
            args.device,
            embedding,
            img_gen,
            pos_enc
        ).to(args.device)
    disc = DCGANDiscriminator(image_channels=args.data_loader.n_channels).to(args.device)
    model = YTID(feature_extractor, transformer, disc, args, criterion).to(args.device)

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config/mnist.json") as conf:
        args = AttrDict(json.load(conf))

    main(args)