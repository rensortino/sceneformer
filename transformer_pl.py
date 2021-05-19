import math
import json
import torch
from ytid import YTID, ImageGenerator, PositionalEncoding
import numpy as np
from attrdict import AttrDict
from log_utils import Logging


from feature_extractor import ResNet18
from data_modules import MNISTDataModule, CIFAR10DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import wandb

from cgan import weights_init, DCGANDiscriminator

# TODO Input mask should zero attention where there is pad
# Target mask should do this and also the triu mask

TOKENS = {
        "SOS": 11,
        "EOS": 12,
        "PAD": 13
}


def main(args):

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data_loader.batch_size == args.model.seq_len * args.model.seq_bs, "Batch size is not seq_len * transformer_batch"

    data_module = CIFAR10DataModule(args)
    wandb_logger = WandbLogger(project="MNIST Transformer")
    tb_logger = TensorBoardLogger("tb_logs", name="YTID")
    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        # overfit_batches=0.01, # 1% of training set used as batch to make it overfit
        # limit_train_batches=0.1, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        num_sanity_val_steps=2,
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
        nlayers = args.model.n_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncowandder
        nheads = args.model.n_heads, # the number of heads in the multiheadattention models
        dropout = args.model.dropout, # the dropout value
        lr = args.model.lr,
        emb_size = args.model.emb_size,
        img_h = args.data_loader.img_h,
        img_w = args.data_loader.img_w,
    )

    wandb.init(
        config=hparams,
        mode="disabled"
    )

    wandb.mode = "disabled"

    #wandb.run.name = "LR 100"

    config = wandb.config

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = (args.data_loader.img_w * int(math.sqrt(args.model.seq_len)), args.data_loader.img_h * int(math.sqrt(args.model.seq_len)))
    
    feature_extractor = ResNet18(args.model.fe_weights_path).to(args.device)
    pos_enc = PositionalEncoding(args.model.emb_size, args.model.dropout).to(args.device)
    embedding = torch.nn.Embedding(args.model.num_classes + len(TOKENS), args.model.emb_size).to(args.device)

    if args.model.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif args.model.loss == "l1":
        criterion = torch.nn.L1Loss()

    img_gen = ImageGenerator(image_size, emb_size=args.model.emb_size, ngf=16, channels=args.data_loader.n_channels).to(args.device)
    disc = DCGANDiscriminator(image_channels=args.data_loader.n_channels).to(args.device)
    model = YTID(feature_extractor, embedding, pos_enc, img_gen, disc, args, criterion).to(args.device)

    #wandb_logger.watch(model, log_freq=1, log="all")

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config_cifar.json") as conf:
        args = AttrDict(json.load(conf))

    main(args)