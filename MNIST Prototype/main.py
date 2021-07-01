from datetime import datetime
from log_utils import SummaryWriter
from data_processing import build_vocab
from feature_extractor import ResNet18
import math
import json
import torch
from ytid import YTID
from layers import *
from attrdict import AttrDict
from log_utils import Logging
from layers import ImageGenerator, Discriminator
import os

from data_modules import MNISTDataModule, CIFAR10DataModule

import pytorch_lightning as pl

import wandb

tokens = {
    'src': {
        "SOS": 14,
        "EOS": 15,
        "PAD": 16
    },
    'tgt': {
        "SOS": 0.0,
        "EOS": 1.0
    }
}

def main(args):

    # Setup

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data.batch_size % args.model.seq_len == 0, "Batch size not divisible by sequence length"

    os.makedirs(args.trainer.output_dir, exist_ok=True)
    os.makedirs(args.trainer.weight_dir, exist_ok=True)
    
    args['model']['seq_bs'] = int(args.data.batch_size / args.model.seq_len)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image_size = (args.data.img_w * int(math.sqrt(args.model.seq_len)), args.data.img_h * int(math.sqrt(args.model.seq_len)))
    image_size = (args.data.img_w, args.data.img_h)

    # Data Loading

    if args.name == "MNIST":
        data_module = MNISTDataModule(args.data, image_size)
    elif args.name == "CIFAR10":
        data_module = CIFAR10DataModule(args.data)

    # Logging
    
    current_date = datetime.now()
    formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'

    wandb.login()

    hparams = dict(
        name = args.name,
        nhid = args.model.ff_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.model.n_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = args.model.n_heads, # the number of heads in the multiheadattention models
        dropout = args.model.dropout, # the dropout value
        seq_len = args.model.seq_len,
        seq_bs = args.model.seq_bs,
        t_lr = args.optimizer.t_lr,
        g_lr = args.optimizer.g_lr,
        d_lr = args.optimizer.d_lr,
        emb_size = args.model.emb_size,
        img_size = args.data.img_h,
        one_batch = args.trainer.debug_one_batch,
        ngf = args.model.ngf,
        ndf = args.model.ndf
    )

    if args.test:
        run_mode = "disabled"
        run_name = "Test"
        run_notes = "Testing..."
        
    else:
        run_mode = "online"
        run_name = input("Name your run: ")
        run_notes = input("Describe your run: ")

    wandb.init(
        config=hparams,
        notes=run_notes,
        name=run_name,
        mode=run_mode
    )

    # TODO Delete?
    tb_writer = SummaryWriter(os.path.join(args.trainer.tensorboard_dir, run_name, formatted_date))

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="pl_logs",
        name=run_name,
        version=formatted_date
        # default_hp_metric=False,
    )

    # Modules Instances

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
    img_gen = ImageGenerator(image_size, emb_size=args.model.emb_size, ngf=args.model.ngf, channels=args.data.n_channels).to(args.device)

    model = YTID(hparams, transformer, img_gen, disc, feature_extractor, tb_writer, args).to(args.device)

    # Create Vocabulary of embeddings

    if os.path.exists('vocab.pth'):
        vocab = torch.load('vocab.pth')
    else:
        data_module.setup()
        vocab = build_vocab(feature_extractor, data_module.train_dataloader(), tokens)

    # Define trainer module

    trainer = pl.Trainer(
        gpus=args.n_gpu,
        logger=tb_logger,
        # default_hp_metric=False,
        # fast_dev_run=True,
        # overfit_batches=0.001, # 1% of training set used as batch to make it overfit
        # limit_train_batches=0.25, # 10% of training data
        # limit_val_batches=0.0011641443538998836, # 10% of validation data
        num_sanity_val_steps=0,
        flush_logs_every_n_steps=20,
        progress_bar_refresh_rate=20,
        max_epochs=args.trainer.epochs,
        #profiler=True, # This is very time consuming
        callbacks=[
            Logging(tb_logger),
        ]
    )

    # Load weights

    if args.trainer.resume:
        checkpoint = torch.load(f'{args.trainer.weight_dir}/checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        trainer.current_epoch = checkpoint['epoch']

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config/mnist.json") as conf:
        args = AttrDict(json.load(conf))

    main(args)