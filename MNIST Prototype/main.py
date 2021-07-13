from datetime import datetime
from feature_extractor import ResNet18
import math
import json
import torch
from ytid import YTID
from layers import *
from attrdict import AttrDict
import argparse
from log_utils import Logging
from layers import ImageGenerator, Discriminator
import os

from data_modules import MNISTDataModule, CIFAR10DataModule

import pytorch_lightning as pl

import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('Set YTID', add_help=False)

    # * Training Parameters
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--t_lr', default=1e-4, type=float)
    parser.add_argument('--g_lr', default=1e-5, type=float)
    parser.add_argument('--d_lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seq_len', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone_ckpt', default='', type=str,
                        help="Path of the saved weights for the backbone")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--ff_dim', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--emb_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * GAN
    parser.add_argument('--ngf', default=16, type=int,
                        help="Number of Generator filters")
    parser.add_argument('--ndf', default=16, type=int,
                        help="Number of Discriminator filters")

    # * Dataset parameters        
    parser.add_argument('--data_module', default='mnist',
                        help="Name of the Pytorch Lightning DataModule to load")
    parser.add_argument('--data_path', default="data/",
                        help="Directory where data is stored")
    parser.add_argument('--validation_split', default=0.1, type=float,
                        help="Fraction of data to put in validation")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--persistent_workers', action='store_true')

    # * Data parameters
    parser.add_argument('--num_classes', default='10', type=str,
                        help="Number of classes in the dataset")
    parser.add_argument('--img_w', default=32, type=int,
                        help="Image width")
    parser.add_argument('--img_h', default=32, type=int,
                        help="Image height")
    parser.add_argument('--num_channels', default=1, type=str,
                        help="Number of channels of the images")

    # * Other parameters
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_weights_every', default=3,
                        help='Frequency to save weights')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--debug', action='store_true')


    return parser

def main(args):

    # Setup
    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data.batch_size % args.data.seq_len == 0, "Batch size not divisible by sequence length"

    os.makedirs(args.trainer.output_dir, exist_ok=True)
    os.makedirs(args.trainer.weight_dir, exist_ok=True)
    
    args['data']['seq_bs'] = int(args.data.batch_size / args.data.seq_len)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image_size = (args.data.img_w * int(math.sqrt(args.model.seq_len)), args.data.img_h * int(math.sqrt(args.model.seq_len)))
    image_size = (args.data.img_w, args.data.img_h)

    feature_extractor = ResNet18(args.model.fe_weights_path, num_classes=args.model.num_classes)

    # feature_extractor_cpu = ResNet18CPU(args.model.fe_weights_path, args.model.num_classes)

    # Data Loading

    if args.name == "MNIST":
        data_module = MNISTDataModule(args.data, image_size, feature_extractor, debug=args.debug)
    elif args.name == "CIFAR10":
        data_module = CIFAR10DataModule(args.data, feature_extractor, debug=args.debug)

    # Logging
    
    current_date = datetime.now()
    formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'

    wandb.login()

    hparams = dict(
        name = args.name,
        ff_dim = args.model.ff_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.model.n_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = args.model.n_heads, # the number of heads in the multiheadattention models
        dropout = args.model.dropout, # the dropout value
        seq_len = args.data.seq_len,
        seq_bs = args.data.seq_bs,
        t_lr = args.optimizer.t_lr,
        g_lr = args.optimizer.g_lr,
        d_lr = args.optimizer.d_lr,
        emb_size = args.model.emb_size,
        img_size = args.data.img_h,
        one_batch = args.overfit_batches,
        ngf = args.model.ngf,
        ndf = args.model.ndf
    )

    if args.debug:
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

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="logs",
        name=run_name,
        version=formatted_date,
        log_graph=True,
        # default_hp_metric=False,
    )

    # Modules Instances

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
    # disc = Discriminator(args.model.emb_size, ndf=args.model.ngf, channels=args.data.n_channels)
    # img_gen = ImageGenerator(image_size, emb_size=args.model.emb_size, ngf=args.model.ngf, channels=args.data.n_channels).to(args.device)


    data_module.setup()
    example_input = data_module.get_example_batch()

    img_gen = Generator(args.model.emb_size, (args.data.n_channels, args.data.img_h * 2, args.data.img_w * 2))
    disc = Discriminator((args.data.n_channels, args.data.img_h, args.data.img_w))

    batches_fraction = 0.0
    limit_val_batches = 1.0
    if args.overfit_batches:
        limit_val_batches = 0.0
        batches_fraction = math.ceil(args.data.batch_size / len(data_module.train_dataset))

    model = YTID(hparams, transformer, img_gen, disc, feature_extractor, example_input, args).to(args.device)

    # Define trainer module
    

    trainer = pl.Trainer(
        gpus=args.n_gpu,
        logger=tb_logger,
        # default_hp_metric=False,
        # fast_dev_run=True,
        overfit_batches=batches_fraction,
        # limit_train_batches=0.25, # 10% of training data
        limit_val_batches=limit_val_batches, # 10% of validation data
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

    try:
        trainer.fit(model, data_module)
    finally:
        print('Exiting...')
        wandb.finish()

    
if __name__ == '__main__':
    with open("config/mnist.json") as conf:
        args = AttrDict(json.load(conf))

    main(args)