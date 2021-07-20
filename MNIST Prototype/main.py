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
    parser.add_argument('--overfit_batches', action='store_true')
    parser.add_argument('--log_weights_change', action='store_true')
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--max_seq_len', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # * Optimizer
    parser.add_argument('--t_lr', default=1e-4, type=float)
    parser.add_argument('--g_lr', default=1e-2, type=float)
    parser.add_argument('--d_lr', default=1e-5, type=float)
    parser.add_argument('--opt', default="Adam", type=str)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # TODO Here
    # parser.add_argument('--backbone_ckpt', default='ckpt/mnist/resnet18_10classes.pt', type=str,
    #                     help="Path of the saved weights for the backbone")

    # * Transformer
    parser.add_argument('--n_enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--n_dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--ff_dim', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--emb_size', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--n_heads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * GAN
    parser.add_argument('--ngf', default=16, type=int,
                        help="Number of Generator filters")
    parser.add_argument('--ndf', default=16, type=int,
                        help="Number of Discriminator filters")

    # * Dataset parameters        
    parser.add_argument('--data_module', default='mnist',
                        help="Name of the Pytorch Lightning DataModule to load")
    parser.add_argument('--data_dir', default="data/",
                        help="Directory where data is stored")
    parser.add_argument('--validation_split', default=0.1, type=float,
                        help="Fraction of data to put in validation")
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--persistent_workers', action='store_true')

    # * Data parameters # TODO Replace with config file
    # parser.add_argument('--n_classes', default=10, type=int,
    #                     help="Number of classes in the dataset")
    # parser.add_argument('--img_w', default=28, type=int,
    #                     help="Image width")
    # parser.add_argument('--img_h', default=28, type=int,
    #                     help="Image height")
    # parser.add_argument('--n_channels', default=1, type=str,
    #                     help="Number of channels of the images")

    # * Other parameters
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--weight_dir', default='weights',
                        help='path where to save weigths, empty for no saving')
    parser.add_argument('--tb_dir', default='logs',
                        help='path where to save tensorboard logs')
    parser.add_argument('--log_every_n_steps', default=25,
                        help='Frequency to log gradients and images')
    parser.add_argument('--save_weights_every', default=3,
                        help='Frequency to save weights')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')


    return parser

def main(args):

    # Setup
    assert args.emb_size % args.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.batch_size % args.seq_len == 0, "Batch size not divisible by sequence length"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.weight_dir, exist_ok=True)
    
    args.seq_bs = int(args.batch_size / args.seq_len)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image_size = (args.data.img_w * int(math.sqrt(args.seq_len)), args.data.img_h * int(math.sqrt(args.seq_len)))
    image_size = (args.data.img_w, args.data.img_h)

    feature_extractor = ResNet18(args.data.backbone_ckpt, num_classes=args.data.n_classes)

    # Data Loading

    if args.data_module == "mnist":
        data_module = MNISTDataModule(args, image_size, feature_extractor, debug=args.debug)
    elif args.data_module == "cifar":
        data_module = CIFAR10DataModule(args, feature_extractor, debug=args.debug)

    # Logging
    
    current_date = datetime.now()
    formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'

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
        save_dir=args.tb_dir,
        name=run_name,
        version=formatted_date,
        log_graph=True,
        # default_hp_metric=False,
    )

    # Modules Instances

    transformer = ImageTransformer(
            args.emb_size,
            args.n_heads,
            args.n_enc_layers,
            args.n_dec_layers,
            args.ff_dim,
            args.dropout,
            args,
            image_size,
            args.device,
            feature_extractor
        ).to(args.device)
    # disc = Discriminator(args.emb_size, ndf=args.ngf, channels=args.data.n_channels)
    # img_gen = ImageGenerator(image_size, emb_size=args.emb_size, ngf=args.ngf, channels=args.data.n_channels).to(args.device)


    data_module.setup()
    example_input = data_module.get_example_batch()

    img_gen = Generator(args.emb_size, (args.data.n_channels, args.data.img_h * 2, args.data.img_w * 2))
    disc = Discriminator((args.data.n_channels, args.data.img_h, args.data.img_w))

    batches_fraction = 0.0
    limit_val_batches = 1.0
    if args.overfit_batches:
        limit_val_batches = 0.0
        batches_fraction = math.ceil(args.batch_size / len(data_module.train_dataset))

    model = YTID(hparams, transformer, img_gen, disc, feature_extractor, example_input, args).to(args.device)

    # Define trainer module
    

    trainer = pl.Trainer(
        gpus=args.n_gpu,
        logger=tb_logger,
        # default_hp_metric=False,
        # fast_dev_run=True,
        overfit_batches=batches_fraction,
        limit_train_batches=0.25, # 10% of training data
        limit_val_batches=limit_val_batches, # 10% of validation data
        num_sanity_val_steps=0,
        flush_logs_every_n_steps=20,
        progress_bar_refresh_rate=20,
        max_epochs=args.epochs,
        #profiler=True, # This is very time consuming
        callbacks=[
            Logging(tb_logger),
        ]
    )

    # Load weights

    if args.resume:
        checkpoint = torch.load(f'{args.weight_dir}/checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        trainer.current_epoch = checkpoint['epoch']

    try:
        trainer.fit(model, data_module)
    finally:
        print('Exiting...')
        wandb.finish()

    
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    if args.data_module == 'mnist':
        data_config = 'mnist.json'
    if args.data_module == 'cifar':
        data_config = 'cifar.json'
    
    with open(os.path.join("config", data_config)) as conf:
        data_args = AttrDict(json.load(conf))

    args.data = data_args

    main(args)