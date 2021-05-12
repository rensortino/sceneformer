import math
import json
import torch
from torch import nn
from log_utils import Logging, show_image, log_prediction
from attrdict import AttrDict
from torch.nn import Transformer
import torchvision.transforms as T
from feature_extractor import ResNet18
from data_processing import append_tokens, get_targets
from data_modules import MNISTDataModule, CIFAR10DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from cgan import weights_init, DCGANDiscriminator

# TODO Generalize labels for CIFAR

TOKENS = {
        "SOS": -1,
        "EOS": -2,
        "PAD": -3
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageGenerator(nn.Module):
    def __init__(self, img_size, emb_size=1024, ngf=128, channels=3):
        super(ImageGenerator, self).__init__()

        assert len(img_size) == 2, "img_size has to be a tuple (h, w)"

        self.emb_size = emb_size
        self.resize = T.Resize(img_size)

        def block(in_feat, out_feat, kernel, stride, pad, last=False):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel, stride, pad, bias=False)]
            return layers

        #self.model = nn.Sequential(
        #    *block(emb_size, ngf * 8, 4, 1, 0),
        #    *block(ngf * 8, ngf * 4, 4, 2, 1),
        #    *block(ngf * 4, ngf * 2, 4, 2, 1),
        #    *block(ngf * 2, ngf, 4, 2, 1),
        #    *block(ngf, ngf // 2, 4, 2, 1),
        #    *block(ngf // 2, channels, 4, 2, 1, last=True)
        #).cuda()

        self.model = nn.Sequential(
            *block(emb_size, channels, 32, 1, 0),
            #*block(ngf, channels, 4, 2, 1, last=True)
        ).to(device)

    def forward(self, x):
        x = x.view(x.size(0) * x.size(1), self.emb_size, 1, 1)
        x = self.model(x)
        return self.resize(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# TODO Input mask should zero attention where there is pad
# Target mask should do this and also the triu mask

class Sceneformer(pl.LightningModule):
    
    def __init__(self, feature_extractor, img_gen, disc, args, device):
        super(Sceneformer, self).__init__()
        self.dev = device
        self.max_seq_len = args.model.max_seq_len
        self.feature_extractor = feature_extractor
        self.pos_enc = PositionalEncoding(args.model.emb_size, args.model.dropout)
        self.embedding = torch.nn.Embedding(args.model.num_classes, args.model.emb_size)
        
        # Variables for logging
        self.train_step = 1
        self.val_step = 1

        self.args = args
        
        # Transformer
        self.transformer = Transformer(
            args.model.emb_size,
            #args.model.seq_bs,
            #args.model.seq_len,
            args.model.n_heads,
            args.model.n_layers,
            args.model.n_layers,
            args.model.ff_dim,
            args.model.dropout)
            #args.model.num_classes,
            #image_size,
            #self.dev)

        self.img_gen = img_gen
        self.disc = disc


        # self.init_weights()

    def forward(self, in_seq, targets, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, batch_first=False):

        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
        """

        return self.transformer(in_seq, targets, None, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask)

    def training_step(self, batch, batch_idx):#, optimizer_idx):

        # Data loading
        images, labels = batch
        images = images.unsqueeze(1) # add the transformer sequence dimension
        images = images.view(args.model.seq_len, args.model.seq_bs, args.data_loader.n_channels, args.data_loader.img_h, args.data_loader.img_w)
        # Model forward
        in_seq = self.embedding(labels)

        step_loss, predictions, target_imgs = self.transformer_step(in_seq, images)
        # Logging
        self.logger.experiment.log({'Train t_loss with custom step': step_loss, 'train_step': self.train_step})
        self.log("Default Transformer Loss", step_loss)
        self.train_step += 1
        if self.global_step % args.trainer.log_every_n_steps == 0:
            target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], args.data_loader.n_channels, target_imgs.shape[3], target_imgs.shape[4])
            log_prediction(target_imgs, predictions, self.logger, title="Train Transformer Target and Ouptut")

        return step_loss
        #return {'loss': step_loss, 'preds': predictions}

    def transformer_loss(self, outputs, targets):
        # loss = torch.nn.KLDivLoss()
        loss = torch.nn.MSELoss()
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return loss(outputs, targets)

    def transformer_step(self, in_seq, images):
        r'''
        Args:
        in_seq : (In_seq_len, N_batchs, Embedding)
        out_seq : (Out_seq_len, N_batchs, Embedding)
        [src/tgt]_mask : what elements to attend (triangular mask)
        [src/tgt]_key_padding_mask : what is padding (True) and what is value (False)

        images shape:  [B,C,H,W]
        '''
        # Create target images
        targets, tgt_imgs = get_targets(self.feature_extractor, images)
        #padded_tgt = get_padded_tgt(targets)

        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        
        in_seq = in_seq.unsqueeze(1).view(args.model.seq_bs, args.model.seq_len, -1)
        in_seq = append_tokens(in_seq, TOKENS['EOS'], TOKENS['SOS'])
        # FIXME Substitute with batch_firts=True
        in_seq = torch.cat(in_seq).permute(1,0,2)

        # Forward transformer
        # tgt[:-1] (shifted right because the transformer has to predict based on previous output)
        in_seq = in_seq.to(device)
        targets = targets.to(device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(targets.shape[0] - 1).to(device)
        trf_out = self(in_seq, targets[:-1], tgt_mask)

        out_imgs = self.img_gen(trf_out)
        # Compute loss
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        t_loss = self.transformer_loss(out_imgs, tgt_imgs)
        self.log("t_loss", t_loss)
        return t_loss, out_imgs, tgt_imgs


    def training_epoch_end(self, training_step_outputs):
        pass
        # for pred in training_step_outputs:
        #     pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.unsqueeze(1) # add the transformer sequence dimension
        images = images.view(args.model.seq_len, args.model.seq_bs, args.data_loader.n_channels, args.data_loader.img_h, args.data_loader.img_w)
        in_seq = self.embedding(labels)
        step_loss, predictions, target_imgs = self.transformer_step(in_seq, images)
        
        # Logging
        self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
        self.log("Default Transformer Loss", step_loss)

        self.val_step += 1
        if self.global_step % args.trainer.log_every_n_steps == 0:
            target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], args.data_loader.n_channels, target_imgs.shape[3], target_imgs.shape[4])
            log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
        return step_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.model.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

def main(args):

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data_loader.batch_size == args.model.seq_len * args.model.seq_bs, "Batch size is not seq_len * transformer_batch"

    data_module = CIFAR10DataModule(args)
    wandb_logger = WandbLogger(project="MNIST Transformer")
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
        logger=wandb_logger,
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

    wandb.run.name = "CIFAR10 Dataset"

    config = wandb.config

    device = 'cuda:0' if args['n_gpu'] == 1 else 'cpu'

    image_size = (args.data_loader.img_w * int(math.sqrt(args.model.seq_len)), args.data_loader.img_h * int(math.sqrt(args.model.seq_len)))
    
    feature_extractor = ResNet18(args.model.fe_weights_path).to(device)
    img_gen = ImageGenerator(image_size, emb_size=512, channels=args.data_loader.n_channels).to(device)
    disc = DCGANDiscriminator(image_channels=args.data_loader.n_channels)
    model = Sceneformer(feature_extractor, img_gen, disc, args, device)

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config_cifar.json") as conf:
        args = AttrDict(json.load(conf))

    main(args)