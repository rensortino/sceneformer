import math, time, random, cv2
import torch
import copy
from log_utils import Logging, show_image, log_prediction
from torchvision.utils import make_grid
from typing import Optional, Any
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer, ModuleList

from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

import wandb

from cgan import weights_init, DCGANGenerator, DCGANDiscriminator
from feature_extractor import ResNet18


# TODO Current objective is to translate embedding (flattened image) to sequences maintaining an ordering between successive elements

hp = dict(
        nhid = 200, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 8, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = 8, # the number of heads in the multiheadattention models
        dropout = 0.2, # the dropout value
        noise_size = 100,
        lr = 2.0,
        seq_len = 4,
        t_bs = 16,
        batch_size = 64,
        w_path = 'ckpt/resnet18_mnist.pt',
        emb_size = 512,
        epochs = 30,
        num_classes = 10,
        img_h = 16,
        img_w = 16,
        log_every = 20,
    )

assert hp['emb_size'] % hp['nheads'] == 0, "Embedding size not divisible by number of heads"
assert hp['batch_size'] == hp['seq_len'] * hp['t_bs'], "Batch size is not seq_len * transformer_batch"

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ T.Resize((hp['img_h'], hp['img_w'])), T.ToTensor(), T.Normalize((mnist.mean()/255,), (mnist.std()/255,))])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=True)

class ImageGenerator(nn.Module):
    def __init__(self, img_size, emb_size=512, ngf=128, channels=3):
        super(ImageGenerator, self).__init__()

        self.emb_size = emb_size
        self.resize = T.Resize((img_size, img_size))

        def block(in_feat, out_feat, kernel, stride, pad, last=False):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel, stride, pad, bias=False)]
            return layers

        self.model = nn.Sequential(
            *block(emb_size, ngf * 8, 4, 1, 0),
            *block(ngf * 8, ngf * 4, 4, 2, 1),
            *block(ngf * 4, ngf * 2, 4, 2, 1),
            *block(ngf * 2, ngf, 4, 2, 1),
            *block(ngf, ngf // 2, 4, 2, 1),
            *block(ngf // 2, channels, 4, 2, 1, last=True)
        ).cuda()

    def forward(self, x):
        x = x.view(x.size(0) * x.size(1), self.emb_size, 1, 1)
        x = self.model(x)
        return self.resize(x)

class ImageDecoder(nn.TransformerDecoder):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, img_size, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.feature_extractor = ResNet18(hp['w_path'])
        self.img_gen = ImageGenerator(img_size, channels=1)

    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:

            dec_out = mod(output.view(hp['seq_len'], hp['t_bs'], -1), memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

            out_img = self.img_gen(dec_out)
            
            output = self.feature_extractor.get_vectors(out_img, out_img.size(0))

        if self.norm is not None:
            output = self.norm(output)

        return output, out_img

class Transformer(nn.Module):
    def __init__(self, emb_size, img_size=32, nhead=3, nhid=256, nlayers=3, dropout=0.5, num_classes=10):
        super(Transformer, self).__init__()
        
        self.emb_size = emb_size
        self.embedding = nn.Embedding(num_classes, emb_size)
        self.feature_extractor = ResNet18(hp['w_path']).cuda()
        # self.resize = T.Resize((img_size, img_size))
        # Encoder
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        # Decoder
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, nhid, dropout)
        self.decoder = ImageDecoder(decoder_layers, nlayers, img_size)

        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_img_grids(self, img_seq):
        '''
        img_ seq = [seq_len, h, w] = [4, 16, 16]
        '''
        img_seq = img_seq.unsqueeze(1) # Restore Channel dim
        seq_len, c, h, w = img_seq.shape
        grid = torch.zeros(seq_len, c, h, w).cuda()
        img_grids = []
        for i, img in enumerate(img_seq):
            # Place the image in the right quadrant
            grid[i] = img
            # Construct the grid from the batch of images
            img_grid = make_grid(grid.cpu(), nrow=2, padding=0)
            # Take the first channel (Grayscale images, the channels are all the same)
            img_grids.append(img_grid[0].unsqueeze(0).unsqueeze(0)) # Restore channel and batch dimensions
            #img_embedding = self.feature_extractor.get_vectors(img_grid[0], hp['batch_size'])
            #img_embeddings.append(img_embedding.unsqueeze(0)) # Unsqueeze to cat along dim 0 later
        return torch.cat(img_grids)

    def get_targets(self, images):
        img_h, img_w = images.shape[2:]
        images = images.view(hp['seq_len'], hp['t_bs'], img_h, img_w)
        images = images.permute(1,0,2,3) # transpose axes to allow iterating through sequences
        img_grids = [self.get_img_grids(img_seq) for img_seq in images]
        #targets = list(map(lambda x: x.unsqueeze(0), targets))
        target_imgs = torch.cat(img_grids)
        targets = self.feature_extractor.get_vectors(target_imgs, target_imgs.shape[0])
        return targets, target_imgs

    def forward(self, in_seq, images, out_seq, src_mask=None, tgt_mask=None):
        seq_len = hp['seq_len']
        bs = hp['t_bs']
        #targets = self.feature_extractor.get_vectors(images).view(hp['seq_len'], hp['t_bs'], -1)
        src_mask = self.generate_square_subsequent_mask(seq_len)
        obj_emb = self.embedding(in_seq.view(seq_len, bs)) #* math.sqrt(self.emb_size)
        # Run encoder forward
        enc_out = self.encoder(obj_emb)
        # extract image embeddings and reshape for decoder input
        # Run decoder forward
        output, out_img = self.decoder(out_seq.reshape(seq_len, bs, -1).cuda(), enc_out)

        return output, out_img



class Sceneformer(pl.LightningModule):
    
    def __init__(self, emb_size: int = 512):
        super(Sceneformer, self).__init__()
        
        # Transformer
        # TODO Fix hp values
        image_size = hp['img_w'] * int(math.sqrt(hp['seq_len']))
        self.transformer = Transformer(emb_size, image_size, hp['nheads'], hp['nhid'], hp['nlayers'], hp['dropout'], hp['num_classes'])
        
        # self.disc = DCGANDiscriminator(image_channels=1).cuda()
        # self.disc.apply(weights_init)

        self.predictions = None
        self.target_imgs = None
        # self.init_weights()

    def forward(self, labels, images, targets, src_mask=None, tgt_mask=None):
        
        return self.transformer(labels, images, targets)

    def training_step(self, batch, batch_idx):#, optimizer_idx):

        # Data loading
        images, labels = batch
        self.transformer.c = self.transformer.feature_extractor.get_vectors(images, 64)
        # Model forward
        step_loss, predictions, target_imgs = self.transformer_step(labels, images)
        self.log('Train t_loss', step_loss)
        if self.global_step % hp['log_every'] == 0:
            log_prediction(target_imgs, predictions, self.logger, title="Train Transformer Target and Ouptut")

        return {'loss': step_loss, 'preds': predictions}

    def training_epoch_end(self, training_step_outputs):
        pass
        # for pred in training_step_outputs:
        #     pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        step_loss, predictions, targets = self.transformer_step(labels, images)
        self.log('Val t_loss', step_loss)
        #self.log_prediction(targets, predictions, self.logger, "Validation Transformer Target and Ouptut")
        return step_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizerT = torch.optim.SGD(self.parameters(), lr=hp['lr'])
        return [optimizerT] 

    def transformer_loss(self, outputs, targets):
        #loss = nn.KLDivLoss()
        loss = torch.nn.MSELoss()
        outputs = outputs.cuda() #.view(-1, hp['emb_size'])
        targets = targets.cuda() #.view(-1, hp['emb_size'])
        return loss(outputs, targets)

    # TODO Change embedding to resnet18 feature extraction
    def transformer_step(self, labels, images):

        targets, target_imgs = self.transformer.get_targets(images)
        trf_output, predictions = self(labels.unsqueeze(1), images, targets)
        t_loss = self.transformer_loss(predictions, target_imgs)
        return t_loss, predictions, target_imgs

    

def main():
    
    model = Sceneformer(hp['emb_size'])

    wandb.login()

    wandb.init(
        config=hp,
        #mode="disabled"
    )

    config = wandb.config

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # input should be three-dimensional (Seq_len, N_batchs, Embedding)

    data_module = MNISTDataModule()
    # tb_logger = TensorBoardLogger('tb_logs', name='mnist_transformer')
    wandb_logger = WandbLogger(project="MNIST Transformer")
    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        # overfit_batches=0.01, # 1% of training set used as batch to make it overfit
        # limit_train_batches=0.1, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        num_sanity_val_steps=0,
        flush_logs_every_n_steps=100,
        progress_bar_refresh_rate=20,
        profiler=True,
        logger=wandb_logger,
        callbacks=[
            Logging(),
            #TensorboardGenerativeModelImageSampler()
        ]
    )
    trainer.fit(model, data_module)
    
main()

