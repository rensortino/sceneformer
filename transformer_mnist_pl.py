import math
import json
import torch
from torch import nn
from log_utils import Logging, show_image, log_prediction
from attrdict import AttrDict
from torchvision.utils import make_grid
from torchvision import datasets
from torch.nn import Transformer
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from feature_extractor import ResNet18

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from cgan import weights_init, DCGANGenerator, DCGANDiscriminator

# TODO Change cuda() to to(device)

TOKENS = {
        "SOS": -1,
        "EOS": -2,
        "PAD": -3
    }

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
        ).cuda()

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

def get_img_grids(img_seq):
    '''
    img_ seq = [seq_len, h, w] = [4, 16, 16]
    '''
    img_seq = img_seq.unsqueeze(1) # Restore Channel dim
    seq_len, c, h, w = img_seq.shape
    grid = torch.zeros(seq_len, c, h, w, device=img_seq.device)
    img_grids = []
    for i, img in enumerate(img_seq):
        # Place the image in the right quadrant
        grid[i] = img
        # Construct the grid from the batch of images
        img_grid = make_grid(grid.cpu(), nrow=2, padding=0)
        # Take the first channel (Grayscale images, the channels are all the same)
        img_grids.append(img_grid[0].unsqueeze(0).unsqueeze(0)) # Restore channel and batch dimensions
    return torch.cat(img_grids)
    # img_grids = pad_sequence(img_grids, 6, torch.zeros(img_grids[0].shape))

def pad_sequence(seq, max_seq_len):
    '''
    seq: [seq_len, embedding_size]
    '''
    if(len(seq) == max_seq_len):
        return seq.tolist()
    elif (len(seq) > max_seq_len):
        raise("Sequence longer than allowed")
    else :
        # masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
        padded_seq = seq.tolist() + [torch.full([seq.shape[1]], TOKENS['PAD']) for _ in range(max_seq_len - len(seq))]

        return padded_seq


def get_targets(feature_extractor, images, n_channels=1):
    
    '''
    images shape: [seq_len, seq_batch, h, w]
    '''

    # Iterate over sequences
    img_grids = [get_img_grids(images[:,i,:,:]) for i in range(images.shape[1])]

    tgt_images = torch.cat(img_grids, dim=1).to(images.device)
    # seq_len, seq_bs, h, w = tgt_images.shape
    # tgt_images = tgt_images.view(seq_len * seq_bs, n_channels, h, w)

    eos_token = torch.full([1] + list(tgt_images.shape[1:]), TOKENS['EOS'])
    tgt_images = torch.cat((tgt_images.cpu(), eos_token)).cuda()
    
    tgt_vectors = []
    for grid_seq in img_grids:
        # For each sequence, add SOS and EOS token to the extracted vectors
        
        with torch.no_grad():
            vector_seq = feature_extractor.get_vectors(grid_seq)
        # TODO Put sos_token delcaration outside loop
        sos_token = torch.full([1, vector_seq.shape[1]], TOKENS['SOS'])
        eos_token = torch.full([1, vector_seq.shape[1]], TOKENS['EOS'])
        vector_seq = torch.cat((sos_token, vector_seq.cpu(), eos_token))
        tgt_vectors.append(vector_seq.unsqueeze(0))

    return torch.cat(tgt_vectors).permute(1,0,2), tgt_images

def get_padded_tgt(tgt_vectors):

    '''
    tgt_vectors: [batch, seq, emb_size]
    '''
        
    max_seq_len = max([vec.shape[0] for vec in tgt_vectors])

    padded_tgt = []
    for vec_seq in tgt_vectors:
        # Embedding sequence
        padded_vec_seq = pad_sequence(vec_seq, max_seq_len)
        padded_tgt.append(padded_vec_seq)

    return padded_tgt

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ 
            T.Resize((args.data_loader.img_h, args.data_loader.img_w)), 
            T.ToTensor(), 
            T.Normalize((mnist.mean()/255,), (mnist.std()/255,))
        ])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.data_loader.batch_size, num_workers=0, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.data_loader.batch_size, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=args.data_loader.batch_size, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

class Sceneformer(pl.LightningModule):
    
    def __init__(self, feature_extractor, img_gen, args, device):
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
            #feature_extractor,
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
        
        # self.disc = DCGANDiscriminator(image_channels=1).cuda()
        # self.disc.apply(weights_init)

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
        images = images.view(args.model.seq_len, args.model.seq_bs, args.data_loader.img_h, args.data_loader.img_w)
        # Model forward
        in_seq = self.embedding(labels)

        step_loss, predictions, target_imgs = self.transformer_step(in_seq, images)
        # Logging
        self.logger.experiment.log({'Train t_loss with custom step': step_loss, 'train_step': self.train_step})
        self.log("Default Transformer Loss", step_loss)
        self.train_step += 1
        if self.global_step % args.trainer.log_every_n_steps == 0:
            target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], 1, target_imgs.shape[2], target_imgs.shape[3])
            log_prediction(target_imgs, predictions, self.logger, title="Train Transformer Target and Ouptut")

        return step_loss
        #return {'loss': step_loss, 'preds': predictions}

    def training_epoch_end(self, training_step_outputs):
        pass
        # for pred in training_step_outputs:
        #     pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(args.model.seq_len, args.model.seq_bs, args.data_loader.img_h, args.data_loader.img_w)
        in_seq = self.embedding(labels)
        step_loss, predictions, target_imgs = self.transformer_step(in_seq, images)
        
        # Logging
        self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
        self.log("Default Transformer Loss", step_loss)

        self.val_step += 1
        if self.global_step % args.trainer.log_every_n_steps == 0:
            target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], 1, target_imgs.shape[2], target_imgs.shape[3])
            log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
        return step_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.model.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    def transformer_loss(self, outputs, targets):
        # loss = torch.nn.KLDivLoss()
        loss = torch.nn.MSELoss()
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return loss(outputs, targets)

    def transformer_step(self, in_seq, images):

        '''
        images shape:  [B,C,H,W]
        '''
        # Create target images
        targets, tgt_imgs = get_targets(self.feature_extractor, images)
        #padded_tgt = get_padded_tgt(targets)

        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        
        in_seq = in_seq.unsqueeze(1).view(args.model.seq_bs, args.model.seq_len, -1)
        #FIXME Export in function
        new_seq = []
        for seq in in_seq:
            eos_token = torch.full([1, seq.shape[1]], TOKENS['EOS'])
            seq = torch.cat((seq.cpu(), eos_token))
            new_seq.append(seq.unsqueeze(0))
        # FIXME Substitute with batch_firts=True
        in_seq = torch.cat(new_seq).permute(1,0,2)

        # Forward transformer
        # tgt[:-1] (shifted right because the transformer has to predict based on previous output)
        in_seq = in_seq.cuda()
        targets = targets.cuda()
        tgt_mask = self.transformer.generate_square_subsequent_mask(targets.shape[0] - 1).cuda()
        trf_out = self(in_seq, targets[:-1], tgt_mask)

        out_imgs = self.img_gen(trf_out)
        # Compute loss
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        t_loss = self.transformer_loss(out_imgs, tgt_imgs)
        self.log("t_loss", t_loss)
        return t_loss, out_imgs, tgt_imgs

def main(args):

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data_loader.batch_size == args.model.seq_len * args.model.seq_bs, "Batch size is not seq_len * transformer_batch"

    data_module = MNISTDataModule()
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
        #mode="disabled"
    )

    

    wandb.run.name = "MSE Loss"

    config = wandb.config

    device = 'cuda:0' if args['n_gpu'] == 1 else 'cpu'

    image_size = (args.data_loader.img_w * int(math.sqrt(args.model.seq_len)), args.data_loader.img_h * int(math.sqrt(args.model.seq_len)))

    
    feature_extractor = ResNet18(args.model.fe_weights_path).cuda()
    img_gen = ImageGenerator(image_size, emb_size=512, channels=1).cuda()
    model = Sceneformer(feature_extractor, img_gen, args, device)

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config.json") as c:
        args = AttrDict(json.load(c))

    main(args)