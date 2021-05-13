
from torch.nn import Transformer
import pytorch_lightning as pl
from log_utils import log_prediction
import torch
from torch import nn
import torchvision.transforms as T
import math
from data_processing import get_targets, append_tokens
from torchviz import make_dot, make_dot_from_trace
from graphviz import Source

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
        #)

        self.model = nn.Sequential(
            *block(emb_size, ngf, 32, 1, 0),
            *block(ngf, channels, 4, 2, 1, last=True)
        )

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
        x = x.to(self.pe.device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class YTID(pl.LightningModule):
    
    def __init__(self,
                feature_extractor,
                embedding,
                pos_enc,
                img_gen,
                disc,
                args,
                criterion):
        super(YTID, self).__init__()
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.model.max_seq_len
        self.feature_extractor = feature_extractor
        self.pos_enc = pos_enc
        self.embedding = embedding
        self.criterion = criterion

        # Variables for logging
        self.train_step = 1
        self.val_step = 1

        self.args = args
        
        # Transformer
        self.transformer = Transformer(
            args.model.emb_size,
            args.model.n_heads,
            args.model.n_layers,
            args.model.n_layers,
            args.model.ff_dim,
            args.model.dropout,
        )

        self.img_gen = img_gen
        self.disc = disc

        # self.init_weights()

    def forward(self, labels, targets, tgt_maskk=None, src_key_padding_mask=None, tgt_key_padding_mask=None):

        r"""Take in and process masked source/target sequences.

        Args:
            src/tgt: the sequence to the encoder/decoder (required).
            [src/tgt]_mask: the additive mask for the src/tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            [src/tgt]_key_padding_mask: the ByteTensor mask for src/tgt keys per batch (optional).
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

        # Create input embeddings
        in_seq = self.embedding(labels)* math.sqrt(self.args.model.emb_size)
        in_seq = in_seq.unsqueeze(1).view(self.args.model.seq_bs, self.args.model.seq_len, -1)
        in_seq = append_tokens(in_seq, TOKENS['EOS'])
        in_seq = self.pos_enc(in_seq)
        in_seq = in_seq.permute(1,0,2)

        # Forward transformer
        # tgt[:-1] (shifted right because the transformer has to predict based on previous output)
        in_seq = in_seq.to(self.args.device)
        targets = targets.to(self.args.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(targets.shape[0] - 1).to(self.args.device)
        trf_out = self.transformer(in_seq, targets[:-1], tgt_mask)

        out_imgs = self.img_gen(trf_out)
        return out_imgs

    def training_step(self, batch, batch_idx):#, optimizer_idx):

        # Data loading
        images, labels = batch
        images = images.unsqueeze(1) # add the transformer sequence dimension
        images = images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data_loader.n_channels, self.args.data_loader.img_h, self.args.data_loader.img_w)

        opt = self.optimizers()
        opt.zero_grad()
        
        # Model forward
        step_loss, predictions, target_imgs = self.transformer_step(labels, images)

        self.manual_backward(step_loss)
        opt.step()

        # Logging
        #self.logger.experiment.log({'Train t_loss': step_loss, 'train_step': self.train_step})
        self.logger.experiment.add_scalar('Loss/Train', step_loss, self.train_step)
        self.log('Default Train t_loss', step_loss)
        self.train_step += 1
        if self.global_step % self.args.trainer.log_every_n_steps == 0:
            target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], self.args.data_loader.n_channels, target_imgs.shape[3], target_imgs.shape[4])
            log_prediction(target_imgs, predictions, self.logger, title="Train Transformer Target and Ouptut")

        return step_loss
        #return {'loss': step_loss, 'preds': predictions}

    def transformer_step(self, labels, images):
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
        out_imgs = self(labels, targets)
        # Compute loss
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        t_loss = self.transformer_loss(self.criterion, out_imgs, tgt_imgs)
        self.log("t_loss", t_loss)
        return t_loss, out_imgs, tgt_imgs

    def transformer_loss(self, criterion,  outputs, targets):

        targets = targets.view(targets.shape[0] * targets.shape[1], -1)
        outputs = outputs.view(targets.shape)

        loss = 0

        for i in range(outputs.shape[0]):
            seq_loss = criterion(outputs[i], targets[i])
            loss += seq_loss

        
        return loss / (outputs.shape[0] * outputs.shape[1])

    def custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_epoch_end(self,outputs):
        if(self.current_epoch==1):
            sampleImg=torch.rand((4,16,512))
            self.logger.experiment.add_graph(YTID(), sampleImg)

        self.custom_histogram_adder()

 


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.unsqueeze(1) # add the transformer sequence dimension
        # Reshape images to seq_len, batch_size
        images = images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data_loader.n_channels, self.args.data_loader.img_h, self.args.data_loader.img_w)
        step_loss, predictions, target_imgs = self.transformer_step(labels, images)
        
        # Logging
        #self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
        self.logger.experiment.add_scalar('Loss/Val', step_loss, self.val_step)
        self.log("Default Transformer Loss", step_loss)

        self.val_step += 1
        if self.global_step % self.args.trainer.log_every_n_steps == 0:
            target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], self.args.data_loader.n_channels, target_imgs.shape[3], target_imgs.shape[4])
            log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
        return step_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.model.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]
