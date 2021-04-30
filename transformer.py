import torch.nn as nn
import math
import torchvision.transforms as T
import torch
from typing import Optional, Any
import copy
from torch import Tensor
from feature_extractor import ResNet18
from torchvision.utils import make_grid

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer, ModuleList

w_path = 'ckpt/resnet18_mnist.pt'

hp = dict(
        nhid = 200, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 8, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = 8, # the number of heads in the multiheadattention models
        dropout = 0.2, # the dropout value
        noise_size = 100,
        lr = 2.0,
        seq_len = 4,
        seq_bs = 16,
        batch_size = 64,
        w_path = 'ckpt/resnet18_mnist.pt',
        emb_size = 512,
        epochs = 30,
        num_classes = 10,
        img_h = 16,
        img_w = 16,
        log_every = 20,
    )

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

class ImageGenerator(nn.Module):
    def __init__(self, img_size, emb_size=1024, ngf=128, channels=3):
        super(ImageGenerator, self).__init__()

        assert len(img_size) == 2, "img_size has to be a tuple (h, w)"

        self.emb_size = emb_size
        self.resize = T.Resize(img_size)

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

        #self.model = nn.Sequential(
        #    *block(emb_size, channels, 32, 1, 0),
        #    #*block(ngf, channels, 4, 2, 1, last=True)
        #).cuda()

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

    def __init__(self, feature_extractor, emb_size, decoder_layer, num_layers, img_size, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.feature_extractor = feature_extractor
        self.img_gen = ImageGenerator(img_size, emb_size=emb_size, channels=1)

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

            dec_out = mod(output.view(hp['seq_len'], hp['seq_bs'], -1), memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

            out_img = self.img_gen(dec_out)

            with torch.no_grad():
                output = self.feature_extractor.get_vectors(out_img, out_img.size(0))

        if self.norm is not None:
            output = self.norm(output)

        return out_img

class Transformer(nn.Module):
    def __init__(self, emb_size, batch_size, seq_len, device='cpu', img_size=(32,32), nhead=3, nhid=256, nlayers=3, dropout=0.5, num_classes=10):
        super(Transformer, self).__init__()
        
        self.emb_size = emb_size
        #self.hp = hp
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.device = device
        self.pe = PositionalEncoding(emb_size, dropout)
        self.embedding = nn.Embedding(num_classes, emb_size)
        self.feature_extractor = ResNet18(hp['w_path']).cuda()
        # Encoder
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        # Decoder
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, nhid, dropout)
        self.decoder = ImageDecoder(self.feature_extractor, emb_size, decoder_layers, nlayers, img_size)

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
        grid = torch.zeros(seq_len, c, h, w, device=self.device)
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
        images = images.view(self.seq_len, self.batch_size, img_h, img_w)
        images = images.permute(1,0,2,3) # transpose axes to allow iterating through sequences
        img_grids = [self.get_img_grids(img_seq) for img_seq in images]
        #targets = list(map(lambda x: x.unsqueeze(0), targets))
        target_imgs = torch.cat(img_grids).to(self.device)
        with torch.no_grad():
            targets = self.feature_extractor.get_vectors(target_imgs, target_imgs.shape[0])
        return targets, target_imgs

    def forward(self, in_seq, images, out_seq, src_mask=None, tgt_mask=None):
        seq_len = self.seq_len
        bs = self.batch_size
        #targets = self.feature_extractor.get_vectors(images).view(hp['seq_len'], hp['t_bs'], -1)
        src_mask = self.generate_square_subsequent_mask(seq_len)
        obj_emb = self.embedding(in_seq.view(seq_len, bs)) * math.sqrt(self.emb_size)
        obj_emb = self.pe(obj_emb)
        # Run encoder forward
        enc_out = self.encoder(obj_emb)
        # extract image embeddings and reshape for decoder input
        # Run decoder forward
        out_img = self.decoder(out_seq.reshape(seq_len, bs, -1).cuda(), enc_out)

        return out_img