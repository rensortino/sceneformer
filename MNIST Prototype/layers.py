import math
import numpy as np
from torch import nn
from torch.functional import Tensor, split
import torch.nn.functional as F
import torch
from data_processing import *
from torch.nn.init import xavier_uniform_
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import torchvision.transforms as T


class ImageGenerator(nn.Module):
    def __init__(self, img_size, emb_size=1024, ngf=64, channels=3):
        super(ImageGenerator, self).__init__()

        assert len(img_size) == 2, "img_size has to be a tuple (h, w)"

        self.emb_size = emb_size
        self.resize = T.Resize(img_size)

        def block(in_feat, out_feat, kernel, stride, pad, last=False):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel, stride, pad, bias=False)]
            if last:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.BatchNorm2d(out_feat))
                layers.append(nn.LeakyReLU(0.2, True))

            return layers

        self.model = nn.Sequential(
            *block(emb_size, ngf * 4, 4, 1, 0),
            *block(ngf * 4, ngf * 2, 4, 2, 1),
            *block(ngf * 2, ngf, 4, 2, 1),
            *block(ngf, channels, 4, 2, 1, last=True)
        )

    def forward(self, x):
        x = x.view(x.size(0) * x.size(1), self.emb_size, 1, 1)
        return self.model(x)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(self.pe.device)
        x = x * math.sqrt(self.emb_size)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ImageTransformer(nn.Module):
    def __init__(self,  emb_size,
                        n_heads,
                        n_enc_layers,
                        n_dec_layers,
                        ff_dim,
                        dropout,
                        data_options,
                        image_size,
                        device,
                        feature_extractor):
        super(ImageTransformer, self).__init__()

        self.data_options = data_options

        norm = LayerNorm(emb_size)

        # Encoder
        encoder_layer = TransformerEncoderLayer(emb_size, n_heads, ff_dim, dropout)
        self.encoder = TransformerEncoder(encoder_layer, n_enc_layers, norm)

        # Decoder
        decoder_layer = TransformerDecoderLayer(emb_size, n_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layer, n_dec_layers, norm)

        self.emb_size = emb_size
        self.device = device

        self.image_size = image_size

        # TODO Fix names
        src_vocab_size = 16
        tgt_vocab_size = 12

        self.pos_enc = PositionalEncoding(emb_size, dropout).to(device)
        self.src_embedding = torch.nn.Embedding(src_vocab_size, emb_size).to(device)
        # self.dim_reduction = nn.Linear(8192, self.emb_size)
        # self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, emb_size).to(device)
        self.tgt_embedding = feature_extractor
        # TODO Parametrize
        self.box_embedding = torch.nn.Linear(4, 256)
        # TODO Parametrize
        self.box_generator = torch.nn.Linear(256, 4)
        # TODO Parametrize
        self.feature_generator = torch.nn.Linear(256, 256)

        # emb_size includes both classes (12 = 10 + 2) and bbox coordinates (4)
        # TODO Parametrize
        self.fc = nn.Linear(256, tgt_vocab_size).to(device)
        # self.fc_same_dim = LinearSameDim(emb_size).to(device)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'tgt_embedding' not in n:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):

        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, memory, memory_mask, tgt, tgt_mask, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

    def get_probs(self, logits):
        return self.fc(logits)

    def greedy_decode(self, src, src_mask=None, max_len=6, sos_img_tk=None, sos_box_tk=None):
        src = self.src_embedding(src.int())
        src = self.pos_enc(src)
        memory = self.encode(src, src_mask)

        ys = torch.cat((sos_img_tk, sos_box_tk), dim=2).type_as(memory)

        # ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
        for i in range(max_len-1):
            out = self.decode(memory, src_mask, 
                            ys, 
                            self.generate_square_subsequent_mask(ys.size(0))
                                        .type_as(src))

            probs_and_boxes = self.fc(out)
            probs, boxes = split_list(probs_and_boxes)
            _, classes = probs.max(dim=2)
            print(classes)

            ys = torch.cat([ys, out[-1].unsqueeze(0)])
        return ys

    def forward(self, src, targets, src_key_padding_mask=None, tgt_key_padding_mask=None):

        r"""Take in and process masked source/target sequences.

        Args:
            src/tgt: the sequence to the encoder/decoder (required). Shape: (S/T, N, E)
            [src/tgt]_mask: the additive mask for the src/tgt sequence (optional). Shape: (S/T, S/T)
            memory_mask: the additive mask for the encoder output (optional). Shape: (T, S)
            [src/tgt]_key_padding_mask: the ByteTensor mask for src/tgt keys per batch (optional). Shape: (N, S/T)
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional). Shape: (N, S)
        """

        # Create input and target embeddings
        src = self.src_embedding(src.int())
        src = self.pos_enc(src)

        seq_len, bs = targets.shape[0], targets.shape[1]

        features, boxes = split_list(targets, 4)
        boxes = self.box_embedding(boxes)
        targets = torch.cat((features, boxes), dim=2)
        
        targets = self.pos_enc(targets)

        # Forward transformer
        tgt_mask = self.generate_square_subsequent_mask(seq_len)
        
        memory = self.encode(src)
        out = self.decode(memory, None, targets, tgt_mask)

        features, boxes = split_list(out, 256)

        # Remove negative values
        boxes = nn.functional.sigmoid(boxes)
        boxes = self.box_generator(boxes)
        # features = self.feature_generator(features)

        logits = self.get_probs(features)

        return features, logits, boxes

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        

def test_model_size(model, x):
    for i, m in enumerate(model):
        print(f'Module: {i}')
        print(f'Input: {x.shape}')
        print(f'Model: {m}')
        x = m(x)
        print(f'Output: {x.shape}\n')