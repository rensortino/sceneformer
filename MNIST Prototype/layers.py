import math
from torch import nn
import torch


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
        )

    def forward(self, x):
        x = x.view(x.size(0) * x.size(1), self.emb_size, 1, 1)
        x = self.model(x)
        return self.resize(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10):
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

class ImageTransformer(nn.Module):
    def __init__(self,  emb_size,
                        n_heads,
                        n_enc_layers,
                        n_dec_layers,
                        ff_dim,
                        dropout,
                        data_options,
                        device,
                        embedding: nn.Module,
                        img_gen: nn.Module,
                        pos_enc: nn.Module):

        super(ImageTransformer, self).__init__()

        self.data_options = data_options

        # Transformer
        self.transformer = nn.Transformer(
            emb_size,
            n_heads,
            n_enc_layers,
            n_dec_layers,
            ff_dim,
            dropout,
        )

        self.emb_size = emb_size
        self.device = device

        self.embedding = embedding
        self.img_gen = img_gen
        self.pos_enc = pos_enc

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, targets, src_key_padding_mask=None, tgt_key_padding_mask=None):

        r"""Take in and process masked source/target sequences.

        Args:
            src/tgt: the sequence to the encoder/decoder (required). Shape: (S/T, N, E)
            [src/tgt]_mask: the additive mask for the src/tgt sequence (optional). Shape: (S/T, S/T)
            memory_mask: the additive mask for the encoder output (optional). Shape: (T, S)
            [src/tgt]_key_padding_mask: the ByteTensor mask for src/tgt keys per batch (optional). Shape: (N, S/T)
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional). Shape: (N, S)
        """

        # Create input embeddings
        in_seq = self.embedding(src.int())* math.sqrt(self.emb_size)
        in_seq = self.pos_enc(in_seq)
        targets = self.pos_enc(targets)

        # Forward transformer
        in_seq = in_seq.to(self.device)
        targets = targets.to(self.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(targets.shape[0]).to(self.device)
        trf_out = self.transformer(in_seq, targets, tgt_mask=tgt_mask)
        image_vector = trf_out[:,:,:-4]
        bbox = trf_out[:,:,-4:]

        # TODO Restore img_gen
        #out_imgs = self.img_gen(trf_out)
        out_imgs = image_vector.reshape(
            image_vector.shape[0] * image_vector.shape[1],
            self.data_options.n_channels, 
            self.data_options.img_w, 
            self.data_options.img_h
        )

        return out_imgs, image_vector, bbox

