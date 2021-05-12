import torch
import torch.nn as nn
import torch.nn.functional as F
from sublayers import MultiHeadAttention, SublayerConnection
import copy

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sublayers import PositionWiseFeedForwardNetwork

# code from https://github.com/lyeoni/nlp-tutorial/tree/master/translation-transformer


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        return ffn_outputs, attn_weights

class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers.
    Args:
        vocab_size (int)    : vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len    (int)    : input sequence length
        d_model    (int)    : number of expected features in the input
        n_layers   (int)    : number of sub-encoder-layers in the encoder
        n_heads    (int)    : number of heads in the multiheadattention models
        p_drop     (float)  : dropout value
        d_ff       (int)    : dimension of the feedforward network model
        pad_id     (int)    : pad token id
    Examples:
    >>> encoder = TransformerEncoder(vocab_size=1000, seq_len=512)
    >>> inp = torch.arange(512).repeat(2, 1)
    >>> encoder(inp)
    """
    
    def __init__(self, vocab_size, seq_len, d_model=512, n_layers=6, n_heads=8, p_drop=0.1, d_ff=2048, pad_id=0):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id
        # self.sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model) # (seq_len+1, d_model)

        # layers
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])
        # layers to classify
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        # positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        # position_pad_mask = inputs.eq(self.pad_id)
        # positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        #outputs = self.embedding(inputs) + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        attention_weights = []
        for layer in self.layers:
            # outputs, attn_weights = layer(outputs, attn_pad_mask)
            outputs, attn_weights = layer(inputs, attn_pad_mask)

            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attention_weights.append(attn_weights)
        
        outputs, _ = torch.max(outputs, dim=1)
        # |outputs| : (batch_size, d_model)
        outputs = self.softmax(self.linear(outputs))
        # |outputs| : (batch_size, 2)

        return outputs, attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.mha2 = MultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_drop)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, inputs, encoder_outputs, attn_mask, enc_dec_attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |encoder_outputs| : (batch_size, encoder_outputs_len, d_model)
        # |attn_mask| : (batch_size, seq_len ,seq_len)
        # |enc_dec_attn_mask| : (batch_size, seq_len, encoder_outputs_len)

        attn_outputs, attn_weights = self.mha1(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len, d_model)
        # |attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=seq_len))

        enc_dec_attn_outputs, enc_dec_attn_weights = self.mha2(attn_outputs, encoder_outputs, encoder_outputs, enc_dec_attn_mask)
        enc_dec_attn_outputs = self.dropout2(enc_dec_attn_outputs)
        enc_dec_attn_outputs = self.layernorm2(attn_outputs + enc_dec_attn_outputs)
        # |enc_dec_attn_outputs| : (batch_size, seq_len, d_model)
        # |enc_dec_attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=encoder_outputs_len))
        
        ffn_outputs = self.ffn(enc_dec_attn_outputs)
        ffn_outputs = self.dropout3(ffn_outputs)
        ffn_outputs = self.layernorm3(enc_dec_attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)

        return ffn_outputs, attn_weights, enc_dec_attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table):
        super(TransformerDecoder, self).__init__()
        self.pad_id = pad_id

        # layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def forward(self, inputs, encoder_inputs, encoder_outputs):
        # |inputs| : (batch_size, seq_len)
        # |encoder_inputs| : (batch_size, encoder_inputs_len)
        # |encoder_outputs| : (batch_size, encoder_outputs_len(=encoder_inputs_len), d_model)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)
        
        outputs = self.embedding(inputs) + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)
        attn_subsequent_mask = self.get_attention_subsequent_mask(inputs).to(device=attn_pad_mask.device)
        # |attn_subsequent_mask| : (batch_size, seq_len, seq_len)
        attn_mask = torch.gt((attn_pad_mask.to(dtype=attn_subsequent_mask.dtype) + attn_subsequent_mask), 0)
        # |attn_mask| : (batch_size, seq_len, seq_len)

        enc_dec_attn_mask = self.get_attention_padding_mask(inputs, encoder_inputs, self.pad_id)
        # |enc_dec_attn_mask| : (batch_size, seq_len, encoder_inputs_len)

        attention_weights, enc_dec_attention_weights = [], []
        for layer in self.layers:
            outputs, attn_weights, enc_dec_attn_weights = layer(outputs, encoder_outputs, attn_mask, enc_dec_attn_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            # |enc_dec_attn_weights| : (batch_size, n_heads, seq_len, encoder_outputs_len)
            attention_weights.append(attn_weights)
            enc_dec_attention_weights.append(enc_dec_attn_weights)

        return outputs, attention_weights, enc_dec_attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask
    
    def get_attention_subsequent_mask(self, q):
        bs, q_len = q.size()
        subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
        # |subsequent_mask| : (batch_size, q_len, q_len)
        
        return subsequent_mask

class Transformer(nn.Module):
    """Transformer is a stack of N encoder/decoder layers.
    Args:
        src_vocab_size (int)    : encoder-side vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        tgt_vocab_size (int)    : decoder-side vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len        (int)    : input sequence length
        d_model        (int)    : number of expected features in the input
        n_layers       (int)    : number of sub-encoder-layers in the encoder
        n_heads        (int)    : number of heads in the multiheadattention models
        p_drop         (float)  : dropout value
        d_ff           (int)    : dimension of the feedforward network model
        pad_id         (int)    : pad token id
    
    Examples:
    >>> model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000, seq_len=512)
    >>> enc_input, dec_input = torch.arange(512).repeat(2, 1), torch.arange(512).repeat(2, 1)
    >>> model(enc_input, dec_input)
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 seq_len,
                 d_model=512,
                 n_layers=6,
                 n_heads=8,
                 p_drop=0.1,
                 d_ff=2048,
                 pad_id=0):
        super(Transformer, self).__init__()
        sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model) # (seq_len+1, d_model)
        
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        # |encoder_inputs| : (batch_size, encoder_inputs_len(=seq_len))
        # |decoder_inputs| : (batch_size, decoder_inputs_len(=seq_len-1))
        
        encoder_outputs, encoder_attns = self.encoder(encoder_inputs)
        # |encoder_outputs| : (batch_size, encoder_inputs_len, d_model)
        # |encoder_attns| : [(batch_size, n_heads, encoder_inputs_len, encoder_inputs_len)] * n_layers

        decoder_outputs, decoder_attns, enc_dec_attns = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)
        # |decoder_outputs| : (batch_size, decoder_inputs_len, d_model)
        # |decoder_attns| : [(batch_size, n_heads, decoder_inputs_len, decoder_inputs_len)] * n_layers
        # |enc_dec_attns| : [(batch_size, n_heads, decoder_inputs_len, encoder_inputs_len)] * n_layers
        
        outputs = self.linear(decoder_outputs)
        # |outputs| : (batch_size, decoder_inputs_len, tgt_vocab_size)
        
        return outputs, encoder_attns, decoder_attns, enc_dec_attns
    
    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class ScheduledOptim:
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=2000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.current_lr = init_lr
    
    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(self.n_steps ** -0.5, self.n_steps * (self.n_warmup_steps ** -1.5))

    def update_learning_rate(self):
        self.n_steps += 1
        self.current_lr = self.init_lr * self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

    @property
    def get_current_lr(self):
        return self.current_lr


class Trainer:
    def __init__(self, args, train_loader, test_loader, tokenizer_src, tokenizer_tgt):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.src_vocab_size = tokenizer_src.vocab_size
        self.tgt_vocab_size = tokenizer_tgt.vocab_size
        self.pad_id = tokenizer_src.pad_token_id # pad_token_id in tokenizer_tgt.vocab should be the same with this.
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        self.model = Transformer(src_vocab_size = self.src_vocab_size,
                                 tgt_vocab_size = self.tgt_vocab_size,
                                 seq_len        = args.max_seq_len,
                                 d_model        = args.hidden,
                                 n_layers       = args.n_layers,
                                 n_heads        = args.n_attn_heads,
                                 p_drop         = args.dropout,
                                 d_ff           = args.ffn_hidden,
                                 pad_id         = self.pad_id)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.optimizer = ScheduledOptim(optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
                                        init_lr=2.0, d_model=args.hidden)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def train(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            encoder_inputs, decoder_inputs, decoder_outputs = map(lambda x: x.to(self.device), batch)
            # |encoder_inputs| : (batch_size, seq_len), |decoder_inputs| : (batch_size, seq_len-1), |decoder_outputs| : (batch_size, seq_len-1)

            outputs, encoder_attns, decoder_attns, enc_dec_attns = self.model(encoder_inputs, decoder_inputs)
            # |outputs| : (batch_size, seq_len-1, tgt_vocab_size)
            # |encoder_attns| : [(batch_size, n_heads, seq_len, seq_len)] * n_layers
            # |decoder_attns| : [(batch_size, n_heads, seq_len-1, seq_len-1)] * n_layers
            # |enc_dec_attns| : [(batch_size, n_heads, seq_len-1, seq_len)] * n_layers
            
            loss = self.criterion(outputs.view(-1, self.tgt_vocab_size), decoder_outputs.view(-1))
            losses += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.update_learning_rate()
            self.optimizer.step()

            if i % (n_batches//5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f}\tlr: {:.4f}'.format(i, i, n_batches, losses/i, self.optimizer.get_current_lr))
        
        print('Train Epoch: {}\t>\tLoss: {:.4f}'.format(epoch, losses/n_batches))
            
    def validate(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                encoder_inputs, decoder_inputs, decoder_outputs = map(lambda x: x.to(self.device), batch)
                # |encoder_inputs| : (batch_size, seq_len), |decoder_inputs| : (batch_size, seq_len-1), |decoder_outputs| : (batch_size, seq_len-1)

                outputs, encoder_attns, decoder_attns, enc_dec_attns = self.model(encoder_inputs, decoder_inputs)
                # |outputs| : (batch_size, seq_len-1, tgt_vocab_size)
                # |encoder_attns| : [(batch_size, n_heads, seq_len, seq_len)] * n_layers
                # |decoder_attns| : [(batch_size, n_heads, seq_len-1, seq_len-1)] * n_layers
                # |enc_dec_attns| : [(batch_size, n_heads, seq_len-1, seq_len)] * n_layers
                
                loss = self.criterion(outputs.view(-1, self.tgt_vocab_size), decoder_outputs.view(-1))
                losses += loss.item()

        print('Valid Epoch: {}\t>\tLoss: {:.4f}'.format(epoch, losses/n_batches))

    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        
        torch.save(self.model, path)