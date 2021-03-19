import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gcn import GraphConv
from sublayers import *
from transformer import *
import copy
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Transformer

def build_transformer(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class OntoGAN(nn.Module):
  def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
               gconv_dim=128,
               **kwargs):
    super(OntoGAN, self).__init__()

    self.vocab = vocab
    self.image_size = image_size
    self.batch_size = kwargs['batch_size']
    self.embedding_dim = embedding_dim

    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
    self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

    gconv_dim = embedding_dim # Embedding size must be equal to weight matrix (input/output dim)

    self.gconv_net = GraphConv(gconv_dim)

    #mha = MultiHeadAttention(self.embedding_dim, self.embedding_dim, 4)
    #enc = EncoderBlock(self.embedding_dim, 4, 256)
    encoder_args = {
      'input_dim': self.embedding_dim,
      'num_heads': 1,
      'dim_feedforward': self.embedding_dim * 4
    }
    # self.trf = TransformerEncoder(num_objs, 10, self.embedding_dim)

    self.trf = Transformer(embedding_dim)



  def forward(self, objs, triples, obj_to_img=None, triple_to_img=None):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
      means that objects[o] is an object in image i. If not given then
      all objects are assumed to belong to the same image.
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """
    O, T = objs.size(0), triples.size(0)
    s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
    s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
    edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)

    img_to_obj_num = [(obj_to_img == i).sum().item() for i in range(self.batch_size)]
    
    if obj_to_img is None:
      obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

    obj_vecs = self.obj_embeddings(objs)
    obj_vecs_orig = obj_vecs
    pred_vecs = self.pred_embeddings(p)

    # obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    # if self.gconv_net is not None:
    obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

    #Extract object embeddings for each image, that will be given to the transformer

    obj_offset = 0

    batch = torch.split(obj_vecs, img_to_obj_num)
    batch = pad_sequence(batch, True) # Sequence length will be 11 (max_obj_num + 1) because of the special __image__ object id
    # Input size (S,N,E) 
    #   S = Sequence length (Number of objects = 10 + 1)
    #   N = Batch Size
    #   E = Embedding Size
    x = self.trf(batch.transpose(1,0))


    for i in range(self.batch_size):
      img_obj_vecs = obj_vecs[obj_offset : obj_offset + img_to_obj_num[i]]     # Object processed vectors for i-th image 
      # y = pred_vecs[pred_offset : pred_offset + img_to_pred_num[i]] # Predicate processed vectors for i-th image
      # z = edges[pred_offset : pred_offset + img_to_pred_num[i]]     # Edges for i-th image
      # z = z % img_to_obj_num[i]                                     # Normalized edges for local batch

      # subjects = x[z[:,0]]
      # objects = x[z[:,1]]


      # image_triples = torch.stack([subjects, y[i], objects], dim=1)   

     

      

      obj_offset += img_to_obj_num[i]
      #pred_offset += img_to_pred_num[i]

    # return img, boxes_pred, masks_pred, rel_scores
    return img_obj_vecs