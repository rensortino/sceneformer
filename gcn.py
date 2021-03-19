import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

"""
PyTorch modules for dealing with graphs.
"""

def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)


class GraphConvLayer(nn.Module):
  """
  A single layer of scene graph convolution, using PyTorch Geometric.
  """
  def __init__(self, input_dim, gconv_model=GCNConv, output_dim=None):
    super(GraphConvLayer, self).__init__()

    if output_dim is None:
      output_dim = input_dim

    self.input_dim = input_dim
    self.output_dim = output_dim

    self.gconv = gconv_model(input_dim, output_dim)
    self.gconv.apply(_init_weights)


  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, Dout = self.input_dim, self.output_dim
    
    obj_rel = torch.cat([obj_vecs, pred_vecs], dim=0)
    edge_index = []
    for k, (s, o) in enumerate(edges):
        s_p = (s.item(), k)
        p_o = (k, o.item())
        edge_index.append(s_p)
        edge_index.append(p_o)

    edge_index = torch.tensor(edge_index).t().to(device)

    new_embeddings = self.gconv(obj_rel, edge_index)

    new_obj_embeddings = new_embeddings[:O, :]
    new_pred_embeddings = new_embeddings[O:, :]

    return new_obj_embeddings, new_pred_embeddings


class GraphConv(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim, num_layers=5, pooling_dim=32):
    super(GraphConv, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    for _ in range(self.num_layers):
      self.gconvs.append(GraphConvLayer(input_dim))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)

    
    return obj_vecs, pred_vecs


