import torch
from torchvision.utils import make_grid

TOKENS = {
        "SOS": -0.1,
        "EOS": -0.2,
        "PAD": -3
    }

def get_targets(feature_extractor, images, n_channels=1):
    
    '''
    images shape: [seq_len, seq_batch, h, w]
    '''

    # Iterate over sequences
    img_grids = [get_img_grids(images[:,i,:,:]) for i in range(images.shape[1])]

    tgt_images = torch.tensor(img_grids, device=images.device).permute(1,0,2,3,4)
    # seq_len, seq_bs, h, w = tgt_images.shape
    # tgt_images = tgt_images.view(seq_len * seq_bs, n_channels, h, w)

    eos_token = torch.full([1] + list(tgt_images.shape[1:]), TOKENS['EOS'], device=images.device)
    tgt_images = torch.cat((tgt_images, eos_token))

    img_grids = torch.tensor(img_grids, device=images.device)
    bs, sl, ch, h, w = img_grids.shape
    img_grids = img_grids.view(bs * sl, 1, ch, w, h).squeeze(1) # [batch * seq_len, channels, height, width]
    with torch.no_grad():
        vector_seq = feature_extractor(img_grids.detach().clone()) # [batch * seq_len, emb_size]
    vector_seq = vector_seq.unsqueeze(1).view(bs, sl, -1) # [batch, seq_len, emb_size]

    tgt_list = append_tokens(vector_seq.tolist(), TOKENS['EOS'], TOKENS['SOS'])

    tgt_vectors = torch.tensor(tgt_list)

    return tgt_vectors.permute(1,0,2), tgt_images

def get_img_grids(img_seq):
    '''
    img_ seq = [seq_len, c, h, w] = [4, 16, 16]
    '''
    seq_len, c, h, w = img_seq.shape
    grid = torch.zeros(seq_len, c, h, w, device=img_seq.device)
    img_grids = []
    for i, img in enumerate(img_seq):
        # Place the image in the right quadrant
        grid[i] = img
        # Construct the grid from the batch of images
        img_grid = make_grid(grid.cpu(), nrow=2, padding=0)
        # Take the first channel (Grayscale images, the channels are all the same)
        img_grids.append(img_grid.unsqueeze(0)) # Restore channel and batch dimensions
    return torch.cat(img_grids).tolist()
    # img_grids = pad_sequence(img_grids, 6, torch.zeros(img_grids[0].shape))

# TODO Work only on lists
def append_tokens(sequences, eos_token, sos_token=None):

    out_seq = []
    for seq in sequences:
        # For each sequence, add SOS and EOS token to the extracted vectors
        if type(seq[0]) == int:
            seq.insert(0, sos_token)
            seq.append(eos_token)
        elif type(seq[0]) == list:
            seq.insert(0, [sos_token] * len(seq[0]))
            seq.append([eos_token] * len(seq[0]))
        else:
            raise Exception("Input type unrecognized")                        
        #if sos_token is not None:
        #    sos = torch.full([1, seq.shape[1]], sos_token)
        #    seq = torch.cat((sos, seq.cpu()))
        #eos = torch.full([1, seq.shape[1]], eos_token)
        #seq = torch.cat((seq.cpu(), eos))
        out_seq.append(seq)
    return out_seq

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



def pad_sequence(seq, max_seq_len):
    '''
    seq: [seq_len, embedding_size]
    '''
    padding_masks = []
    if(len(seq) == max_seq_len):
        return seq.tolist()
    elif (len(seq) > max_seq_len):
        raise("Sequence longer than allowed")
    else :
        padding_masks.append([False for _ in range(len(seq))] + [True for _ in range(max_seq_len - len(seq))])
        padded_seq = seq.tolist() + [torch.full([seq.shape[1]], TOKENS['PAD']) for _ in range(max_seq_len - len(seq))]

        return padded_seq