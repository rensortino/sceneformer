import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np

def get_embedding_from_vocab(src, vocab):
    embedding = torch.zeros(1,512)
    for s in src.reshape(-1):
        if (embedding == torch.zeros(1,512)).all():
            embedding = vocab[s.item()].unsqueeze(0)
        else:
            embedding = torch.cat((embedding, vocab[s.item()].unsqueeze(0)))

    return embedding.reshape(*src.shape, -1).cuda()


def get_target_images(images, TOKENS):
    
    '''
    images shape: [seq_len, seq_batch, h, w]
    '''

    # Iterate over sequences
    seq_len, bs, c, h, w = images.shape
    bboxes = []
    for i in range(seq_len):
        bbox = [w * (i % 2), h * (i  > 1), w, h] #[xywh]
        bboxes.append(bbox)

    # Repeat boxes over batch dimension and swap this dimension with the seq_len
    # TODO Parametrize
    norm_boxes = torch.tensor(bboxes, device=images.device) / (h*2)
    tgt_boxes = norm_boxes.unsqueeze(0).repeat(bs,1,1)

    

    sos_token = torch.full([1] + list(images.shape[1:]), TOKENS['SOS'], device=images.device)
    eos_token = torch.full([1] + list(images.shape[1:]), TOKENS['EOS'], device=images.device)
    tgt_images = torch.cat((sos_token, images, eos_token))

    return tgt_images


def process_labels(labels, sos_token, eos_token):
    '''
    labels: [bs, seq_len]
    '''
    in_list = append_tokens(labels.tolist(), eos_token, sos_token)
    in_seq = torch.tensor(in_list, device=labels.device)
    # Convert in transformer dimension order
    return in_seq.t()


def get_succession(labels):
    '''
    labels: [bs, seq_len]
    '''
    successions = []
    for seq in labels:
        succession = []
        for i, num in enumerate(seq):
            if i == 0:
                continue
            elif i == 1:
                succession.append(num.item() + seq[i -1].item())
            else:
                succession.append(num.item() + succession[i -2])
        successions.append(succession)
    successions = torch.tensor(successions, device=labels.device)
    # Convert in transformer dimension order
    return successions

def get_bboxes(img_seq):
    '''
    img_ seq = [seq_len, c, h, w] = [4, 3, 16, 16]
    '''
    seq_len, c, h, w = img_seq.shape
    bboxes = []
    for i in range(seq_len):
        # bbox is computed fixed, it's arranged in a square grid 
        bbox = [w * (i % 2), h * (i  > 1), w, h] #[xywh]
        # TODO Parametrize
        rescaled = map(lambda x: x / 32, bbox)
        bboxes.append(list(rescaled))

    return bboxes
    # img_grids = pad_sequence(img_grids, 6, torch.zeros(img_grids[0].shape))

def get_img_grids(img_seq, n_channels):
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

        if n_channels == 1:
            # Take the first channel (Grayscale images, the channels are all the same)
            img_grids.append(img_grid[0].unsqueeze(0).unsqueeze(0))
        elif n_channels == 3:
            img_grids.append(img_grid.unsqueeze(0)) # Restore channel and batch dimensions
        else:
            raise Exception("Incorrect number of channels")

    return torch.cat(img_grids).tolist()
    # img_grids = pad_sequence(img_grids, 6, torch.zeros(img_grids[0].shape))

def get_one_hot(labels):
    return F.one_hot(labels, 512).float()
    
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
        out_seq.append(seq)
    return out_seq


###########
# Padding #
###########
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


def build_vocab(feature_extractor, dataloader, tokens, vocab_size=16):
    vocab_list = [torch.zeros(1,512) for i in range(vocab_size)]
    for images, labels in dataloader:

        with torch.no_grad():
            feature_vecs = feature_extractor(images.cuda(), True).cpu()
        
        for i, label in enumerate(labels):
            vocab_list[label.item()] = torch.cat((vocab_list[label.item()], feature_vecs[i].unsqueeze(0)))

    with torch.no_grad():
        sos_token = torch.zeros(32,1,32,32)
        sos_vec = feature_extractor(sos_token.cuda(), True).cpu()
        eos_token = torch.ones(32,1,32,32)
        eos_vec = feature_extractor(eos_token.cuda(), True).cpu()
    vocab = {i: vocab_list[i][1:].mean(dim=0) for i in range(vocab_size)}
    vocab[tokens['src']['SOS']] = sos_vec.mean(dim=0)
    vocab[tokens['src']['EOS']] = eos_vec.mean(dim=0)
    torch.save(vocab, 'vocab.pth')
    return vocab

