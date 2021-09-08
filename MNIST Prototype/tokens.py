import torch
class TokenContainer:

    def __init__(self):
        self.src_token = {
                "sos": 10,
                "eos": 11,
                "pad": 16
        }
        self.tgt_token = {
                "sos": [0.6, 0.9],
                "eos": [0.1, 0.4]
            }


    def set_img_token(self, shape):
        self.sos_img_token = torch.randn(shape).uniform_(*self.tgt_token['sos']).unsqueeze(0)
        self.eos_img_token = torch.randn(shape).uniform_(*self.tgt_token['eos']).unsqueeze(0)
    
    def set_box_token(self, seq_bs):
        self.sos_box_token = torch.tensor([0,0,0,0]).repeat(1, seq_bs, 1)
        self.eos_box_token = torch.tensor([1,1,0,0]).repeat(1, seq_bs, 1)