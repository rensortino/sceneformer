from feature_extractor import ResNet18
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from data_processing import extract_features, get_bboxes, get_img_grid, process_labels, get_embedding_from_vocab, build_vocab, wrong_process_labels
tokens = {
    'src': {
        "sos": 14,
        "eos": 15,
        "pad": 16
    },
    'tgt': {
        "sos": [0.6, 0.9],
        "eos": [0.1, 0.4]
    }
}

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, args, image_size, backbone, data_dir="data", debug=False):
        super(GenericDataModule, self).__init__()
        self.data_dir = data_dir
        self.args = args
        self.image_size = image_size
        self.debug = debug
        self.backbone = backbone

    def grid_collate_fn(self, batch):
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        images, labels = torch.utils.data._utils.collate.default_collate(batch)
        data = images.reshape(self.args.seq_len, self.args.seq_bs, self.args.data.n_channels, self.args.data.img_h, self.args.data.img_w)
        labels = labels.reshape(self.args.seq_len, self.args.seq_bs)

        src_seq = process_labels(labels, tokens['src']['sos'], tokens['src']['eos'])

        for i in range(data.shape[1]):
            img_grid = get_img_grid(data[:,i], self.args.data.n_channels)
            img_grid = img_grid.unsqueeze(1)
            if i == 0:
                img_grids = img_grid
            else:
                img_grids = torch.cat((img_grids, img_grid), dim=1)

        sos_token = img_grids[0].clone().uniform_(*tokens['tgt']['sos']).unsqueeze(0)
        eos_token = img_grids[0].clone().uniform_(*tokens['tgt']['eos']).unsqueeze(0)

        tgt_seq = torch.cat((sos_token, img_grids, eos_token))

        return images, src_seq, tgt_seq

    def bbox_collate_fn(self, batch):
        images, labels = torch.utils.data._utils.collate.default_collate(batch)
        data = images.reshape(self.args.seq_len, self.args.seq_bs, self.args.data.n_channels, self.args.data.img_h, self.args.data.img_w)
        labels = labels.reshape(self.args.seq_len, self.args.seq_bs)

        src_seq = process_labels(labels, tokens['src']['sos'], tokens['src']['eos'])

        for i in range(data.shape[1]):
            img_w_boxes = get_bboxes(data[:,i])
            img_w_boxes = img_w_boxes.unsqueeze(1)
            if i == 0:
                boxes = img_w_boxes
            else:
                boxes = torch.cat((boxes, img_w_boxes), dim=1)

        sos_img_token = data[0].clone().uniform_(*tokens['tgt']['sos']).unsqueeze(0)
        eos_img_token = data[0].clone().uniform_(*tokens['tgt']['eos']).unsqueeze(0)

        sos_box_token = torch.tensor([0,0,0,0]).repeat(1,self.args.seq_bs,1)
        eos_box_token = torch.tensor([1,1,0,0]).repeat(1,self.args.seq_bs,1)

        tgt_seq = torch.cat((sos_img_token, data, eos_img_token))
        tgt_boxes = torch.cat((sos_box_token, boxes, eos_box_token))

        return images, src_seq, tgt_seq, tgt_boxes

    def get_example_batch(self):
        src_seq, tgt_images = next(iter(DataLoader(self.train_dataset, batch_size=self.args.batch_size, drop_last=True, collate_fn=self.grid_collate_fn)))[1:]
        tgt_seq = extract_features(self.backbone, tgt_images)
        return src_seq, tgt_seq

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, persistent_workers=self.args.persistent_workers, drop_last=True, collate_fn=self.bbox_collate_fn)

    def val_dataloader(self):
        if not self.debug:
            return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.bbox_collate_fn)

    def test_dataloader(self):
        if not self.debug:
            return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.bbox_collate_fn)


class MNISTDataModule(GenericDataModule):

    def __init__(self, args, image_size, backbone, data_dir="data", debug=False):
        super(MNISTDataModule, self).__init__(args, image_size, backbone, data_dir, debug)

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ 
            T.Resize((self.image_size)),
            T.ToTensor()
        ])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)



class CIFAR10DataModule(GenericDataModule):
    def __init__(self, args, backbone, data_dir="data", debug=False):
        super(CIFAR10DataModule, self).__init__(args, 32, backbone, data_dir, debug)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)


    def setup(self, stage=None):
        self.train_transforms = T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.Resize((self.args.data.img_h, self.args.data.img_w)),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transforms = T.Compose(
            [
                T.Resize((self.args.data.img_h, self.args.data.img_w)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_train = datasets.CIFAR10(self.data_dir, download=True, train=True, transform=self.train_transforms)
            self.train_dataset, self.val_dataset = random_split(cifar_train, [46000, 4000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(self.data_dir, download=True, train=False, transform=self.test_transforms)
