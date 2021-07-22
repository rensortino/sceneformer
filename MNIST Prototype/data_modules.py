from feature_extractor import ResNet18
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import pytorch_lightning as pl
import torch
import os
import torchvision.transforms as T
from data_processing import extract_features, get_img_grid, process_labels, get_embedding_from_vocab, build_vocab, wrong_process_labels
# TODO Generalize with abstract datamodule class
tokens = {
    'src': {
        "sos": 14,
        "eos": 15,
        "pad": 16
    },
    'tgt': {
        "sos": 0.0,
        "eos": 1.0
    }
}

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, args, image_size, backbone, data_dir="data", debug=False):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir
        self.args = args
        self.image_size = image_size
        self.debug = debug
        self.backbone = backbone
        # FIXME Change this
        self.embedding = torch.nn.Embedding(16, 512)

        
        self.vocab = torch.load('vocab.pth')

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def custom_collate_fn(self, batch):
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

        sos_token = img_grids[0].clone().uniform_(0.6,0.9).unsqueeze(0)
        eos_token = img_grids[0].clone().uniform_(0.1,0.4).unsqueeze(0)

        tgt_seq = torch.cat((sos_token, img_grids, eos_token))

        # tgt_seq = get_embedding_from_vocab(src_seq, self.vocab)

        return images, src_seq, tgt_seq

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

        # Create vocab if doesn't exist
        # FIXME Fix embeddings for eos and sos
        if os.path.exists('vocab.pth'):
            self.vocab = torch.load('vocab.pth')
        else:
            self.vocab = build_vocab(self.train_dataloader(), tokens)


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def get_example_batch(self):
        src_seq, tgt_images = next(iter(DataLoader(self.train_dataset, batch_size=self.args.batch_size, drop_last=True, collate_fn=self.custom_collate_fn)))[1:]
        tgt_seq = extract_features(self.backbone, tgt_images)
        return src_seq, tgt_seq

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, persistent_workers=self.args.persistent_workers, drop_last=True, collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        if not self.debug:
            return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        if not self.debug:
            return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.custom_collate_fn)




class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, args, backbone, data_dir="data", debug=False):
        super().__init__()
        self.hparams = args
        self.data_dir = data_dir
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.backbone = backbone
        self.debug = debug
        self.embedding = torch.nn.Embedding(16,512)
        

    def setup(self, stage=None):
        self.transforms = T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.Resize((self.hparams.data.img_h, self.hparams.data.img_w)),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_train = datasets.CIFAR10(self.data_dir, download=True, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(cifar_train, [46000, 4000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(self.data_dir, download=True, train=False, transform=self.transforms)

    def custom_collate_fn(self, batch):
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        images, labels = torch.utils.data._utils.collate.default_collate(batch)
        data = images.reshape(self.hparams.seq_len, self.hparams.seq_bs, self.hparams.data.n_channels, self.hparams.data.img_h, self.hparams.data.img_w)
        labels = labels.reshape(self.hparams.seq_len, self.hparams.seq_bs)

        src_seq = process_labels(labels, tokens['src']['sos'], tokens['src']['eos'])

        # with torch.no_grad():
        #     tgt_features = self.backbone(images.detach())
        # tgt_features = tgt_features.reshape(self.hparams.seq_len, self.hparams.seq_bs, -1)
        # sos_token = torch.full([1] + list(data.shape[1:]), tokens['src']['sos'], device=images.device)
        # eos_token = torch.full([1] + list(data.shape[1:]), tokens['src']['eos'], device=images.device)

        # sos = self.embedding(torch.tensor(tokens['src']['sos'], device=images.device))
        # eos = self.embedding(torch.tensor(tokens['src']['eos'], device=images.device))

        # sos = sos.unsqueeze(0).unsqueeze(0).repeat(1,16,1)
        # eos = eos.unsqueeze(0).unsqueeze(0).repeat(1,16,1)
        for i in range(data.shape[1]):
            img_grid = get_img_grid(data[:,i], self.hparams.data.n_channels)
            img_grid = img_grid.unsqueeze(1)
            if i == 0:
                img_grids = img_grid
            else:
                img_grids = torch.cat((img_grids, img_grid), dim=1)

        sos_token = img_grids[0].clone().uniform_(0.6,0.9).unsqueeze(0)
        eos_token = img_grids[0].clone().uniform_(0.1,0.4).unsqueeze(0)

        tgt_seq = torch.cat((sos_token, img_grids, eos_token))

        # tgt_seq = get_embedding_from_vocab(src_seq, self.vocab)

        return images, src_seq, tgt_seq

    def train_dataloader(self):
        
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.custom_collate_fn
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
                T.Resize((self.hparams.data.img_h, self.hparams.data.img_w)),
            ]
        )
        dataset = datasets.CIFAR10(root=self.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            collate_fn=self.custom_collate_fn
            # pin_memory=True,
        )
        if not self.debug:
            return dataloader

    def test_dataloader(self):
        if not self.debug:
            return self.val_dataloader()

    def get_example_batch(self):
        src_seq, tgt_images = next(iter(DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, drop_last=True, collate_fn=self.custom_collate_fn)))[1:]
        tgt_seq = extract_features(self.backbone, tgt_images)
        return src_seq, tgt_seq
