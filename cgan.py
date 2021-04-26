import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def _make_block(
    model: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    bias: bool = False,
    batch_norm: bool = True,
    last_block: bool = False,
) -> nn.Sequential:
    if model == "generator":
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Tanh(),
            )
        return gen_block

    elif model == "discriminator":
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                #nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return disc_block

    else: 
        raise Exception(f"No model named: {model}")

class ImageReconstructor(nn.Module):

    def __init__(self, image_size: int, feature_maps: int = 128, image_channels: int = 3) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.criterion = nn.BCELoss()
        self.model = nn.Sequential(
            _make_block("generator", image_size, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            _make_block("generator", feature_maps * 8, feature_maps * 4),
            _make_block("generator", feature_maps * 4, feature_maps * 2),
            _make_block("generator", feature_maps * 2, feature_maps),
            _make_block("generator", feature_maps, image_channels, last_block=True),
        )


    def forward(self, noise, cond) -> torch.Tensor:
        assert noise.shape[0] == cond.shape[0], "Noise and Condition have different shape 0"
        x = torch.cat((noise, cond), 1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.model(x)


class DCGANGenerator(nn.Module):

    def __init__(self, noise_size: int, cond_size: int, feature_maps: int = 128, image_channels: int = 3) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.criterion = nn.BCELoss()
        self.model = nn.Sequential(
            _make_block("generator", noise_size + cond_size, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            _make_block("generator", feature_maps * 8, feature_maps * 4),
            _make_block("generator", feature_maps * 4, feature_maps * 2),
            _make_block("generator", feature_maps * 2, feature_maps),
            _make_block("generator", feature_maps, image_channels, last_block=True),
        )


    def forward(self, noise, cond) -> torch.Tensor:
        assert noise.shape[0] == cond.shape[0], "Noise and Condition have different shape 0"
        x = torch.cat((noise, cond), 1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.model(x)

class DCGANDiscriminator(nn.Module):

    def __init__(self, feature_maps: int = 128, image_channels: int = 3) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.criterion = nn.BCELoss()
        self.model = nn.Sequential(
            _make_block("discriminator", image_channels, feature_maps, batch_norm=False),
            _make_block("discriminator", feature_maps, feature_maps * 2),
            _make_block("discriminator", feature_maps * 2, feature_maps * 4),
            _make_block("discriminator", feature_maps * 4, feature_maps * 8),
            _make_block("discriminator", feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1, 1).squeeze(1)


def generator_loss(self, labels, images):
        # Train with fake
        fake_pred = self._get_fake_pred(labels, images)
        fake_gt = torch.ones_like(fake_pred)
        g_loss = self.gen.criterion(fake_pred, fake_gt)

        return g_loss

def discriminator_loss(self, labels, images):
    # Train with real
    real_pred = self.disc(images)
    real_gt = torch.ones_like(real_pred)
    real_loss = self.disc.criterion(real_pred, real_gt)

    # Train with fake
    fake_pred = self._get_fake_pred(labels, images)
    fake_gt = torch.zeros_like(fake_pred)
    fake_loss = self.disc.criterion(fake_pred, fake_gt)

    d_loss = real_loss + fake_loss

    return d_loss

def _get_fake_pred(self, labels, images, step="generator") -> torch.Tensor:
    #noise = torch.randn(batch_size, self.noise_size, device='cuda')
    noise = torch.cuda.FloatTensor(batch_size, self.noise_size).normal_(0,1)
    cond = self(labels, images)
    fake = self.gen(noise, cond.view(-1, hp['emb_size']))
    if step == "discriminator":
        fake_pred = self.disc(fake.detach())

    return fake_pred

def generator_step(self, labels, images):

    g_loss = self.generator_loss(labels, images)

    self.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
    return g_loss

def discriminator_step(self, labels, images):

    d_loss = self.discriminator_loss(labels, images)

    self.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
    return d_loss