import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# tensorboard
os.makedirs("../../logs", exist_ok=True)
tbwriter = SummaryWriter(log_dir='../../logs/gan', comment='GAN')  # 使用tensorboard记录中间输出
tbwriter.add_graph(model= generator, input_to_model=torch.randn(size=(1, 100)))
tbwriter.add_graph(model= discriminator, input_to_model=torch.randn(size=(1, 1, 28, 28)))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    epoch_step = len(dataloader)
    with tqdm(total = epoch_step,desc = f'Epoch {epoch+1:3d}/{opt.n_epochs:3d}',postfix=dict,mininterval=0.3) as pbar:
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False) # 得到单位张量作为真实标签 1
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)  # 得到零张量作为假标签 0

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)) # 放入GPU中
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))) 
            # Generate a batch of images 
            gen_imgs = generator(z) # 生成图片

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_out = discriminator(real_imgs)
            real_loss = adversarial_loss(real_out, valid) # 真实图片的损失
            fake_out = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(fake_out, fake) # log(1-D(G(z)))
            d_loss = (real_loss + fake_loss) / 2 # 总的损失 x-logD(x) + z-log(1-D(G(z))) 
            
            d_x = real_out.mean()
            d_g_z1 = fake_out.mean()

            d_loss.backward() # 反向传播
            optimizer_D.step() # 更新生成网络的参数

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            output = discriminator(gen_imgs)
            g_loss = adversarial_loss(output, valid) # 得到假的图片和真实图片的label的loss log(D(G(z)))
            
            d_g_z2 = output.mean()

            g_loss.backward()
            optimizer_G.step()

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            # )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
 
            acc = torch.mean(discriminator(gen_imgs.detach()),axis=0)[0]
            pbar.set_postfix(**{'D Loss' : d_loss.item(),
                                'G Loss' : g_loss.item(),
                                'D(x)': '{:.4f}'.format(d_x.item()),
                                'D(G(z))': '{:.4f} /{:.4f} '.format(d_g_z1, d_g_z2)})
            pbar.update(1)
        img_grid = make_grid(real_imgs)
        tbwriter.add_image(tag='real_image',img_tensor=img_grid,global_step=epoch+1)
        img_grid = make_grid(gen_imgs)
        tbwriter.add_image(tag='fake_image',img_tensor=img_grid,global_step=epoch+1)
        tbwriter.add_scalar('D(x)',d_x.item(),global_step=epoch+1)
        tbwriter.add_scalar('D(G(z))',d_g_z2.item(),global_step=epoch+1)
        tbwriter.add_scalar('dist_loss', d_loss.item(),global_step=epoch+1)
        tbwriter.add_scalar('gene_loss',g_loss.item(),global_step=epoch+1)

torch.save(discriminator, './gan_dist.pth')
torch.save(generator, './gan_gen.pth')