import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
os.makedirs("images", exist_ok=True)

# 命令行参数，通过命令行修改默认的参数即可运行
# python acgan.py n_epcohs 10 --batch_size=32 即是修改迭代10次，bs改为32
# python acgan.py -h 可以查看参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--cuda",action='store_false', help="use GPU?")
opt = parser.parse_args()
print(opt)

# 是否使用cuda
cuda = opt.cuda if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# ACGAN 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 将label映射成于z一样的维度
        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        # 利用卷积网络
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # 相乘起来这样便使得noise的输入是建立在label作为条件的基础上
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        # 判别网络也利用卷积网路
        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        # 辅助分类层
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        # 判别器的输出有两个,一个是判断真假的validity,一个图片对应的label信息
        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# tensorboard 可视化结果
os.makedirs("../../logs", exist_ok=True)
tbwriter = SummaryWriter(log_dir='../../logs/acgan', comment='ACGAN')  # 使用tensorboard记录中间输出
labels = torch.randint(10,size=(1,))
tbwriter.add_graph(model= generator, input_to_model=[torch.randn(size=(1, opt.latent_dim)),labels])
tbwriter.add_graph(model= discriminator, input_to_model=torch.randn(size=(1, opt.channels, opt.img_size, opt.img_size)))

# 如果使用GPU，则放在GPU上运行
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader 构建Dataloader，使用mnist数据集 （也可以使用自己的数据集）
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

# Optimizers 定义使用的优化器Adam，并且定义使用的参数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 这里定义类别，之后可以进行转化
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 得到一个网格图片，并且保存在images文件下，方便可视化
# 并且利用TensorBoard可视化
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels) # 生成结果
    img_grid = make_grid(gen_imgs, nrow = n_row) # 得到网格图像
    tbwriter.add_image(tag='fake_image',img_tensor=img_grid,global_step=batches_done)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    epoch_step = len(dataloader)
    with tqdm(total = epoch_step,desc = f'Epoch {epoch+1:3d}/{opt.n_epochs:3d}',postfix=dict,mininterval=0.3) as pbar:
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            # 得到对抗的ground truths，valid全为1， fake全为0
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input 转化数据格式
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # Sample noise and labels as generator input
            # z为设置的噪声，用正态分布来随机生成，均值为0，方差为1
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            # 生成一个batch的图片，并且ACGAN加入生成的labels
            gen_imgs = generator(z, gen_labels)

            # ---------------------
            #  Train Discriminator 训练判别器
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            # 不仅需要判别类别正确，并且也需要正确判定真假
            d_loss = (d_real_loss + d_fake_loss) / 2  # -(LS + LC)

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward() # 进行反向传播
            optimizer_D.step() # 更新参数

            # -----------------
            #  Train Generator 训练生成器
            # -----------------

            optimizer_G.zero_grad()

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            # 使得生成图片的loss最小，并且生成图片的label与指定的loss也是最小的
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) 
            
            g_loss.backward() # 进行反向传播
            optimizer_G.step() # 更新参数

            # 每sample_interval个batchs后保存一次images，图片放在images文件夹下
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)
            
            # 利用tqdm实时的得到我们的损失的结果
            pbar.set_postfix(**{'D Loss' : d_loss.item(),
                                'G Loss' : g_loss.item(),
                                'ACGAM Acc': '{:.4f}%'.format(100*d_acc.item())})
            pbar.update(1)
        # 进行tensorboard可视化， 得到真实的图片
        # 以及记录各个值，这样有助于我们判断损失的变化
        img_grid = make_grid(real_imgs)
        tbwriter.add_image(tag='real_image',img_tensor=img_grid,global_step=epoch+1)
        # img_grid = make_grid(gen_imgs)
        # tbwriter.add_image(tag='fake_image',img_tensor=img_grid,global_step=epoch+1)
        tbwriter.add_scalar('ACGAN_acc',d_acc.item(), global_step=epoch+1)
        tbwriter.add_scalar('dist_loss', d_loss.item(),global_step=epoch+1)
        tbwriter.add_scalar('gene_loss',g_loss.item(),global_step=epoch+1)
# 最后保存迭代后的模型
torch.save(discriminator, './acgan_dist.pth')
torch.save(generator, './acgan_gen.pth')
