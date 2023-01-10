# GAN Step By Step



![logo](https://img-blog.csdnimg.cn/dc199d960b704e0c9331376e069be96e.png#pic_center)

## 心血来潮

**[GSBS][3]**，顾名思义，我希望我自己能够一步一步的学习GAN。GAN 又名 生成对抗网络，是最近几年很热门的一种无监督算法，他能生成出非常逼真的照片，图像甚至视频。GAN是一个图像的全新的领域，从2014的GAN的发展现在，在计算机视觉中扮演这越来越重要的角色，并且到每年都能产出各色各样的东西，GAN的理论和发展都蛮多的。我感觉最近有很多人都在学习GAN，但是国内可能缺少比较多的GAN的理论及其实现，所以我也想着和大家一起学习，并且提供主流框架下 **pytorch,tensorflow,keras** 的一些实现教学。

在一个2016年的研讨会，`杨立昆`描述生成式对抗网络是“`机器学习这二十年来最酷的想法`”。

---

## Step3 DCGAN (Deep Convolutional GAN)

### Deep Convolutional GAN

*Deep Convolutional Generative Adversarial Network*

#### Authors

Alec Radford, Luke Metz, Soumith Chintala

#### Abstract

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

[[Paper\]](https://arxiv.org/abs/1511.06434) [[code\]][4]

![](https://img-blog.csdnimg.cn/img_convert/a57b50b1dd75df94f99618602e5cf139.gif#pic_center)

DCGAN,全称叫**Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**

顾名思义，就是在生成器和判别器特征提取层用卷积神经网络代替了原始GAN中的多层感知机。

为什么这样提出呢，我们从摘要可以看到，因为CNN在supervised learning 领域取得了非常了不起的成就(比如大规模的图片分类，目标检测等等)，但是在unsupervised learning领域却没有特别大的进展。所以作者想弥补CNN在supervised 和 unsupervised之间的隔阂（gap）。作者提出了 **将CNN和GAN相结合** 的DCGAN,并展示了它在unsupervised learning所取得的不俗的成绩。作者通过在大量不同的image datasets上的训练，充分展示了DCGAN的generator(生成器)和discriminator(鉴别器)不论是在物体的组成部分(parts of object)还是场景方面(scenes)都学习到了丰富的层次表达(hierarchy representations)。作者还将学习到的特征应用于新的任务上(比如image classification),结果表明这些特征是非常好的通用图片表达(具有非常好的泛化能力)。

其实DCGAN深度卷积生成对抗网络特别简单，就是将生成网络和对抗网络都改成了卷积网络的形式，下面我们来实现一下

## MNIST数据集实验

### 卷积判别网络

卷积判别网络就是一个一般的卷积网络，结构如下

* 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
* Max Pool 2x2, Stride 2
* 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
* Max Pool 2x2, Stride 2
* Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
* Fully Connected size 1

这里可以看到，我们使用的是一般的卷积网络的判别器，最后输出一个值作为我们的结果

```python
class build_dc_classifier(nn.Module):
    def __init__(self):
        super(build_dc_classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
```



### 卷积生成网络

卷积生成网络需要将一个低维的噪声向量变成一个图片数据，结构如下

* Fully connected of size 1024, ReLU
* BatchNorm
* Fully connected of size 7 x 7 x 128, ReLU
* BatchNorm
* Reshape into Image Tensor
* 64 conv2d$^T$ filters of 4x4, stride 2, padding 1, ReLU
* BatchNorm
* 1 conv2d$^T$ filter of 4x4, stride 2, padding 1, TanH

这一部分的卷积生成网络是利用是卷积+上采样的方法来实现将100维的噪声输入$z$经过多层的卷积和上采样后得到的，在pytorch中，我们就是利用卷积转置，也就是我们的反卷积进行上采样的

```python
class build_dc_generator(nn.Module): 
    def __init__(self, noise_dim=NOISE_DIM):
        super(build_dc_generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7) # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x
```



最后我们进行训练，这一部分的训练方式和GAN是相同的，只是网络变成了卷积神经网络。

```python
def train_dc_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=250, 
                noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = x.cuda() # 真实数据
            logits_real = D_net(real_data) # 判别网络得分
            
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
            g_fake_seed = sample_noise.cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据
            logits_fake = D_net(fake_images) # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # 优化判别网络
            
            # 生成网络
            g_fake_seed = sample_noise.cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data, g_error.data))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
```

### 训练DCGAN

```python
D_DC = build_dc_classifier().cuda()
G_DC = build_dc_generator().cuda()

D_DC_optim = get_optimizer(D_DC)
G_DC_optim = get_optimizer(G_DC)

train_dc_gan(D_DC, G_DC, D_DC_optim, G_DC_optim, discriminator_loss, generator_loss, num_epochs=5)
```

> Iter: 0, D: 1.387, G:0.6381
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/d267c4b9fd7e44d2a083ac3686045274.png)
>
> Iter: 250, D: 0.7821, G:1.807
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/7bad9a7d6bd24ffea8c9ec7f7f15c0dd.png)
>
> ......
>
> Iter: 1500, D: 1.216, G:0.7218
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/38630561cf604a89a5cecb38bc138fd8.png)
>
> Iter: 1750, D: 1.143, G:1.092
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/785438d7d5f74c51a5fd2a0736a692bd.png)

可以看到，通过 DCGANs 能够得到更加清楚的结果，而且也可以更快地收敛

DCGAN也几乎是GAN的入门级模板了，这也不过多阐述了，都看到这里了，如果感兴趣的小伙伴们，觉得还不错的话，可以三连支持一下，点赞+评论+收藏，你们的支持就是我最大的动力啦！😄



[1]: <https://blog.csdn.net/liuxiao214/article/details/74502975>
[2]: https://pytorch.apachecn.org/#/docs/1.7/22

[3]: https://redamancy.blog.csdn.net/article/details/127082815
[4]: https://github.com/Dreaming-future/GAN_Step_By_Step/blob/main/Step3/dcgan/dcgan.py

