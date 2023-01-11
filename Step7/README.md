# GAN Step By Step



![logo](https://img-blog.csdnimg.cn/dc199d960b704e0c9331376e069be96e.png#pic_center)

## 心血来潮

**[GSBS][1]**，顾名思义，我希望我自己能够一步一步的学习GAN。GAN 又名 生成对抗网络，是最近几年很热门的一种无监督算法，他能生成出非常逼真的照片，图像甚至视频。GAN是一个图像的全新的领域，从2014的GAN的发展现在，在计算机视觉中扮演这越来越重要的角色，并且到每年都能产出各色各样的东西，GAN的理论和发展都蛮多的。我感觉最近有很多人都在学习GAN，但是国内可能缺少比较多的GAN的理论及其实现，所以我也想着和大家一起学习，并且提供主流框架下 **pytorch,tensorflow,keras** 的一些实现教学。

在一个2016年的研讨会，`杨立昆`描述生成式对抗网络是“`机器学习这二十年来最酷的想法`”。

---

## Step7: WGAN(Wasserstein GAN)

### Wasserstein GAN

*Wasserstein GAN*

#### Authors

Martin Arjovsky, Soumith Chintala, Léon Bottou

#### Abstract

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

[[Paper\]](https://arxiv.org/abs/1701.07875) [[Code\]](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py)

WGAN，即Wasserstein GAN，算是GAN史上一个比较重要的理论突破结果，它将GAN中两个概率分布的度量从f散度改为了Wasserstein距离，从而使得WGAN的训练过程更加稳定，而且生成质量通常也更好。Wasserstein距离跟最优传输相关，属于Integral Probability Metric（IPM）的一种，这类概率度量通常有着更优良的理论性质，因此WGAN的出现也吸引了很多人从最优传输和IPMs的角度来理解和研究GAN模型。



## GAN面临的难题

其实要知道自从[2014年Ian Goodfellow提出](https%3A//arxiv.org/abs/1406.2661)以来，GAN就存在着训练困难、生成器和判别器的loss无法指示训练进程、生成样本缺乏多样性等问题。从那时起，很多论文都在尝试解决，但是效果不尽人意，比如最有名的一个改进[DCGAN](https%3A//arxiv.org/abs/1511.06434)依靠的是对判别器和生成器的架构进行实验枚举，最终找到一组比较好的网络架构设置，但是实际上是治标不治本，没有彻底解决问题。

训练 GAN 真的是一件掉头发的事情，有人说，因为训练GAN时，生成器和判别器是一种对抗的状态， 任何一方太强，都会碾压对方，对抗的平衡被打破，训练就会失败。以至于需要自己摸索出一套 magic tricks 才能偶然训练好。

训练GAN最常出现的问题有以下几点

- 损失震荡，训练的时候存在不稳定性，有时候很难收敛
- 模式收缩(Mode Collapse)，生成器(generator)趋向于重复生成同样的样本，而不是生成多样化的样本。
- 不提供信息的损失函数，有时候生成器的损失与图像质量没有相关性。
- 超参数。需要进过反复实验才能找到一个好的超参数



## WGAN：解决GAN面临的难题

而今天的主角Wasserstein GAN（下面简称WGAN）成功地做到了以下爆炸性的几点：

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode（模型收缩）的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

 一句话描述 WGAN：**它用一种更好的方式来测量生成数据与真实数据的差距，并用这种方式来拉近他们的距离。**



## WGAN与GAN

在讲 WGAN 之前,我还是要来回顾一下原始的 GAN的核心公式:

![在这里插入图片描述](https://img-blog.csdnimg.cn/3db08c50f25245c68760b305cb316fda.png#pic_center)

在用上述的Loss的时候，我们可以把原始GAN定义的生成器$loss$等价变换为最小化真实分布$P_r$与生成分布$P_g$之间的$JS$散度。我们越训练判别器，它就越接近最优，最小化生成器的$loss$也就会越近似于最小化$P_r$和$P_g$之间的JS散度。

WGAN的论文中就用数学推导分析了一下JS散度的一些缺陷， 分析过程太长了，可以看看知乎的一篇文章[令人拍案叫绝的Wasserstein GAN][2]，两个数据分布的 JS 散度越小，则两个数据分布越近，生成数据和真实数据也越像，GAN就像通过拉近 JS 散度来优化模型，这没什么问题。 但是 JS 散度本身有一个缺陷。当两个分布没有重叠部分，或重叠部分比较小的话，它的可以用来更新模型的梯度就是0，或忽略不计。 这种梯度为0的情况很常发生在当你的判别器能力比生成器要强很多的时候（对抗平衡被打破）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f85c8f51e68640408613b25db12a5089.png#pic_center)

这样就会得到一个结论，**如果你不小心（非常容易不小心）把你的判别器训练得比生成器好（非常容易训练得比生成器好），那么就很容易导致更新网络的梯度很小，学不动，学废了。**

除了一些梯度上的问题，还有就是mode collapse问题，生成固定的模式，生成多样性非常少。 比如下图中，上半部分是我们希望的多样性，但是模型可能只生成下半部分的6，这就是 collapse 了。判别器因为GAN原始目标函数目标不一致， 导致模型更愿意放弃多样性，生成一种模式，这样它认为更安全一点，这样生成虽然可能损失下降了，但是这些都不是我们想要的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a3592c088e4a4498be2ea054b3f429ba.png#pic_center)



WGAN 就尝试使用不同的 loss 来解决上述问题。如果说 JS 散度对GAN的训练不友善， 那 WGAN 就换掉它，取而代之的是 Wasserstein 距离，也称为 Earth-Mover (EM) 推土距离。

![在这里插入图片描述](https://img-blog.csdnimg.cn/525bc24969d24d3abcd33fb1d5f6aea1.png#pic_center)

用一张图来展示这个“推土”过程：

[![推土机距离图示。左边p(x)每处的沙土被分为若干部分，然后运输到右端q(x)的同色的位置（或者不动）](https://kexue.fm/usr/uploads/2019/01/417385384.png#pic_center)](https://kexue.fm/usr/uploads/2019/01/417385384.png#pic_center)

<center>推土机距离图示。左边p(x)每处的沙土被分为若干部分，然后运输到右端q(x)的同色的位置（或者不动）



在之前我们说过，GAN 学着拉近生成数据和真实数据的 **数据分布**，这个 Earth-Mover 也是要拉近分布，它用的方法是把两个数据的分布变成土堆。 把生成数据的土堆，推成真实数据的土堆。怎么推？就选消耗的工作量最少的推法。 上图就是将高低不同的土堆量给填充补全成目标土堆的样子。而最优的推土搬运成本就是 Wasserstein 距离。 所以 Wasserstein 距离也就被称为 Earth-Mover distance 推土机距离。

![在这里插入图片描述](https://img-blog.csdnimg.cn/d5f1dd30c2b54a7dba8193eb1a126f12.png#pic_center)

我们来看这个损失，实际上就是会最大化真实图像的预测与生成图像的预测之间的差异，真实图像的得分会更高，所以这个损失也是我们的判别器的损失

为了训练WGAN的生成器，**discriminator 本身要最大化这个距离， generator 的任务就变成了最小化 EM 距离**



### WGAN的改进

在WGAN的论文里，推了一堆公式定理，最后给了改进的算法流程。

**WGAN与原始GAN相比，只改了四点：**

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c(满足利普希茨连续)
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

算法截图如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/b13ef5e2fc61433faabb00917f697a6c.png#pic_center)

## MNIST数据集实验

由于注意原始GAN的判别器做的是真假二分类任务，所以最后一层是sigmoid，但是现在WGAN中的判别器fw做的是近似拟合Wasserstein距离，属于回归任务，所以要把最后一层的sigmoid拿掉。

除此之外，训练完判别器，在截断它的参数， 让它被限制起来，也就是施加利普希茨约束，限制在一个范围之内



### 生成网络

生成器跟原来的是差不多，这里面用的还是线性层，为什么不用卷积层呢，实际上在论文作者在做实验的时候，WGAN用卷积神经网络的时候，DCGAN真的是泣不成声，出现了mode collapse。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b05e8a86b777498a9e080379e06ef488.png)

```python
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
        img = img.view(img.shape[0], *img_shape)
        return img

```



### 判别网络

判别网络的话，也是类似的，只不过后续是不需要加sigmoid层的，所以是直接输出线性层的结果。

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
```





### 训练过程

由于在上述，我们去掉了判别器的最后一层的S激活，这样预测的结果就不一定会落在[0,1]的范围内，现在它可以是[-∞，+∞]范围内的任何数字，所以WGAN的判别器，我们也通常称为评论者(critic)。

在使用Wasserstein损失函数的时候，我们应该训练到评论者收敛为止，目的是为了确保生成器梯度更新的准确性，这是与标准的GAN是不同的，在标准的GAN中有一点很重要：**不要让判别器过强，以免梯度消失**。这样，我们就可以利用WGAN，在生成器的两次更新之间训练几次评论者，以确保它接近收敛。最常见的比例是评论者更新5次，生成器更新一次。

```python
for epoch in range(opt.n_epochs):
    epoch_step = len(dataloader)
    with tqdm(total = epoch_step,desc = f'Epoch {epoch+1:3d}/{opt.n_epochs:3d}',postfix=dict,mininterval=0.3) as pbar:

        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator 训练评论者
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            # clip weights, 施加利普希茨约束
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            # 最常见的比例是评论者更新5次，生成器更新一次
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

```

其实这里还有一个 **clip 网络参数**的操作，权重裁剪，说是因为判别器需要满足 1-Lipschitz（利普希茨连续），将其的梯度更新约束起来， **限制判别器的能力，让判别器别那么浪，** 而 clip 是一种简单粗暴的满足方法。

但是后续WGAN也受到了批评，因为由于clip 了权重，所以其学习能力也大大降低了，实际上，即使在WGAN原来的论文中，作者也写道："裁剪权重不是施加利普希茨约束的一个好方法"，后续也有一些更好的方法如WGAN-gp，去提高WGAN学习复杂特征的能力，

改动是如此简单，效果却惊人地好，所以有时候，知道如何改动才是最重要的，就像有个故事说的一样，一个工程师解决拧了一下螺丝，就报了50元，最后技术报告写，拧螺丝1元，知道拧哪根，拧多少角度49元，虽然只是简单的，但是这其实后面包含了很多的数学推导。都看到这里了，如果感兴趣的小伙伴们，觉得还不错的话，可以三连支持一下，点赞+评论+收藏，你们的支持就是我最大的动力啦！😄



[1]: https://mofanpy.com/tutorials/machine-learning/gan/wgan
[2]: https://zhuanlan.zhihu.com/p/25071913
[3]: https://zhuanlan.zhihu.com/p/361808267
[4]: https://zhuanlan.zhihu.com/p/149815753