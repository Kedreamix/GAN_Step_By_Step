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

其实要知道自从[2014年Ian Goodfellow提出](https%3A//arxiv.org/abs/1406.2661)以来，GAN就存在着训练困难、生成器和判别器的loss无法指示训练进程、生成样本缺乏多样性等问题。从那时起，很多论文都在尝试解决，但是效果不尽人意，比如最有名的一个改进[DCGAN](https%3A//arxiv.org/abs/1511.06434)依靠的是对判别器和生成器的架构进行实验枚举，最终找到一组比较好的网络架构设置，但是实际上是治标不治本，没有彻底解决问题。而今天的主角Wasserstein GAN（下面简称WGAN）成功地做到了以下爆炸性的几点：

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

 一句话描述 WGAN：**它用一种更好的方式来测量生成数据与真实数据的差距，并用这种方式来拉近他们的距离。**





在讲 WGAN 之前,我还是要来回顾一下原始的 GAN的核心公式:

![image_1ct4sn8kqg8ftika3b1nmj6i1j.png-43kB](https://tva1.sinaimg.cn/large/006tKfTcly1g1lv49i8yzj30w805it98.jpg)

 我们来看**Gennerator和Discriminator的KL散度和JS散度解释**

**Generator中θ的极大似然估计(是求真实分布和生成器分布的KL散度的最小值)**

![image-20190331122346341](https://tva1.sinaimg.cn/large/006tKfTcly1g1lv4a78lfj30gz0cnq3l.jpg)

![image-20190331122401672](https://tva1.sinaimg.cn/large/006tKfTcly1g1lv4aofulj30ln0ggt9q.jpg)

我们可以把原始GAN定义的生成器$loss$等价变换为最小化真实分布$P_r$与生成分布$P_g$之间的$JS$散度。我们越训练判别器，它就越接近最优，最小化生成器的$loss$也就会越近似于最小化$P_r$和$P_g$之间的JS散度。



而在 WGAN 中,我们对这个 KL 散度进行了改进,该文章围绕 Wasserstein 距离对两个分布p(X),q(X)做出如下定义:

![image-20190428121136284](https://tva2.sinaimg.cn/large/006tNc79ly1g2i81gba70j318g04omyp.jpg)

![image-20190331123200062](https://tva4.sinaimg.cn/large/006tKfTcly1g1lva1f13kj30f70onae4.jpg)

(该部分转载于大神苏剑林:<https://kexue.fm/archives/6280>)

### 话锋一转-代码实现

我们从现在看这个可能还是觉得有些困难,包括在实现 WGAN 的过程中涉及到了 loss 函数的反向传播需要满足$Lipschitz连续$等等问题,我也是找了很多的博客等等,直接了当地总结了如何在 GAN的基础上实现 WGAN:

![image-20190331124000273](https://tva1.sinaimg.cn/large/006tKfTcly1g1lvid8bfjj310k0lbqtl.jpg)

上文说过，WGAN与原始GAN相比，只改了四点：

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c(满足利普西斯连续)
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

改动是如此简单，效果却惊人地好，所以有时候，知道如何改动才是最重要的，就像有个故事说的一样，一个工程师解决拧了一下螺丝，就报了50元，最后技术报告写，拧螺丝1元，知道拧哪根，拧多少角度49元，虽然只是简单的，但是这其实后面包含了很多的数学推导

https://mofanpy.com/tutorials/machine-learning/gan/wgan

https://zhuanlan.zhihu.com/p/25071913

https://zhuanlan.zhihu.com/p/361808267

https://zhuanlan.zhihu.com/p/149815753