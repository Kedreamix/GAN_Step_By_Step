# GAN Step By Step



![logo](https://img-blog.csdnimg.cn/dc199d960b704e0c9331376e069be96e.png#pic_center)

## 心血来潮

**GSBS**，顾名思义，我希望我自己能够一步一步的学习GAN。GAN 又名 生成对抗网络，是最近几年很热门的一种无监督算法，他能生成出非常逼真的照片，图像甚至视频。GAN是一个图像的全新的领域，从2014的GAN的发展现在，在计算机视觉中扮演这越来越重要的角色，并且到每年都能产出各色各样的东西，GAN的理论和发展都蛮多的。我感觉最近有很多人都在学习GAN，但是国内可能缺少比较多的GAN的理论及其实现，所以我也想着和大家一起学习，并且提供主流框架下 **pytorch,tensorflow,keras** 的一些实现教学。

在一个2016年的研讨会，`杨立昆`描述生成式对抗网络是“`机器学习这二十年来最酷的想法`”。

---



## Step2 GAN的详细介绍及其应用

### GAN基本框架

上一次已经介绍了一下GAN的基本框架和基本公式，如图所示

![](https://img-blog.csdnimg.cn/img_convert/87ade1b79a03c3a165b5957613abeeba.png)

----



### 10大典型的GAN算法

GAN 算法有数百种之多，大家对于 GAN 的研究呈指数级的上涨，目前每个月都有数百篇论坛是关于对抗网络的。

下图是2014-2018每个月关于 GAN 的论文发表数量：

![关于GANs的论文呈指数级增长](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-07-16-paper.png)

如果你对 GANs 算法感兴趣，可以在 「[GANs动物园](https://github.com/hindupuravinash/the-gan-zoo)」里查看几乎所有的算法。我们为大家从众多算法中挑选了10个比较有代表性的算法，技术人员可以看看他的论文和代码。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f4c3995b62bf48118af9291bb68c9210.png)

<center>GAN MAPs

接下来，我也会一一学习这些GAN的知识，一起遍历这个GAN的地图

| 算法     | 论文                                           | 代码                                                         |
| -------- | ---------------------------------------------- | ------------------------------------------------------------ |
| GAN      | [论文地址](https://arxiv.org/abs/1406.2661)    | [代码地址](https://github.com/goodfeli/adversarial)          |
| DCGAN    | [论文地址](https://arxiv.org/abs/1511.06434)   | [代码地址](https://github.com/floydhub/dcgan)                |
| CGAN     | [论文地址](https://arxiv.org/abs/1411.1784)    | [代码地址](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras) |
| CycleGAN | [论文地址](https://arxiv.org/abs/1703.10593v6) | [代码地址](https://github.com/junyanz/CycleGAN)              |
| CoGAN    | [论文地址](https://arxiv.org/abs/1606.07536)   | [代码地址](https://github.com/mingyuliutw/CoGAN)             |
| ProGAN   | [论文地址](https://arxiv.org/abs/1710.10196)   | [代码地址](https://github.com/tkarras/progressive_growing_of_gans) |
| WGAN     | [论文地址](https://arxiv.org/abs/1701.07875v3) | [代码地址](https://github.com/eriklindernoren/Keras-GAN)     |
| SAGAN    | [论文地址](https://arxiv.org/abs/1805.08318v1) | [代码地址](https://github.com/heykeetae/Self-Attention-GAN)  |
| BigGAN   | [论文地址](https://arxiv.org/abs/1809.11096v2) | [代码地址](https://github.com/huggingface/pytorch-pretrained-BigGAN) |

上面内容整理自《[Generative Adversarial Networks – The Story So Far](https://blog.floydhub.com/gans-story-so-far/)》原文中对算法有一些粗略的说明，感兴趣的可以看看。



### GAN的优缺点

**3个优势**

1. 能更好建模数据分布（图像更锐利、清晰）
2. 理论上，GANs 能训练任何一种生成器网络。其他的框架需要生成器网络有一些特定的函数形式，比如输出层是高斯的。
3. 无需利用马尔科夫链反复采样，无需在学习过程中进行推断，没有复杂的变分下界，避开近似计算棘手的概率的难题。

**2个缺陷**

1. 难训练，不稳定。生成器和判别器之间需要很好的同步，但是在实际训练中很容易D收敛，G发散。D/G 的训练需要精心的设计。
2. 模式缺失（Mode Collapse）问题。GANs的学习过程可能出现模式缺失，生成器开始退化，总是生成同样的样本点，无法继续学习。





## Application

----

### **[姿势引导人形像生成](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1705.09368.pdf)**

通过姿势的附加输入，我们可以将图像转换为不同的姿势。例如，右上角图像是基础姿势，右下角是生成的图像。

![img](https://img-blog.csdnimg.cn/img_convert/bd4f3c0f0d6bf5aeb3f6dd56bb3edc02.png)


---

下面的优化结果列是生成的图像。



![img](https://img-blog.csdnimg.cn/img_convert/9f27e9d4f8b397bcdcb61240b185aac8.png)


----

该设计由二级图像发生器和鉴频器组成。生成器使用元数据（姿势）和原始图像重建图像。判别器使用原始图像作为[CGAN](https://link.zhihu.com/?target=https%3A//medium.com/%40jonathan_hui/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d)设计标签输入的一部分。



![img](https://img-blog.csdnimg.cn/img_convert/73628d80b26c50cc01997af5a13416c0.png)

----

### **CycleGAN**

跨域名转让将很可能成为第一批商业应用。GANs将图像从一个领域（如真实的风景）转换为另一个领域（莫奈绘画或梵高）。



![img](https://img-blog.csdnimg.cn/img_convert/7d1330c866a65a2fa0c8e4adf9233150.png)

---

例如，它可以在斑马和马之间转换图片。



![img](https://img-blog.csdnimg.cn/img_convert/57f78f4578386714815cd5b7f8cb662e.png)

----

Cyclegan构建了两个网络G和F来构建从一个域到另一个域以及反向的图像。它使用判别器d来批评生成的图像有多好。例如，G将真实图像转换为梵高风格的绘画，并且DY用于区分图像是真实的还是生成的。

域A到域B：------------------------>


![bg right width:600px](https://img-blog.csdnimg.cn/img_convert/5afa83b6dc010f35c9e919a8c32eedc8.png)



我们在反向域B域A中重复该过程：

![width:600px](https://img-blog.csdnimg.cn/img_convert/8c038e991f1f1bba6282cf610dda9c0b.png)

----

**PixelDTGAN**

根据名人图片推荐商品已经成为时尚博客和电子商务的热门话题。Pixeldtgan的作用就是从图像中创建服装图像和样式。



![img](https://img-blog.csdnimg.cn/img_convert/07e43babf08d4f9cd9d33f2daf913c84.png)

---

![bg  width:600px](https://img-blog.csdnimg.cn/img_convert/328b761064e33428ad50da4ea7f5b024.png)


![bg width:600px](https://img-blog.csdnimg.cn/img_convert/4effa1f26a3786178631ace651e5c5f3.png)

---

**超分辨率**

从低分辨率创建超分辨率图像。这是GAN显示出非常令人印象深刻的结果，也是具有直接商业可能性的一个领域。



![img](https://img-blog.csdnimg.cn/img_convert/91c46ad04f6a43164e5cee793e658d7a.png)


---

与许多GAN的设计类似，它是由多层卷积层、批标准化、高级relu和跳跃连接组成。

![img](https://img-blog.csdnimg.cn/img_convert/870b79f0a3a91f3db09af6d8a6b2aab7.png)

---

**PGGAN**

Progressive GAN可能是第一个展示商业化图像质量的GAN之一。以下是由GAN创建的1024×1024名人形象。



![img](https://img-blog.csdnimg.cn/img_convert/16ba10415bbfcf34bd375997a786d46d.png)

----

它采用分而治之的策略，使训练更加可行。卷积层的一次又一次训练构建出2倍分辨率的图像。


![img](https://img-blog.csdnimg.cn/img_convert/a85061635f5afd504958a36559d6ae38.png)

----

在9个阶段中，生成1024×1024图像。

![img](https://img-blog.csdnimg.cn/img_convert/0765799d0587a696de0e8afb62716402.png)

---

### **高分辨率图像合成**

需要注意的是这并非图像分割，而是从语义图上生成图像。由于采集样本非常昂贵，我们采用生成的数据来补充培训数据集，以降低开发成本。在训练自动驾驶汽车时可以自动生成视频，而不是看到它们在附近巡航，这就为我们的生活带来了便捷。

网络设计:

![img](https://img-blog.csdnimg.cn/img_convert/b741341dc7dce64a2de26303f7c03423.png)



![bg right:30% width:600px](https://img-blog.csdnimg.cn/img_convert/5a7462bf9d277ebce60bc6ef2bdd7704.png)

---

## **文本到图像（[StackGAN](https://link.zhihu.com/?target=https%3A//github.com/hanzhanggit/StackGAN)）**

文本到图像是域转移GAN的早期应用之一。比如，我们输入一个句子就可以生成多个符合描述的图像。


![bg vertical width:600px](https://img-blog.csdnimg.cn/img_convert/1b4ac817d241f0c4adf701d7dbd6eaff.png)

![bg right:60% width:600px](https://img-blog.csdnimg.cn/img_convert/8da4cc950571b9128f95948b214e9095.png)

---

### **文本到图像合成**

另一个比较通用的实现：

![img](https://img-blog.csdnimg.cn/img_convert/80974f51f5fecf1e454bd127b7c28942.png)

---

### **人脸合成**

不同姿态下的合成面：使用单个输入图像，我们可以在不同的视角下创建面。例如，我们可以使用它来转换更容易进行人脸识别图像。



![ ](https://img-blog.csdnimg.cn/img_convert/acc56d5569aa8190d07cc8f427bfca86.png)

![bg right width:700px](https://img-blog.csdnimg.cn/img_convert/e1b6621920bfee2cd5d7313d841858ee.png)

----

### **图像修复**

几十年前，修复图像一直是一个重要的课题。gan就可以用于修复图像并用创建的“内容”填充缺失的部分。



![img](https://img-blog.csdnimg.cn/img_convert/bcf808db1ecc75a0798054f43aed7da7.png)

----

### **学习联合分配**

用面部字符P（金发，女性，微笑，戴眼镜），P（棕色，男性，微笑，没有眼镜）等不同组合创建GAN是很不现实的。维数的诅咒使得GAN的数量呈指数增长。但我们可以学习单个数据分布并将它们组合以形成不同的分布，即不同的属性组合。


![width:800px](https://img-blog.csdnimg.cn/img_convert/acb15cfe34393a22781b30664b1a4e82.png)

![bg right width:600px](https://img-blog.csdnimg.cn/img_convert/1d721426e4fa790c9d123c20afcd9a96.png)

----

### **DiscoGAN**

DiscoGAN提供了匹配的风格：许多潜在的应用程序。DiscoGAN在没有标签或配对的情况下学习跨域关系。例如，它成功地将样式（或图案）从一个域（手提包）传输到另一个域（鞋子）。



![img](https://img-blog.csdnimg.cn/img_convert/59beba704ee4cb638caf2cd60cbf69d7.png)


DiscoGAN和cyclegan在网络设计中非常相似。

----


![img](https://img-blog.csdnimg.cn/img_convert/cee0755a4fb41a65e8e18fbfecdeadac.png)


----

### **[Pix2Pix](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.07004.pdf)**

PIX2PIx是一种图像到图像的翻译，在跨域Gan的论文中经常被引用。例如，它可以将卫星图像转换为地图（图片左下角）。

![img](https://img-blog.csdnimg.cn/img_convert/0e0f7f84a07e65f0235435946bbb7e74.png)


----

### **[DTN](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1611.02200.pdf)**

从图片中创建表情符号。



![bg right:40% width:500px](https://img-blog.csdnimg.cn/img_convert/98d6eaacac60d3e8a1039350b668e773.png)

![img](https://img-blog.csdnimg.cn/img_convert/eab55c0309c5cab90155b0a3d41472ea.png)

----

### **纹理合成**

![img](https://img-blog.csdnimg.cn/img_convert/d0c1b42a285515ea7d6be46d1058a265.png)

----

### **图像编辑 ([IcGAN](https://link.zhihu.com/?target=https%3A//github.com/Guim3/IcGAN))**

重建或编辑具有特定属性的图像。



![width:500px](https://img-blog.csdnimg.cn/img_convert/c2ea75b32951dce181dbec077f9f964f.png)

![img](https://img-blog.csdnimg.cn/img_convert/82d18f044b49646f084b5bf21d2872a3.png)

----

### **人脸老化([Age-cGAN](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1702.01983.pdf))**



![width:500px](https://img-blog.csdnimg.cn/img_convert/cb96b232d736b336c690ed5dc7e6900c.png)

![img](https://img-blog.csdnimg.cn/img_convert/8044a3ceebb748d10bce4bad0417980b.png)

---


### **创建动画角色**

众所周知，游戏开发和动画制作成本很高，并且雇佣了许多制作艺术家来完成相对常规的任务。但通过GAN就可以自动生成动画角色并为其上色。

![center width:400px](https://img-blog.csdnimg.cn/img_convert/202c9708debb1e1e34399a648bb38409.png)

使用Generative Adversarial Networks创建自动动画人物角色

----


生成器和判别器由多层卷积层、批标准化和具有跳过链接的relu组成。


![img](https://img-blog.csdnimg.cn/img_convert/fc85ec71e877cdb23c18e5c9fc828bd2.png)

----

### **神经照片编辑器**

基于内容的图像编辑：例如，扩展发带。



![img](https://img-blog.csdnimg.cn/img_convert/89f045f9caf08c26b97ddec27d8cb7fd.png)神经照片编辑

---

### **细化图像**



![img](https://img-blog.csdnimg.cn/img_convert/a4e2985fa10f838529c1679a20eb8068.png)



### **目标检测**

这是用gan增强现有解决方案的一个应用程序。



![img](https://img-blog.csdnimg.cn/img_convert/e3e54d8cb337c5a541a909a3cd6f1001.png)

---

### **图像融合**

将图像混合在一起。

![img](https://img-blog.csdnimg.cn/img_convert/684ca7a82f79f48bc5736eb23cd06dd6.png)


### **生成三维对象**

这是用gan创建三维对象时经常引用的一篇文章。



![img](https://img-blog.csdnimg.cn/img_convert/340d70479b014bb0fe91d35387628f65.png)

----

### **音乐的产生**

GaN可以应用于非图像领域，如作曲。



![img](https://img-blog.csdnimg.cn/img_convert/23393f2d9cbb3d599b38febbc670590b.png)

![img](https://img-blog.csdnimg.cn/img_convert/93ba4bb119408943e36241eb095934f6.png)

----

### **医疗（异常检测）**

GAN还可以扩展到其他行业，例如医学中的肿瘤检测。


![img](https://img-blog.csdnimg.cn/img_convert/1690f896b5f36a438cc03dcac6ce06e1.png)

![img](https://img-blog.csdnimg.cn/img_convert/15a9044bf333e359fcac2f8a0ce42f26.png)