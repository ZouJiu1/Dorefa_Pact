## Dorefa、Pact<br>
[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160) <br>
[PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085)<br>

## About pact
Pact is used to replace relu activation function, not in convolution<br>
If using Pact in QuantConv2d, the precision will be very slow<br>
`
The loaded network is trained with the proposed
quantization scheme in which ReLU is replaced with the proposed parameterized clipping ActFn for
each of its seven convolution layers.
` <br>

## commit log<br>
2023-01-08, upload dorefa、pact<br>

I'm not the author, I just complish an unofficial implementation of dorefa and pact.<br>

pytorch==1.11.0+cu113<br>

You should train 32-bit float model firstly, then you can finetune a low bit-width quantization QAT model by loading the trained 32-bit float model<br>

Dataset used for training is [CIFAR10](https://share.weiyun.com/o5wmm1hk) and model used is Resnet18 revised<br>

## The Train Results 
### For the below table all set a_bit=8, w_bit=8
| version | learning rate | batchsize | Accuracy | models
| ------  | ------ | ------ | ------  | ------ |
| Float 32bit| <=66 0.1<br><=86 0.01<br><=99 0.001<br><=112 0.0001 | 128 | 92.6 | [download](https://share.weiyun.com/g7P6cL23) |
| dorefa | <=31 0.1<br><=51 0.01<br><=71 0.001| 128*7+30 | 95 | [download](https://share.weiyun.com/2wEeFGaX) |
| pact | <=31 0.1<br><=51 0.01<br><=71 0.001| 128*7+30 | 95 | [download](https://share.weiyun.com/msSItAk5) |
<br>




### References<br>
[https://github.com/ZouJiu1/LSQplus](https://github.com/ZouJiu1/LSQplus)<br>
[https://github.com/666DZY666/micronet](https://github.com/666DZY666/micronet)<br>
[https://github.com/hustzxd/LSQuantization](https://github.com/hustzxd/LSQuantization)<br>
[https://github.com/zhutmost/lsq-net](https://github.com/zhutmost/lsq-net)<br>
[https://github.com/Zhen-Dong/HAWQ](https://github.com/Zhen-Dong/HAWQ)<br>
[https://github.com/KwangHoonAn/PACT](https://github.com/KwangHoonAn/PACT)<br>
[https://github.com/Jermmy/pytorch-quantization-demo](https://github.com/Jermmy/pytorch-quantization-demo)<br>