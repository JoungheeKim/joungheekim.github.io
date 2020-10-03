---
layout:     post
title:      "[코드리뷰]LSTM AutoEncoder"
subtitle:   "A Generative Model for Raw Audio"
mathjax: true
tags:
  - Text-to-Speech
  - Speech Synthesis
  - Deep Learning
---

# [코드리뷰] - [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681), ICML 2015

비디오는 여러개의 이미지 프레임으로 이루어진 Sequence 데이터 입니다.
따라서 비디오 데이터는 한개의 이미지로 이루어진 데이터보다 큰 차원을 다루므로 학습에 많은 비용이 필요하며 한정적인 Labeled 데이터만으로 학습하기 어렵습니다.
이를 해결하기 위하여 Unlabeled 데이터를 활용하여 일련의 데이터를 학습하는 Unsupervised 방법이 필요합니다.

AutoEncoder는 원본데이터를 특징백터(Feature)로 인코딩하고 재구성(Reconstruction)하는 방법으로 학습하기 때문에 Labeled 데이터 없이 학습이 가능한 Unsupervised 방법입니다.
오늘 포스팅할 논문은 AutoEncoder에 LSTM 구조를 추가하여 Sequence 데이터를 Self-Supervised 방법으로 학습하는 `LSTM AutoEncoder` 입니다.
이 글은 [Unsupervised Learning of Video Representations using LSTMs 논문](https://arxiv.org/abs/1502.04681) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 Pytorch 라이브러리를 이용하여 코드를 구현한 후 자세하게 설명드리겠습니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. 비디오와 같은 Sequence 데이터를 학습할 수 있는 **Self-Supervised 방법**을 제시합니다.
2. 비디오 데이터로부터 **이미지의 모습**과 **이미지의 이동 방향** 등의 정보를 담고 있는 Feature 벡터를 추출할 수 있습니다.
3. 이미지 재구성(Reconstruction) 문제와 이미지를 예측(Prediction) 문제를 함께 수행하여 Sequence 데이터를 효율적으로 활용하는 구조를 제시합니다. 

## 모델 구조
![](/img/in-post/2020/2020-10-11/model_structure.gif)
<center>모델 상세 구조 예시</center>

모델은 **Encoder, Reconstruction Decoder, Predict Decoder** 로 구성됩니다.
Encoder에서 <u>이미지 Sequence를 압축</u>하고 Reconstruction Decoder에서 <u>이미지 Sequence를 재구성</u>하는 **AutoEncoder의 형태**를 띄고 있습니다.
Prediction Decoder는 Encoder에서 압축된 Feature를 이용하여 이후에 나올 <u>이미지 Sequence를 생성</u>합니다.

### [1]Encoder

Encoder는 **Sequence 데이터를 압축**하는 LSTM 모듈입니다. 
Sequence 데이터는 차례대로 LSTM 모듈의 Input으로 사용되어 Feature 벡터로 변환됩니다.
Feature 벡터는 Sequence 데이터를 압축한 형태로 <u>이미지의 모습</u>과 <u>이미지의 이동방향</u> 등의 정보가 포함되어 있습니다.

### [2]Reconstruction Decoder

Reconstruction Decoder는 Encoder에서 생성된 Feature 벡터를 이용하여 Input Sequence 데이터를 복원하는 LSTM 모듈입니다.
복원 순서는 Input Sequence의 반대 방향으로 진행합니다.
즉 Input Sequence가 $v_1, v_2, ..., v_n$ 이었다면 Reconstruction Decoder에서는 $v_n, v_{n-1}, ..., v_1$ 순으로 복원합니다.
>복원 순서를 Input Sequence의 반대방향으로 하는 것이 생성 상관관계를 줄이는 역할을 하여 모델이 잘 학습되도록 한다고 논문에서 주장합니다.

복원 과정의 첫번째 Hidden 벡터는 앞서 Encoder에서 만든 Feature 벡터입니다.
복원 과정의 매 $t$ 시점에서 사용하는 hidden 벡터는 이전 시점 $t-1$에서 Reconstruction Deocder에서 생성된 hidden 벡터 $h_{t-1}^r$ 입니다.
복원 과정의 첫번째 Input은 임의로 생성한 0으로 구성된 벡터를 사용합니다.
복원 과정의 매 $t$ 시점에서 사용하는 Input은 이전 시점 $t-1$에서 Reconstruction Deocder에서 생성된 이미지인 $\hat{v_{t-1}^r}$ 입니다.

### [3]Prediction Decoder

Prediction Decoder는 Encoder에서 생성된 Feature 벡터를 이용하여 Input Sequence 데이터 이후 이미지를 생성하는 LSTM 모듈입니다.
즉 Input Sequence가 $v_1, v_2, ..., v_n$ 이었다면 Prediction Decoder에서는 $k$개의 Sequence, $v_{n+1}, v_{n+2}, ..., v_{n+k}$ 를 생성합니다.

생성 과정의 첫번째 Hidden 벡터는 앞서 Encoder에서 만든 Feature 벡터입니다.
생성 과정의 매 $t$ 시점에서 사용하는 hidden 벡터는 이전 시점 $t-1$에서 Prediction Deocder에서 생성된 hidden 벡터 $h_{t-1}^p$ 입니다.
생성 과정의 첫번째 Input은 임의로 생성한 0으로 구성된 벡터를 사용합니다.
생성 과정의 매 $t$ 시점에서 사용하는 Input은 이전 시점 $t-1$에서 Prediction Deocder에서 생성된 이미지인 $\hat{v_{t-1}^p}$ 입니다.

### 구조적 장점

LSTM AutoEncoder는 Reconstruction Task와 Prediction Task를 함께 학습함으로써 각각의 Task만을 학습할 경우 발생하는 단점을 극복할 수 있습니다.

Reconstruction Task만을 수행하여 모델을 학습할 경우 모델은 Input의 사소한 정보까지 보존하여 Feature 벡터를 생성합니다.
즉 사소한 정보가 저장될 수 없게 Feature 벡터의 크기를 작게 설정하지 않으면 Overfitting이 발생하는 단점이 존재합니다.

Prediction Task만을 수행하여 모델을 학습할 경우 모델은 Input의 최근 Sequence 정보만을 이용하여 학습합니다.
일반적으로 Prediction에 필요한 정보는 예측하기 전 시점에 가까울수록 상관관계가 높기 때문입니다.
따라서 과거 시점의 정보를 활용하지 못하는 단점이 존재합니다.

두가지 Task를 함께 학습함으로써 모델이 모든 정보를 저장하지 않고 중요정보(이미지 모습, 이동방향 등)를 Feature에 저장하도록 유도할 수 있습니다.  
또한 Sequence 데이터의 모든 시점 정보를 활용하여 학습함으로써 모델이 쉽게 학습할 수 있도록 돕는 역할을 합니다.

### 코드 구현

---
**NOTE**  
이 Tutorial은 pytorch, numpy, torchvision, easydict, tqdm, matplotlib, celluloid 라이브러리가 필요합니다.  
2020.10.11 기준 최신 버전의 라이브러리를 이용하여 구현하였고 이후 업데이트 버전에 따른 변경은 고려하고 있지 않습니다.
---

#### 데이터
![](/img/in-post/2020/2020-10-11/data_description.gif)
<center>Moving MNIST 데이터 예시</center>

Tutorial에서 사용하는 데이터는 [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) 입니다.
이 데이터는 9000개의 학습 비디오 데이터와 1000개의 평가 비디오 데이터로 구성되어 있습니다. 
비디오는 20개의 이미지 Frame으로 구성되어 있습니다.
각 이미지는 64×64 픽셀 크기와 1개의 Channel로 구성되어 있고, 이미지 내에 두개의 숫자가 임의의 좌표에 위치해 있습니다.
각 비디오는 두개의 임의의 숫자가 원을 그리며 각각 다른 속도를 갖고 움직이고 있습니다.

![](/img/in-post/2020/2020-10-11/data_download.gif)
<center>데이터 모듈 다운로드 예시</center>

![](/img/in-post/2020/2020-10-11/data_module.png)
<center>MovingMNIST.py 파일 예시</center>

데이터 제공 모듈(**MovingMNIST.py**)을 [[MovingMNIST GITHUB]](https://github.com/tychovdo/MovingMNIST) 에서 다운 받아 작업하고 있는 폴더에 위치시킵니다.

 







총 9개의 숫자로 구성된 손글씨(MNIST) 데이터를 임의로 2개 추출한 다음 Velocity 




#### Summary
이 튜토리얼은 간략하게 LSTM AutoEncoder 구조를 설명하기 위하여 이용하여 제작되었습니다.
LSTM AutoEncoder는 비디오와 같이 여러개의 이미지 프레임으로 이루어진 데이터를 학습하는 모듈입니다.
LSTM AutoEncoder의 목표는 연속된 이미지로부터 이미지의 변동(Velocity)과 이미지의 모습을 학습하는 것 입니다.


### 데이터
실험데이터로 [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) 를 사용합니다.
Moving MNIST 데이터는 MNIST 데이터를 



Pytorch를 이용하여 
This tutorial provides a brief explanation of the U-Net architecture as well as a way to implement it using Theano and Lasagne. 
U-Net is a Fully Convolutional Network (FCN) that does image segmentation. 
Its goal is then to predict each pixel’s class. 
See Fully Convolutional Networks (FCN) for 2D segmentation for differences between network architecture for classification and segmentation tasks.
