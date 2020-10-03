---
layout:     post
title:      "[코드리뷰]LSTM AutoEncoder"
subtitle:   "Unsupervised Learning of Video Representations using LSTMs"
mathjax: true
tags:
  - AutoEncoder
  - Vision
  - Deep Learning
---

# [코드리뷰] - [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681), ICML 2015

비디오는 여러개의 이미지 프레임으로 이루어진 Sequence 데이터 입니다.
따라서 비디오 데이터는 한개의 이미지로 이루어진 데이터보다 큰 차원을 다루므로 <u>학습에 많은 비용이 필요</u>하며 한정적인 labeled 데이터만으로 학습하기 어렵습니다.
이를 해결하기 위하여 unlabeled 데이터를 활용하여 일련의 데이터를 학습하는 **unsupervised** 방법이 필요합니다.

**AutoEncoder**는 원본데이터를 특징백터(feature)로 인코딩하고 재구성(reconstruction)하는 방법으로 학습하기 때문에 <u>labeled 데이터 없이 학습이 가능</u>한 unsupervised 방법입니다.
오늘 포스팅할 논문은 **AutoEncoder에 LSTM 구조를 추가**하여 sequence 데이터를 Self-Supervised 방법으로 학습하는 `LSTM AutoEncoder` 입니다.

이 글은 [Unsupervised Learning of Video Representations using LSTMs 논문](https://arxiv.org/abs/1502.04681) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 **pytorch** 라이브러리를 이용하여 <u>코드를 구현</u>한 후 자세하게 설명드리겠습니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. 비디오와 같은 sequence 데이터를 학습할 수 있는 **Self-Supervised 방법**을 제시합니다.
2. 비디오 데이터로부터 **이미지의 모습**과 **이미지의 이동 방향** 등의 정보를 담고 있는 feature 벡터를 추출할 수 있습니다.
3. 이미지 재구성(reconstruction) 문제와 이미지를 예측(prediction) 문제를 함께 수행하여 **sequence 데이터를 효율적으로 활용**하는 구조를 제시합니다. 

## 모델 구조
![](/img/in-post/2020/2020-10-11/model_structure.gif)
<center><b>모델 상세 구조 예시</b></center>

모델은 **Encoder, Reconstruction Decoder, Predict Decoder** 로 구성됩니다.
Encoder에서 <u>이미지 sequence를 압축</u>하고 Reconstruction Decoder에서 <u>이미지 sequence를 재구성</u>하는 **AutoEncoder의 형태**를 띄고 있습니다.
Prediction Decoder는 Encoder에서 압축된 feature를 이용하여 이후에 나올 <u>이미지 sequence를 생성</u>합니다.

### [1] Encoder

Encoder는 **sequence 데이터를 압축**하는 LSTM 모듈입니다. 
sequence 데이터는 차례대로 LSTM 모듈의 input으로 사용되어 feature 벡터로 변환됩니다.
Feature 벡터는 sequence 데이터를 압축한 형태로 <u>이미지의 모습</u>과 <u>이미지의 이동방향</u> 등의 정보가 포함되어 있습니다.

### [2] Reconstruction Decoder

Reconstruction Decoder는 Encoder에서 생성된 feature 벡터를 이용하여 input Sequence 데이터를 복원하는 LSTM 모듈입니다.
복원 순서는 input Sequence의 반대 방향으로 진행합니다.
즉 input sequence가 $v_1, v_2, ..., v_n$ 이었다면 Reconstruction Decoder에서는 $v_n, v_{n-1}, ..., v_1$ 순으로 복원합니다.
>복원 순서를 input sequence의 반대방향으로 하는 것이 생성 상관관계를 줄이는 역할을 하여 모델이 잘 학습되도록 한다고 논문에서 주장합니다.

복원 과정의 첫번째 hidden 벡터는 앞서 Encoder에서 만든 feature 벡터입니다.
복원 과정의 매 $t$ 시점에서 사용하는 hidden 벡터는 이전 시점 $t-1$에서 Reconstruction Deocder에서 생성된 hidden 벡터 $h_{t-1}^r$ 입니다.
복원 과정의 첫번째 input은 임의로 생성한 0으로 구성된 벡터를 사용합니다.
복원 과정의 매 $t$ 시점에서 사용하는 input은 이전 시점 $t-1$에서 Reconstruction Deocder에서 생성된 이미지인 $\hat{v^r_{t-1}}$ 입니다.

### [3] Prediction Decoder

Prediction Decoder는 Encoder에서 생성된 feature 벡터를 이용하여 input sequence 데이터 이후 이미지를 생성하는 LSTM 모듈입니다.
input sequence가 $v_1, v_2, ..., v_n$ 이었다면 Prediction Decoder에서는 $k$개의 sequence, 즉 $v_{n+1}, v_{n+2}, ..., v_{n+k}$ 를 생성합니다.

생성 과정의 첫번째 hidden 벡터는 앞서 Encoder에서 만든 feature 벡터입니다.
생성 과정의 매 $t$ 시점에서 사용하는 hidden 벡터는 이전 시점 $t-1$에서 Prediction Decoder에서 생성된 hidden 벡터 $h_{t-1}^p$ 입니다.
생성 과정의 첫번째 input은 임의로 생성한 0으로 구성된 벡터를 사용합니다.
생성 과정의 매 $t$ 시점에서 사용하는 input은 이전 시점 $t-1$에서 Prediction Decoder에서 생성된 이미지인 $\hat{v^p_{t-1}}$ 입니다.

### 구조적 장점

LSTM AutoEncoder는 reconstruction task와 prediction task를 함께 학습함으로써 각각의 task만을 학습할 경우 발생하는 단점을 극복할 수 있습니다.

reconstruction task만을 수행하여 모델을 학습할 경우 모델은 input의 사소한 정보까지 보존하여 Feature 벡터를 생성합니다.
즉 사소한 정보가 저장될 수 없게 Feature 벡터의 크기를 작게 설정하지 않으면 과적합(overfitting)이 발생하는 단점이 존재합니다.

prediction task만을 수행하여 모델을 학습할 경우 모델은 input의 최근 sequence 정보만을 이용하여 학습합니다.
일반적으로 prediction에 필요한 정보는 예측하기 전 시점에 가까울수록 상관관계가 높기 때문입니다.
따라서 과거 시점의 정보를 활용하지 못하는 단점이 존재합니다.

두가지 task를 함께 학습함으로써 모델이 모든 정보를 저장하지 않고 중요정보(이미지 모습, 이동방향 등)를 feature에 저장하도록 유도할 수 있습니다.
또한 Sequence 데이터의 모든 시점 정보를 활용하여 학습함으로써 모델이 쉽게 학습할 수 있도록 돕는 역할을 합니다.

## 코드 구현

:warning: **주의** :warning: 
 
Tutorial은 pytorch, numpy, torchvision, easydict, tqdm, matplotlib, celluloid, tqdm 라이브러리가 필요합니다.
Jupyter로 구현한 코드를 기반으로 글을 작성하고 있습니다. 따라서 tqdm 라이브러리를 python 코드로 옮길때 주의가 필요합니다.
2020.10.11 기준 최신 버전의 라이브러리를 이용하여 구현하였고 이후 업데이트 버전에 따른 변경은 고려하고 있지 않습니다.

#### 데이터
![](/img/in-post/2020/2020-10-11/data_description.gif)
<center><b>Moving MNIST 데이터 예시</b></center>

Tutorial에서 사용하는 데이터는 [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) 입니다.
이 데이터는 9000개의 학습 비디오 데이터와 1000개의 평가 비디오 데이터로 구성되어 있습니다. 
비디오는 20개의 이미지 frame으로 구성되어 있습니다.
각 이미지는 64×64 픽셀 크기와 1개의 channel로 구성되어 있고, 이미지 내에 두개의 숫자가 임의의 좌표에 위치해 있습니다.
각 비디오는 두개의 임의의 숫자가 원을 그리며 각각 다른 속도를 갖고 움직이고 있습니다.

![](/img/in-post/2020/2020-10-11/data_download.gif)
<center><b>데이터 모듈 다운로드 예시</b></center>

![](/img/in-post/2020/2020-10-11/data_module.png)
<center><b>작업 폴더 예시</b></center>

해당 데이터는 직접 다운로드할 수 있고, 데이터 제공 모듈을 이용하여 다운받을 수 있습니다.
편의상 데이터 제공 모듈을 활용합니다.
데이터 제공 모듈을 [[MovingMNIST GITHUB]](https://github.com/tychovdo/MovingMNIST) 에서 다운 받아 압축을 풀고 작업하고 있는 폴더에 `MovingMNIST.py`를 위치시킵니다.

##### 1. 라이브러리 Import
``` python
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms, datasets
import easydict
from tqdm.notebook import tqdm
from tqdm.notebook import trange
import torch.utils.data as data
from celluloid import Camera
```
모델을 구현하는데 필요한 라이브러리를 Import 합니다.
Import 에러가 발생하면 해당 라이브러리를 설치한 후 진행합니다.

##### 2. 데이터 불러오기
``` python
from MovingMNIST import MovingMNIST

## Train Data를 불러오기
train_set = MovingMNIST(root='.data/mnist', train=True, download=True, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())

## Test Data를 불러오기
test_set = MovingMNIST(root='.data/mnist', train=False, download=True, transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
```
데이터 모듈 `MovingMNIST.py` 을 Import 하고 데이터 모듈을 이용하여 데이터를 다운받습니다.
옵션으로 transform, target_transform을 설정하여 다운 받은 파일을 불러올 때 전처리를 할 수 있습니다.
숫자열로 구성된 이미지 데이터를 pytorch의 tensor로 변형하기 위하여 `transforms.ToTensor()` 를 옵션으로 넣어줍니다.

``` python 
## 데이터 시각화
def imshow(past_data, title='없음'):
    num_img = len(past_data)
    fig = fig=plt.figure(figsize=(4*num_img, 4))
    
    for idx in range(1, num_img+1):
        ax = fig.add_subplot(1, num_img+1, idx)
        ax.imshow(past_data[idx-1])
    plt.suptitle(title, fontsize=30)

## 데이터는 Tuple 형태로 되어 있음.
## past_data 10개, future_data 10개로 구성
past_data, future_data = train_set[0]
imshow(past_data, title='input')
```
![](/img/in-post/2020/2020-10-11/past_data_visualization.png)

데이터 모듈은 Tuple 형태로 데이터를 제공합니다.
테이터는 과거 데이터와 미래 데이터로 구성됩니다.
데이터 모듈로부터 데이터를 로드하고 시각화하여 잘 다운로드 되었는지 확인합니다.

##### 3. 모델 구성하기
``` python 
class Encoder(nn.Module):

    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.1, bidirectional=False)

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)

class Decoder(nn.Module):

    def __init__(self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)        
        self.fc = nn.Linear(hidden_size, output_size)
   
        
    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)
```

논문에서 제시한 모델은 `Encoder`와 `Decoder` 모듈로 구성됩니다.
Deocder는 쓰임세에 따라 Reconstruction Decoder, Prediction Decoder로 나뉩니다.

``` python
class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        hidden_size = args.hidden_size
        input_size = args.input_size
        output_size = args.output_size
        num_layers = args.num_layers
        
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        self.predict_decoder = Decoder(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        self.criterion = nn.MSELoss()
    
    ## Loss 출력
    def forward(self, src, trg):
        # src: tensor of shape (batch_size, seq_length, hidden_size)
        # trg: tensor of shape (batch_size, seq_length, hidden_size)

        batch_size, sequence_length, img_size = src.size()
        
        ## Encoder 넣기
        encoder_hidden = self.encoder(src)
        
        ## Prediction Loss 계산 
        predict_output = []
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = self.predict_decoder(temp_input, hidden)
            predict_output.append(temp_input)
            
        predict_output = torch.cat(predict_output, dim=1)
        predict_loss = self.criterion(predict_output, trg)
        
        ## Reconstruction Loss 계산
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_output = []
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)
        reconstruct_loss = self.criterion(reconstruct_output, src[:, inv_idx, :])
            
        return reconstruct_loss, predict_loss
    
    ## 이미지 생성(Prediction)
    def generate(self, src):
        batch_size, sequence_length, img_size = src.size()
        
        ## Encoder 넣기
        hidden = self.encoder(src)
        
        outputs = []
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        for t in range(sequence_length):
            temp_input, hidden = self.predict_decoder(temp_input, hidden)
            outputs.append(temp_input)
        
        return torch.cat(outputs, dim=1)
    
    ## 이미지 복구(Reconstruction) 
    def reconstruct(self, src):
        batch_size, sequence_length, img_size = src.size()
        
        ## Encoder 넣기
        hidden = self.encoder(src)
        
        outputs = []
        temp_input = torch.zeros((batch_size,1,img_size), dtype=torch.float).to(src.device)
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            outputs.append(temp_input)
        
        return torch.cat(outputs, dim=1) 
```
앞서 구성한 Encoder, Decoder를 모듈을 이용하여 `Seq2Seq`를 구성합니다.
Seq2Seq 모듈에 3가지 함수을 구현하였습니다.
`def forward` 는 과거이미지와 미래이미지를 받아 Loss를 계산하는 함수입니다.
`def generate` 는 과거이미지를 이용하여 미래이미지를 생성하는 함수입니다.
`def reconstruct` 는 과거이미지를 encoding한 다음 다시 복구하는 함수입니다.

>Loss를 계산하기 위한 함수로 Mean Squared Loss Function을 사용하였습니다.
>논문에서는 Binary Entropy Loss를 사용하였으나 개인적으로 실험을 했을 때 MSE를 사용한 모델이 이미지를 더 명확하게 추출합니다.
>     





##### 4. 학습 구성


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
