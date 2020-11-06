---
layout:     post
title:      "[코드리뷰]LSTM-based Anomaly Detection"
subtitle:   "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
mathjax: true
tags:
  - AutoEncoder
  - Anomaly Detection
  - Deep Learning
---

# [코드리뷰] - [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148), ICML 2016

유압기, 회전엔진 등에 부착된 센서를 이용하여 **기계의 수명 및 건강**을 확인하는 것은 산업분야에서 매우 중요한 TASK 중 하나입니다.
하지만 <u>다양한 외부 환경</u>과 <u>복잡한 물리적 연결관계</u>를 고려하여 기계의 이상을 감지하는 것은 매우 어려운 일입니다.
게다가 일반적으로 센서데이터는 이상치 데이터가 적고 정상 데이터가 많은 **Label Unbalance 데이터** 이므로 Supervised 방법론으로 학습하기 어렵습니다.
이를 해결하기 위하여 **Unsupervised Anomaly Detection** 방법이 필요합니다.

오늘 포스팅할 논문은 다변량 시계열데이터의 정상데이터를 <u>Unsupervised 방법으로 학습하고 이상치를 탐지</u>하는 모델인 **LSTM based AutoEncoder** 입니다.
시계열데이터가 생성되는 다양한 분야에 보편적으로 적용할 수 있다는 장점을 갖고 있기 때문에 활용성이 높은 방법론입니다.

이 글은 **[LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)** 논문을 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 pytorch 라이브러리를 이용하여 코드를 구현한 내용을 자세히 설명드리겠습니다.
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 2가지는 아래와 같습니다.

1. LSTM Auto-Encoder를 활용하여 **다변량 시계열 데이터를 학습**하는 방법을 제시합니다..
2. Label Unbalance 시계열 데이터에 **Unsupervised Anomaly Detection 방법론**을 적용하는 방법에 대해 제시합니다.

## 모델 구조
![](/img/in-post/2020/2020-11-14/autoencoder_example.png)
<center><b>모델 구조 예시</b></center>

LSTM Auto-Encoder 모델은 LSTM-Encoder와 LSTM-Decoder로 구성되어 있습니다.
Encoder는 다변량 데이터를 압축하여 feature로 변환하는 역할을 합니다.
Decoder는 Encoder에서 받은 feature를 이용하여 Encoder에서 받은 다변량 데이터를 재구성하는 역할을 합니다.
Encoder의 input과 Decoder에서 나온 output의 차이를 줄이도록 학습함으로써 Auto-Encoder는 데이터의 특징을 압축할 수 있습니다. 

### 학습 과정(Training)
![](/img/in-post/2020/2020-11-14/training_process.png)
<center><b>모델 학습 과정 예시</b></center>

모델 학습과정은 위의 그림과 같습니다.
Encoder는 입력으로 $n$ 개의 연속한 벡터 $x^{(1)}, x_^{(2)}, x_^{(3)}, ..., x_^{(n)}$를 사용합니다.
입력 $x_^(k)$는 다변량 데이터이므로 변수의 갯수 $m$개로 구성된 벡터 ($x_^{(k)} \in R^m$) 입니다.
Encoder는 매 step 입력으로 $x_^{(t)}$와 이전 step에서 Encoder로부터 받은 hidden state인 $h_E^{(t-1)}$을 활용하여 정보를 압축하고 다음 step의 hidden state인 $h_E^{(t)}$를 생성합니다.
Encoder의 마지막 step에서 생성된 $h_E^{(t)}$는 feature vector로 부르며 Decoder의 초기 hidden state로 활용됩니다.

Decoder는 입력으로 Encoder에서 생상한 feature를 받아 original 데이터를 역순으로 재구성합니다.
즉 Deocder는 $\hat{x^{(n)}}, \hat{x^{(n-1)}}, ..., \hat{x^{(1)}}$를 차례로 생성합니다.
이때 Decoder의 매 step 입력으로 original 데이터 역순인 $x^{(t+1)}$과 이전 step에서 Decoder로부터 받은 hidden state인 $h_D^{(t-1)}$을 활용하여 정보를 압축하고 다음 step의 hidden state인 $h_D^{(t)}$를 생성합니다.
다음 step의 Decoder에 hidden state $h_D^{(t)}$를 넘기기 전 Fully Connected Linear layer에 통과시켜 reconstruction 데이터인 $\hat{x^{(n-t+1)}}$를 생성합니다.

Auto-Encoder의 입력인 $x^{(1)}, x_^{(2)}, x_^{(3)}, ..., x_^{(n)}$ 와 출력인 $\hat{x^{(1)}}, \hat{x^{(2)}}, ..., \hat{x^{(n)}}$ 의 차이인 MSE(Mean Squared Error)를 최소화하는 방향으로 학습합니다.

##### 학습 목표
<center><b>$Minimize \sum_{X \in s_N} \sum_{i=1}^L ||x^{(i)} - \hat{x^{(i)}}||$</b></center>
 


 
Decoder는 E

$x_k$     \hat{x^1}X$


### 추론 과정(Inference)
