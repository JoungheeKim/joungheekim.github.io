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
게다가 일반적으로 센서데이터는 이상치 데이터가 적고 정상 데이터가 많은 **unbalanced label 데이터** 이므로 Supervised 방법론으로 학습하기 어렵습니다.
이를 해결하기 위하여 **Unsupervised Anomaly Detection** 방법이 필요합니다.

오늘 포스팅할 논문은 다변량 시계열데이터의 정상데이터를 <u>Unsupervised 방법으로 학습하고 이상치를 탐지</u>하는 모델인 `LSTM based AutoEncoder` 입니다.
시계열데이터가 생성되는 다양한 분야에 보편적으로 적용할 수 있다는 장점을 갖고 있기 때문에 활용성이 높은 방법론입니다.

이 글은 **[LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)** 논문을 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 pytorch 라이브러리를 이용하여 코드를 구현한 내용을 자세히 설명드리겠습니다.
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 2가지는 아래와 같습니다.

1. LSTM Auto-Encoder를 활용하여 **다변량 시계열 데이터를 학습**하는 방법을 제시합니다.
2. Unbalanced label 시계열 데이터에 **Unsupervised Anomaly Detection 방법론**을 적용하는 방법에 대해 제시합니다.

## 논문 리뷰
![](/img/in-post/2020/2020-11-14/autoencoder_example.png)
<center><b>모델 구조 예시</b></center>

LSTM Auto-Encoder 모델은 LSTM-Encoder와 LSTM-Decoder로 구성되어 있습니다.
Encoder는 <u>다변량 데이터를 압축</u>하여 feature로 변환하는 역할을 합니다.
Decoder는 Encoder에서 받은 feature를 이용하여 Encoder에서 받은 <u>다변량 데이터를 재구성</u>하는 역할을 합니다.
Encoder의 input과 Decoder에서 나온 output의 차이를 줄이도록 학습함으로써 Auto-Encoder는 **정상 데이터의 특징을 압축**할 수 있습니다.

### [1]학습 과정(Training)
![](/img/in-post/2020/2020-11-14/training_process.png)
<center><b>모델 학습 과정 예시</b></center>

모델 학습 과정은 위의 그림과 같습니다.
Encoder는 입력으로 $n$ 개의 연속한 벡터 $x^{(1)}, x^{(2)}, ..., x^{(n)}$를 사용합니다.
입력 $x^{(k)}$는 다변량 데이터이므로 변수의 갯수 $m$개로 구성된 벡터 ($x^{(k)} \in R^m$) 입니다.
Encoder는 매 step 입력으로 $x^{(t)}$와 이전 step에서 Encoder로부터 받은 hidden vector인 $h_E^{(t-1)}$을 활용하여 정보를 압축하고 다음 step의 hidden vector인 $h_E^{(t)}$를 생성합니다.
Encoder의 마지막 step에서 생성된 $h_E^{(t)}$는 feature vector로 부르며 Decoder의 초기 hidden vector로 활용됩니다.

Decoder는 입력으로 Encoder에서 생상한 feature를 받아 original 데이터를 역순으로 재구성합니다.
즉 Deocder는 $\hat{x^{(n)}}, \hat{x^{(n-1)}}, ..., \hat{x^{(1)}}$를 차례로 생성합니다.
이때 Decoder의 매 step 입력으로 original 데이터 역순인 $x^{(t+1)}$과 이전 step에서 Decoder로부터 받은 hidden vector인 $h_D^{(t-1)}$을 활용하여 정보를 압축하고 다음 step의 hidden vector인 $h_D^{(t)}$를 생성합니다.
다음 step의 Decoder에 hidden vector $h_D^{(t)}$를 넘기기 전 Fully Connected Linear layer에 통과시켜 reconstruction 데이터인 $\hat{x^{(n-t+1)}}$를 생성합니다.

Auto-Encoder의 입력인 $x^{(1)}, x^{(2)}, ..., x^{(n)}$ 와 출력인 $\hat{x^{(1)}}, \hat{x^{(2)}}, ..., \hat{x^{(n)}}$ 의 차이인 MSE(Mean Squared Error)를 최소화하는 방향으로 학습합니다.

Auto-Encoder를 <u>학습하는 과정</u>에는 이상치 데이터가 없는 **정상 데이터만을 사용**합니다.
또한 학습과정에서 Decoder의 입력으로 original 데이터($x^{(1)}, x^{(2)}, x^{(3)}, ..., x^{(n)}$)를 활용하는 **Teacher Forcing 테크닉을 적용**합니다.

##### 학습 목표
<center><b>$Minimize \sum_{X \in s_N} \sum_{i=1}^N ||x^{(i)} - \hat{x^{(i)}}||^2$</b></center>
**$s_N$** : 정상 데이터  
**$x^{(i)}$** : original 데이터(input)  
**$\hat{x^{(i)}}$** : reconstructed 데이터(output)  
**N** : input sequence length

### [2]추론 과정(Inference)
![](/img/in-post/2020/2020-11-14/inference_process.png)
<center><b>모델 추론 과정 예시</b></center>

모델 추론 과정은 위의 그림과 같습니다.
학습 과정과 비교하면 동일한 점은 Encoder에서 feature vecotor를 생성하고 Decoder의 초기 hidden vector로 Encoder의 feature vector를 사용한다는 점 입니다.
다른점은 Decoder의 입력으로 original 데이터($x^{(n-t+1)}$)가 아닌 이전 step의 Deocder에서 생성된 reconstructed output($\hat{x^{(n-t+1)}}$)을 사용한다는 점 입니다.

##### Reconstruction Error 계산
<center><b>$e^{(i)} = ||x^{(i)} - \hat{x^{(i)}}||$</b></center>
**$e^{(i)}$** : i지점 데이터의 Reconstruction Error  

MSE Loss를 이용하여 학습하였지만 추론 과정에서 Error를 계산하는 방법은 위와 같이 Absolute Error를 활용합니다. 

### [4]데이터 분할
![](/img/in-post/2020/2020-11-14/data_split.png)
<center><b>데이터 활용 예시</b></center>

데이터는 정상구간(noraml), 비정상구간(abnormal)으로 구성되어 있습니다.
본 논문에서는 정상구간을 4개($S_N, V_{N_1}, V_{N_2}, T_N$), 비정상구간을 2개($V_A, T_A$)로 나누어 학습, 검증, 실험에 활용합니다.

#### 학습 데이터
학습용 데이터($S_N$) 는 모델을 학습하는데 사용하는 데이터 입니다. 이 데이터를 활용하여 정상구간의 정보를 압축할 수 있도록 모델을 학습합니다.
학습된 모델을 활용하면 특정 데이터 구간이 입력으로 들어왔을 때 추론 과정을 통해 구간별 Reconstruction Error를 구할 수 있습니다.

#### 파라미터 추정 데이터
파라미터 추정용 데이터($V_{N_1}$)는 Reconstruction Error의 분포의 파라미터를 추정하는데 활용합니다.
Reconstruction Error가 정규분포를 따른다고 가정하고 정규분포의 파라미터 $N(\mu, \sum)$ 를 Maximum Likelihood Estimation(MSE)을 활용하여 구합니다.
이 후 아래와 같은 식을 활용하여 각 구간의 비정상 점수(Anomaly Score)를 계산할 수 있습니다.

<center>$a^{(i)} = ( e^{(i)} - \mu )^T \sum^{-1} ( e^{(i)} - \mu )$</center>
**$a^{(i)}$** : i지점 데이터의 비정상 점수  

이 비정상 점수(Anomaly Score)가 사용자가 지정한 Threshold($\tau$) 를 상회하면 $a^{(i)} > \tau$ 이 지점($i$)를 비정상 이라고 정의합니다.
이를 통해 추론단계에서 각 지점 및 구간의 비정상 여부를 판단 할 수 있습니다.

#### 검증 데이터
앞서 학습용 데이터를 이용하여 모델을 학습하고, 파라미터 추정 데이터를 활용하여 비정상을 정의하기 위한 파라미터를 도출하였습니다.
이를 활용하여 검증용 데이터($V_{N_2}, V_A$)에서 비정상 구간과 정상 구간을 잘 분류하는지 확인합니다.

#### 테스트 데이터
검증 데이터를 활용하여 최종 학습모델과 최종 파라미터를 도출한 후 테스트 데이터($T_N, T_A$)에서 모델의 최종 성능을 도출합니다.

## 코드 구현

<p style="text-align: center;"><b><i class="fa fa-exclamation-triangle" aria-hidden="true"></i> 주의 <i class="fa fa-exclamation-triangle" aria-hidden="true"></i></b></p>  
튜토리얼은 pytorch, numpy, torchvision, easydict, tqdm, matplotlib, celluloid, tqdm 라이브러리가 필요합니다.
2020.10.11 기준 최신 버전의 라이브러리를 이용하여 구현하였고 이후 **업데이트 버전에 따른 변경은 고려하고 있지 않습니다.**
<u>Jupyter로 구현한 코드를 기반</u>으로 글을 작성하고 있습니다. 따라서 tqdm 라이브러리를 python 코드로 옮길때 주의가 필요합니다.

#### 데이터
![](/img/in-post/2020/2020-10-11/data_description.gif)
<center><b>Moving MNIST 데이터 예시</b></center>

튜토리얼에서 사용하는 데이터는 [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) 입니다.
이 데이터는 9000개의 학습 비디오 데이터와 1000개의 평가 비디오 데이터로 구성되어 있습니다. 
비디오는 20개의 이미지 frame으로 구성되어 있습니다.
각 이미지는 64×64 픽셀, 1개의 channel로 구성되어 있고 이미지 내에 두개의 숫자가 임의의 좌표에 위치해 있습니다.
각 비디오는 두개의 임의의 숫자가 원을 그리며 각각 다른 속도로 움직이고 있습니다.





