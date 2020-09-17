---
layout:     post
title:      "[논문리뷰]WaveNet"
subtitle:   "A Generative Model for Raw Audio"
mathjax: true
tags:
  - Time-series
  - Univariate
  - Multiple step prediction 
  - Deep Learning
---

# [논문리뷰] - [A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499), DeepMind

딥러닝 기반 음성합성방법이 등장하기 전까지는 아래와 같은 2가지 대표적인 방식을 주로 채택하여 음성을 생성하거나 합성하였습니다. 

1. Concatenative TTS 방식 : 다량의 음성데이터를 음소로 분리하고 조합하여 새로운 음성을 생성하는 방식 
2. Parametric TTS 방식 : 통계적 모델(은닉 마르코프 모델)을 기반으로 음성을 합성하는 방식 

하지만 위 방법으로부터 생성된 음성은 실제 사람의 음성만큼 매끄럽지 않으며 음편사이의 경계가 부자연스러운 문제들이 있습니다.

2016년 [딥마인드(DeepMind)](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) 에서 딥러닝 기반 음성 생성모델에 관한 논문을 공개하였습니다.  
오늘은 이 딥러닝 기반 End-to-End 음성 생성 모델인 `WaveNet` 을 포스팅하겠습니다.
이 글은 [WaveNet 논문](https://arxiv.org/abs/1609.03499) 과 [Medium 글](https://medium.com/@satyam.kumar.iiitv/understanding-wavenet-architecture-361cc4c2d623) 을 참고하여 정리하였음을 먼저 밝힙니다. 
또한 논문을 이해하기 위하여 필요한 내용을 외부 코드 및 글을 정리하여 추가하였으므로 포스팅한 글은 **논문의 내용만을 담고 있지 않습니다.** 
제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 4가지는 아래와 같습니다.

1. WaveNet은 자연스러운 음성 파형을 직접 생성합니다.
2. 긴 음성 파형을 학습하고 생성할 수 있는 새로운 구조를 제시합니다.  
3. 학습된 모델은 컨디션 모델링으로 인해 다양한 특징적인 음성을 생성할 수 있습니다.
4. 음악을 포함한 다양한 음성 생성분야에서도 좋은 성능을 보입니다.

## 모델 구조
![](/img/in-post/2020/2020-09-17/overview.png)
<center>Figure 1 : <a href="https://www.youtube.com/watch?v=nsrSrYtKkT8">WaveNet 전체구조</a></center>

WaveNet은 30개의 Residaul Block을 쌓은 형태의 구조를 갖고 있습니다.
정수 배열을 Input으로 받아 첫번째 Residaul Block부터 30번째 Residual Block까지 차례대로 들어갑니다.
각각의 Residual Block으로부터 생성된 Output은 Skip Connection을 통해 합쳐지고 이를 모델의 Output으로 활용합니다. 

### 1) Modeling
WaveNet은 확률론적 모형(Probabilistic Model)으로써 T개의 배열로 구성된 음성 데이터 $x_1, ..., x_{T-1} ,x_{T}$ 열이 주어졌을 때 음성으로써 성립할 확률 $P(x_1, ..., x_{T-1} ,x_{T})$ 을 학습하여 이후 생성에 활용합니다.
이 확률은 각 음성 데이터들의 조건부 확률을 이용하여 아래와 같이 표현될수 있습니다.
<center>$P(x_1, ..., x_{T-1} ,x_{T})=P(x_1, ..., x_{T-1}) * P(x_{T}|x_1, ..., x_{T-1})$</center>
<center>$=P(x_1, ..., x_{T-2}) * P(x_{T-1}|x_1, ..., x_{T-2}) * P(x_{T}|x_1, ..., x_{T-1})$</center>
<center>$=\prod_{t=1}^T P(x_t|x_1, x_2, ..., x_{t-1})$</center>

위 조건부확률을 따르는 모델은 $t$ 시점을 기준으로 과거의 음성 데이터 $x_1, ..., x_{t-1} ,x_{t}$ 을 이용하여 한 시점 뒤 음성 데이터 $x_{t+1}$가 나올 확률을 나타낼 수 있으므로 그 확률을 이용하여 음성을 생성할 수 있습니다.
[[참고문서]](https://datascienceschool.net/view-notebook/a0c848e1e2d343d685e6077c35c4203b/)

### 2) Input & Output
![](/img/in-post/2020/2020-09-17/analog_to_digital_conversion.png)
<center>Figure 2 : 아날로그 음성 데이터로부터 디지털 데이터로 변환과정</center>

아날로그 음성 데이터는 연속형(Continous) 데이터입니다. 이 음성 데이터를 컴퓨터에서 처리하거나 저장(`.wav`, `.mp4`)하려면 **디지털 데이터**로 변환해야 합니다.
이 변환하는 과정을 [Analog Digital Conversion](https://hyunlee103.tistory.com/54) 라고 부르며 표본화(Sampling), 양자화(Quantizing)로 구성되어 있습니다.
Analog Digital Conversion 과정을 통해 처리된 음성 데이터는 이산형(Discrete) 디지털 데이터로 변환되어 정수배열(Integer Array)로 표현됩니다.
이 정수배열이 WaveNet의 Input과 Output으로 활용됩니다. [[참고문서]](http://166.104.231.121/ysmoon/mip2017/lecture_note/%EC%A0%9C10%EC%9E%A5.pdf)

### 3) SoftMax Distribution
![](/img/in-post/2020/2020-09-17/input_output.png)
<center>Figure 3 : 모델의 Input & Output 변환과정</center>

일반적인 음성 데이터는 각 샘플을 16(bit) 정수 값으로 저장하므로 앞서 설명한 Anlog Digital Conversion을 통해 생성된 정수배열의 정수는 $-2^7$ ~ $2^7$ 사이의 숫자입니다.
WaveNet은 확률론적 모델링에 따라 매 $t$ 시점 특정 파형이 나올 확률 $P(x_t|x_1, …, x_t−1)$ 을 계산하는데 이 확률을 범주형 분포(Categorical Distribution)로 가정하면 
매 $t$ 시점 $-2^7$ ~ $2^7$ 사이의 숫자가 나올 확률 $P(-2^7|x_1, …, x_t−1), ..., P(2^7|x_1, …, x_t−1)$ 을 계산해야 합니다. 
각 숫자가 나올 확률(총 65,536 개의 확률)을 다루는 것은 매우 어려우므로 WaveNet에서는 이를 256개의 숫자(총 256 개의 확률)로 변환하는 $\mu$-law Companding 이라는 변환방법을 사용합니다.
논문에서는 $\mu$-law Companding와 같은 비선형적인 변환방식이 선형적인 변형방식보다 더 효과적이라고 기술하고 있습니다. 

#### $\mu$-law Companding Transformation
<center>$f(x_t) = sign(x_t)\frac{\ln(1+\mu\mid x_t\mid)}{\ln(1+\mu)}$</center>

결론적으로 WaveNet 모델에서 Input으로 사용하는 것은 $\mu$-law Companding 변환방법을 이용하여 음성 디지털 데이터를 작은 범위의 정수 배열로 변환한 값입니다.
WaveNet 모델로부터 추출된 Output 역시 -127~128(256개) 범위의 정수이며, 이 정수를 Reconstruction을 통해 다시 음성 디지털 데이터로 변형한 것이 최종 결과물입니다.  

### 4) Dilated Causal Convolutions
![](/img/in-post/2020/2020-09-17/dilated_causal_convolution.png)
<center>Figure 4 : Dilated Causal Convolutions 적용범위</center>

모델의 Residual Block은 몇가지 Activation Function과 Neural Layer로 구성되어 있습니다.
이 중에서 과거 음성 정보에 접근하는 구조를 갖는 Layer는 Dilated Causal Convolutions Layer 입니다.
입력의 범위를 중점으로 상세하게 묘사하면 Figure 4의 왼쪽과 같습니다.

Dilated Causal Convolutions Layer 은 Dilated Convolution Layer의 기능과 Causal Convolutions Layer의 기능을 합쳐놓은 Convolution Layer입니다.
[Causal Convolution이란](https://dataplay.tistory.com/29) 시간 순서를 고려하여 Convolution Filter를 적용하는 변형 Convolution Layer입니다.  
Causal Convolution을 위로 쌓을 수록 Input 데이터의 수용 범위(Receptive Field)가 커지므로 RNN 계열의 모델처럼 음성 데이터(시계열 데이터)를 모델링 할 수 있습니다.
다만 Causal Convolution만을 이용하면 수용 범위를 넓히기 위해서 많은 양의 Layer를 쌓아야 하는 단점이 존재합니다. 이를 해결하기 위하여 Dilated Convolution을 함께 적용합니다.

[Dilated Convolution이란](https://dataplay.tistory.com/29) 추출 간격을 조절하여 더 넓은 수용 범위를 갖게 하는 변형 Convolution Layer입니다.
즉 추출 간격을 조절하는 Causal Convolutions 적용하면 적게 Layer를 쌓아도 넓은 수용 범위를 갖을 수 있는 장점을 갖고 있습니다.



WaveNet 논문에서는 이 추출 간격을 1, 2, 4, ..., 512 까지 차례로 늘리며 Layer를 쌓으므로 1024의 수용범위를 갖는 Layer를   
  




RNN 계열의 모델은 학습 및 활용 할때 느린 단점을 갖고 있지만  



WaveNet 모델 역시 시계열 음성 데이터를 모델링 해야 하므로 이 Layer를 적용할 수 있습니다.

 

과거의 음성 데이터 $x_1, ..., x_{t-1} ,x_{t}$ 만을 이용하여 다음 음성 데이터 $x_t$를 생성해야 하는 WaveNet의 모델에 사용됩니다.



 

 

WaveNet에서는 과거의 음성 데이터 $x_1, ..., x_{t-1} ,x_{t}$ 만을 이용하여 다음 음성 데이터 $x_t$를 생성해야 합니다.
따라서 과거 음성데이터만 접근할 수 있는 구조인 Causal Convolutional Network를 쌓아올린 구조로 과거의 정보에 접근합니다.
이런 구조는 Recurrent Network와 비슷한 효과를 갖을 수 있지만 더 빠르다는 장점을 갖고 있습니다.






위 조건부확률을 따르는 모델은 과거 데이터를 이용하여 다음 시점을

<center>$p(x)=\prod_{t=1}^T p(x_t|x_1, x_2, ..., x_{t-1}) $</center>
Wavenet의 음성 생성 가정은 조건부확률로 위와 같이 나타냅니다. 이는 모든 음성 데이터 $x_1, ..., x_{T-1} ,x_{T}$ 가 주어졋을 때 
 














## Reference
- [[BLOG]](https://medium.com/@satyam.kumar.iiitv/understanding-wavenet-architecture-361cc4c2d623) Understanding WaveNet architecture, Satyam Kumar
- [[BLOG]](https://ahnjg.tistory.com/94) WaveNet 이란?, JG Ahn
- [[PAPER]](https://www.eksss.org/archive/view_article?pid=pss-10-1-39) 한국어 text-to-speech(TTS) 시스템을 위한 엔드투엔드 합성 방식 연구, 최연주
- [[BLOG]](https://hyunlee103.tistory.com/54)  오디오 데이터 전처리, Hyunlee103
- [[BLOG]](https://hanseokhyeon.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90%EC%84%9C-librosa-%ED%8C%A8%ED%82%A4%EC%A7%80%EB%A1%9C-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EA%B7%B8%EB%A6%AC%EA%B8%B0) 파이썬 librosa 패키지로 스펙트럼 그리기, HanSeokhyeon
- [[YOUTUBE]](https://www.youtube.com/watch?v=GyQnex_DK2k) A Generative Model for Raw Audio, 모두의연구소
- [[YOUTUBE]](https://www.youtube.com/watch?v=nsrSrYtKkT8) Generative Model-Based Text-to-Speech Synthesis, Heiga Zen 
 
