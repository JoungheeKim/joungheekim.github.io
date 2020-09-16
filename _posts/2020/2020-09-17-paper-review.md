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

2016년 [딥마인드(DeepMind)](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) 에서 딥러닝 기반 오디오 생성모델에 관한 논문을 공개하였습니다.  
오늘은 이 딥러닝 기반 End-to-End 오디오 생성 모델인 `WaveNet` 을 포스팅하겠습니다.
이 글은 [WaveNet 논문](https://arxiv.org/abs/1609.03499) 과 [Medium 글](https://medium.com/@satyam.kumar.iiitv/understanding-wavenet-architecture-361cc4c2d623) 을 참고하여 정리하였음을 먼저 밝힙니다. 
또한 논문을 이해하기 위하여 필요한 내용을 외부 코드 및 글을 정리하여 추가하였으므로 포스팅한 글은 논문의 내용만을 담고 있지 않습니다. 
제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 4가지는 아래와 같습니다.

1. WaveNet은 자연스러운 오디오 파형을 직접 생성합니다.
2. 긴 오디오 파형을 학습하고 생성할 수 있는 새로운 구조를 제시합니다.  
3. 학습된 모델은 컨디션 모델링으로 인해 다양한 특징적인 음성을 생성할 수 있습니다.
4. 음악을 포함한 다양한 오디오 생성분야에서도 좋은 성능을 보입니다.

## 모델 구조
모델의 Input과 Output은 파형입니다. 일반적인 

### 1) Input & Output
![](/img/in-post/2020/2020-09-17/analog_to_digital_conversion.png)
<center>Figure 1 : 오디오 데이터를 디지털 데이터로 변환과정</center>

오디오 데이터는 연속형(Continous) 아날로그 데이터입니다. 이 오디오 데이터를 컴퓨터에서 처리하거나 저장(`.wav`, `.mp4`)하려면 **디지털 데이터**로 변환해야 합니다.
이 변환하는 과정을 [Analog Digital Conversion](https://hyunlee103.tistory.com/54) 라고 부르며 표본화(Sampling), 양자화(Quantizing)로 구성되어 있습니다.
Analog Digital Conversion 과정을 통해 처리된 오디오 데이터는 이산형(Discrete) 디지털 데이터로 변환되어 정수배열(Integer Array)로 표현됩니다.
이 정수배열이 WaveNet의 Input과 Output으로 활용됩니다.

### 2) Modeling
WaveNet은 확률론적 모형(Probabilistic Model)으로써 T개의 배열로 구성된 오디오 데이터 $x_1, ..., x_{T-1} ,x_{T}$ 열이 주어졌을 때 음성으로써 성립할 확률 $P(x_1, ..., x_{T-1} ,x_{T})$ 을 학습하여 이후 생성에 활용합니다.
이 확률은 각 음성 데이터들의 조건부 확률을 이용하여 아래와 같이 표현될수 있습니다.
<center>$P(x_1, ..., x_{T-1} ,x_{T})=P(x_1, ..., x_{T-1}) \dot$ P(x_{T}|x_1, ..., x_{T-1})$</center>
<center>$=P(x_1, ..., x_{T-2}) \dot$ P(x_{T-1}|x_1, ..., x_{T-2}) \dot$ P(x_{T}|x_1, ..., x_{T-1})$</center>
<center>$=\prod_{t=1}^T P(x_t|x_1, x_2, ..., x_{t-1})$</center>


<center>$p(x)=\prod_{t=1}^T p(x_t|x_1, x_2, ..., x_{t-1}) $</center>
Wavenet의 음성 생성 가정은 조건부확률로 위와 같이 나타냅니다. 이는 모든 음성 데이터 $x_1, ..., x_{T-1} ,x_{T}$ 가 주어졋을 때 
위 조건부확률을 따르는 모델은 $t$를 기준으로 과거의 음성 데이터 $x_1, ..., x_{t-1} ,x_{t}$ 을 이용하여 한 시점 뒤 음성 데이터 $x_{t+1}$을 생성할 수 있습니다.
 














## Reference
- [[BLOG]](https://medium.com/@satyam.kumar.iiitv/understanding-wavenet-architecture-361cc4c2d623) Understanding WaveNet architecture, Satyam Kumar
- [[BLOG]](https://ahnjg.tistory.com/94) WaveNet 이란?, JG Ahn
- [[PAPER]](https://www.eksss.org/archive/view_article?pid=pss-10-1-39) 한국어 text-to-speech(TTS) 시스템을 위한 엔드투엔드 합성 방식 연구, 최연주
- [[BLOG]](https://hyunlee103.tistory.com/54)  오디오 데이터 전처리, Hyunlee103
- [[BLOG]](https://hanseokhyeon.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90%EC%84%9C-librosa-%ED%8C%A8%ED%82%A4%EC%A7%80%EB%A1%9C-%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC-%EA%B7%B8%EB%A6%AC%EA%B8%B0) 파이썬 librosa 패키지로 스펙트럼 그리기, HanSeokhyeon 
