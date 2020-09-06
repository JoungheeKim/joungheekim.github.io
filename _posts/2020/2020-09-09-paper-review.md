---
layout:     post
title:      "작업중"
subtitle:   "N-BEATS"
tags:
  - time-series
  - univariate
  - multiple step prediction 
---

# [논문리뷰] - [NEURAL BASIS EXPANSION ANALYSIS FOR INTERPRETABLE TIME SERIES FORECASTING](https://arxiv.org/abs/1905.10437), ICLR 2020

2018년 세계적인 시계열(Time-Series) 경진대회인 [M4 Competition](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785) 가 개최되었습니다.
지금까지 대회에서 1등을 하던 모델은 항상 통계기반 모델이었지만 그 대회에서 1등을 차지한 모델은 [ES-RNN(Exponential Smoothing Long Short Term Memory networks)](https://arxiv.org/abs/1907.03329) 으로 통계적 방법론과 머신러닝 방법론을 잘 섞은 구조의 모델입니다.
그런데 그 ES-RNN보다 더 좋은 예측 성능을 보이는 순수 머신러닝 방법론이 등장하였습니다.
그것이 바로 오늘 포스팅할 모델인 `N-BEATS`이라고 불리는 단변량 예측 모델입니다.

#### Short Summary
이 논문의 2가지 부분에서 기여점이 있습니다.
1. 딥러닝 아키텍처 : 통계적 접근법을 전혀 사용하지 않고 시계열 문제(Time-Series)에서 뛰어난 성과를 보인 딥러닝 아키텍처를 제시합니다.
2. 해석이 가능한 구조 : "계절적 트렌드" 와 같은 특징들을 분석할 수 있는 해석이 가능한 딥러닝 결과물을 도출합니다.  

## 모델 구조
![](/img/in-post/2020/2020-09-09/model_architect.png)

전체구조는 동일하나 **해석이 가능한 구성요소**를 포함시키는지 여부에 따라 Generic Architecture 와 Interpretable architecture 로 나뉩니다.

### Input & Output
관측된 시점을 $t$ 시점이라고 가정하면 모델로부터 나오는 Output은 길이 H의 예측값 $[t+1, t+2, ... t+H]$ 이고, Input은 길이가 nH(n은 hyper-parameter) 관측값 $[t-nH, ..., t-1, t]$ 입니다.
본 논문에서는 $n$을 2~7로 설정하여 Input의 길이를 2H~9H로 활용합니다. 

### 모델 구성요소
Basic Block과 Stack Block이 단계별로 구성되어 있습니다.

#### Basic Block
Basic Block은 총 4개의 FC layer와 









## Reference
 - [[BLOG] N-BEATS — Beating Statistical Models with Pure Neural Nets, Neo Yi Peng](https://towardsdatascience.com/n-beats-beating-statistical-models-with-neural-nets-28a4ba4a4de8)
 - [[REPO] N-Beats Pytorch, Keras Implemntation](https://github.com/philipperemy/n-beats) 