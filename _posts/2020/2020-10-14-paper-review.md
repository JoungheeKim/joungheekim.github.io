---
layout:     post
title:      "[논문리뷰]Grad-CAM"
subtitle:   "Learning Deep Features for Discriminative Localization"
mathjax: true
tags:
  - Image Localization
  - Vision
  - Deep Learning
---

# [논문리뷰] - [Grad-CAM : Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), ICCV 2017

인공지능은 이미 거의 모든 분야에서 다양한 용도로 사용되고 있습니다.
대부분 성능이 뛰어나지만 때로는 오작동을 합니다.
하지만 현재 대부분 AI는 딥러닝 기반 모델이므로 오작동하는 이유에 대해서 파악하기 힘들다는 단점을 갖고 있습니다.

![](/img/in-post/2020/2020-10-14/cam_sample.png)
<center>CAM 예시</center>

이러한 문제를 해결하고자 이미지 분야에서는 [Class Activation Mapping(CAM)](https://joungheekim.github.io/2020/09/29/paper-review/) 이라는 방법이 고안되었습니다.
CAM은 이미지 분류 모델로 사진을 분류할 때 사진의 어느 부분이 영향력을 끼치는지를 추출할 수 있는 방법론입니다.
다만 이 방법론을 사용하기 위해서는 [논문에서 제시한 것](https://arxiv.org/abs/1512.04150)처럼 모델의 구조 변경이 필요하기 때문에 이미 학습한 모델은 구조를 변경한 후 재학습이 필요합니다. 
또한 이러한 구조는 <u>성능을 희생이 필요</u>하다는 단점이 존재합니다.

다행히 CAM방법론이 제시된 다음해에 모델의 구조를 변경하지 않으면서 CAM을 사용할 수 있는 방법론이 ICCV, 2017 에서 공개되었습니다.
오늘 포스팅은 모델의 구조를 변경하지 않으면서 CAM을 사용할 수 있는 방법론에 대해 다룬 논문인 `Grad-CAM`을 리뷰하도록 하겠습니다.
이 글은 [Visual Explanations from Deep Networks via Gradient-based Localization 논문](https://arxiv.org/abs/1610.02391) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문 그대로를 리뷰하기보다는 *생각을 정리하는 목적으로 제작*하고 있기 때문에 실제 내용과 다른점이 존재할 수 있습니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.


#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. 모델의 구조를 변경하지 않으면서 <u>객체의 위치를 추출하는 방법</u>인 **Grad-CAM(Class Activation Mapping)** 을 제시합니다.
2. 

### CAM 적용방법
Grad-CAM에 대해 리뷰하기 전에 기존 CAM의 특징과 적용방법에 대해서 알아 보겠습니다.
CAM의 목표는 이미지 분류 문제를 학습한 모델을 이용하여 분류를 할 때 모델은 이미지의 어느 부분을 보는지를 표시한 Class Activation MAP을 추출하는 것입니다.
이는 학습된 모델의 weight($w_k^c$)와 이미지를 모델에 넣어 생성된 feature map($A_{i,j}^k$)을 이용하여 만들 수 있습니다.

![](/img/in-post/2020/2020-10-14/cam_example.png)
<center>CAM 구조 예시</center>

위 그림에서 처럼 이미지를 넣으면 모델의 forward pass에 따라 feature map($A_{i,j}^k$)이 생성됩니다.
이 feature map은 **Global Average Pooling(GAP)** layer와 연결되어 있고 이후 한번에 FC layer + softmax함수를 거쳐 각 class 확률을 출력합니다.
FC layer가 의미하는 것은 각 feature map($A_{i,j}^k$)이 class($c$)에 얼마나 영향을 미치는지를 나타내는 weight($w_k^c$)와 같습니다.
따라서 feature map($A_{i,j}^k$)와 weight($w_k^c$) 을 이용하여 좌표($i, j$)가 특정 class($c$)에 영향을 미치는 정도($M_{i,j}^c$)를 수식으로 나타내면 아래와 같습니다.

<center>$F^k = \frac{1}{Z} \sum_{i,j} A_{i,j}^k$</center>
<center>$Y^c = \sum_k w_k^c F^k$</center>
<center>$Y^c = \sum_k w_k^c \frac{1}{Z} \sum_{i,j} A_{i,j}^k$</center>
<center>$Y^c = \sum_{i,j} \frac{1}{Z} \sum_k w_k^c A_{i,j}^k$</center>
<center>$Y^c = \sum_{i,j} M_{i,j}^c$</center>

$A_{i,j}^k$ : feature map k의 가로($i$), 세로($j$)에 해당하는 값    
$F^k$ : feature $k$의 Global Average Pooling 값   
$Y^c$ : class $c$에 대한 score  
$k$ : feature map의 index    
$i, j$ : feature map의 가로, 세로 좌표    
$w_k^c$ : feature map $k$가 class $c$에 기여하는 weight  
$M_{i,j}^c$ : 좌표 $i$, $j$의 class $c$에 대한 영향력(class activation value)    

즉 CAM인 $M_{i,j}^c$는 weights($w_k^c$)와 $F_k$로 이루어져 있음을 알 수 있습니다.

### Grad-CAM 적용방법
위 수식처럼 CAM을 구하려면 특정 구조로부터 $w_k^c$가 추출되어야 합니다.
즉 GAP 이후 FC layer 한개만 연결되어 있는 형태여야만 위 수식을 이용하여 CAM을 추출할 수 있습니다.
따라서 CAM 논문에서는 CAM을 사용하기 위해서는 모델을 특정구조로 변경해야 한다고 명시합니다.

![](/img/in-post/2020/2020-10-14/grad_cam_example.png)
<center>Grad-CAM 구조 예시</center>

CAM과는 달리 Grad-CAM은 CNN을 사용한 일반적인 모든 구조에서 CAM을 활용할 수 있는 방법을 제시합니다.
바로 weights에 해당하는 부분을 gradient로 대채함으로써 모든 구조에서 특정 class에 feature map 미치는 영향력를 구할 수 있습니다.
Grad-CAM에서는 weights에 해당하는 gradient를 다음과 같이 정의합니다.

<center>$w_k^c = \sum_{i,j} \frac{\partial Y^c}{\partial A_{i,j}^k}$</center>
 
$Y^c$는 특정 class c의 score이고 이를 feature map의 각 부분($A_{i,j}^k$)에 대하여 미분한 다음 pixcel($i, j$)에 대하여 모두 더하면 feature map $k$가 class $c$에 미치는 영향인 $w_k^c$를 추출할 수 있습니다.
따라서 CAM에서 정의한 $M_{i,j}^c$를 다시 풀어쓰면 아래와 같습니다. 

<center>$M_{i,j}^c = \frac{1}{Z} \sum_k w_k^c A_{i,j}^k$</center>
<center>$M_{i,j}^c = \frac{1}{Z} \sum_k A_{i,j}^k \sum_{i,j} \frac{\partial Y^c}{\partial A_{i,j}^k}$</center>
<center>$M_{i,j}^c = \sum_k a_k^c A_{i,j}^k $</center>
<center>$a_k^c = \frac{1}{Z} \frac{\partial Y^c}{\partial A_{i,j}^k} $</center>
$a_k^c$ : $w_k^c$를 대채한 gradients

논문에서는 이미지에서 특정 클래스에 긍정적인 영향을 미치는 부분에만 관심이 있으므로 ReLU 비선형 함수를 적용하여 Grad-CAM을 구성한다고 합니다.
따라서 최종적으로 추출된 Grad-CAM의 식은 아래와 같습니다.

<center>$L_{Grad_CAM}^c = ReLU(M_{i,j}^c) = ReLU(\sum_k a_k^c A_{i,j}^k)$</center>


#### Grad-CAM 증명
Grad-CAM은 CAM의 일반화한 case입니다. 이를 증명하는 과정은 다음과 같습니다.
분류모델의 output인 class score($Y^c$)를 feature의 average pooling 값인 $F^k$로 미분하여 gradint를 나타내면 아래와 같은 $A_{i,j}^k$에 대한 식으로 표현됩니다.

<center>$\frac{\partial Y^c}{\partial F^k} = \frac{\frac{\partial Y^c}{\partial A_{i,j}^k}}{\frac{\partial F^k}{\partial A_{i,j}^k}}$</center>

$\frac{\partial F^k}{\partial A_{i,j}^k} = \frac{1}{Z}$ 이고, $\frac{\partial Y^c}{\partial F^k} = w_k^c$ 이므로 아래와 같은 식으로 변형이 가능합니다.

<center>$\frac{\partial Y^c}{\partial F^k} = \frac{\partial Y^c}{\partial A_{i,j}^k} \cdot Z$</center>
<center>$w_k^c = \frac{\partial Y^c}{\partial A_{i,j}^k} \cdot Z$</center>  

$Z$와 $w_k^c$는 pixcel($i, j$)와는 무관하므로 위의 식을 풀어서 쓰면 $w_k^c$와 gradient 사이의 관계로 변형이 가능합니다.

<center>$\sum_{i,j} w_k^c = \sum_{i,j} \frac{\partial Y^c}{\partial A_{i,j}^k} \cdot Z $</center>
<center>$\sum_{i,j} w_k^c = Z \sum_{i,j} \frac{\partial Y^c}{\partial A_{i,j}^k} $</center>

$Z=\sum_{i,j}1$을 이용하여 정리하면 $w_c^k$와 gradient의 관계식을 추출할 수 있습니다.

<center>$Z w_k^c = Z \sum_{i,j} \frac{\partial Y^c}{\partial A_{i,j}^k} $</center>
<center>$w_k^c = \sum_{i,j} \frac{\partial Y^c}{\partial A_{i,j}^k} $</center>

CAM에서 제안한 Global Average Pooling(GAP)가 적용된 구조에서 Grad-CAM을 구하는 것과 CAM을 구하는 것은 동일하다는 것을 알 수 있습니다.
즉 CAM은 특정 구조에서 Grad-CAM을 구하는 방법으로 해석할 수 있으므로 Grad-CAM은 CAM의 일반화한 방법이라고 논문에서 주장합니다.




## Reference
- [[BLOG]](https://github.com/jacobgil/pytorch-grad-cam) Grad-CAM implementation in Pytorch





  
Grad-CAM




Grad-CAM이란 CAM의 일반화 과정으로 이해 할 수 있습니다.

