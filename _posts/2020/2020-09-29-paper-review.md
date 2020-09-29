---
layout:     post
title:      "[논문리뷰]CAM"
subtitle:   "Learning Deep Features for Discriminative Localization"
mathjax: true
tags:
  - Image Localization
  - Vision
  - Deep Learning
---

# [논문리뷰] - [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150), CVPR 2016

<u>이미지 분야</u>에서 자주 쓰이는 딥러닝 네트워크의 순위를 매기면 항상 상위권에 위치한 모델이 바로 **Convolution Neural Network(CNN)**입니다.
이 CNN 구조가 나온 이후로 다양한 모델에 적용되면서 이미지 딥러닝 분야는 꾸준히 발전하고 있습니다.
그 이유는 CNN이 이미지의 **지역적 특징**을 잘 포착하기 때문입니다.

![](/img/in-post/2020/2020-09-29/cam_example.png)
<center>이미지에서 객체 추출(Feat. 고양이 줄리)</center>

오늘 포스팅은 CNN을 이용하여 지역적 특징을 잘 포착하는지 여부에 대해 **해석(시각화)이 가능한 방법** 를 제시한 논문을 리뷰하도록 하겠습니다.
이 글은 [Learning Deep Features for Discriminative Localization 논문](https://arxiv.org/abs/1512.04150)과 [MEDIUM 글](https://towardsdatascience.com/learning-deep-features-for-discriminative-localization-class-activation-mapping-2a653572be7f?gi=f6a5717f2f12) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문 그대로를 리뷰하기보다는 *생각을 정리하는 목적으로 제작*하고 있기 때문에 실제 내용과 다른점이 존재할 수 있습니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. **Global Average Pooling layer(GAP)**를 적용하여 <u>해석(시각화)가 가능한 구조</u>를 제시합니다.
2. <u>객체의 위치를 추출하는 방법</u>인 **Class Activation Mapping(CAM)** 을 제시합니다.
3. 객체의 위치정보 없이 <u>카테고리 정보<u/>만을 학습한 후 객체 위치를 추출하여 **Localization Test**에서 좋은 평가점수를 획득하였습니다.

## 모델 구조

### 일반적인 분류 모델 구조
![](/img/in-post/2020/2020-09-29/formal_structure.png)
<center>일반적인 이미지 분류 모델 구조 예시</center>

일반적인 이미지 분류 모델의 구조를 간단하게 위 그림에서 표현하였습니다.
모델의 구조는 크게 **특징을 추출**하는 Feature Extraction 단계와 추출된 특징을 이용하여 **이미지를 분류**하는 Classification 단계로 구분됩니다.
Feature Extraction을 통해 생성된 Feature Map은 3차원 벡터(채널,가로,세로)이므로 이를 2차원 벡터(채널,특징)로 변경하는 Flatten 단계를 수행하여 Classification 단계의 Input으로 활용합니다.
Classification 단계는 여러층의 Fully Connected Layer(FC Layer)로 구성되어 있으며 마지막 FC Layer에서 이미지를 분류합니다.
Feature Extraction 단계에서 추출한 Feature Map은 <u>여러층의 FC Layer를 통과</u>할 때 **위치정보가 소실**되므로 이 구조는 객체의 위치정보를 추출할 수 없습니다.         

### 객체 위치 추출이 가능한 모델 구조 

![](/img/in-post/2020/2020-09-29/suggest_structure.png)
<center>GAP가 적용된 이미지 분류 모델 구조 예시</center>

카테고리 정보만을 학습하여 모델이 객체의 위치 추출 능력을 갖추기 위하여 본 논문에서는 Flatten 단계에서 Global Average Pooling(GAP) 방법을 사용해야 한다고 주장합니다.
또한 Fully Connected Layer의 수를 줄이고 마지막 Classification Layer 하나만을 이용하여 모델을 구성합니다.
위 구조에서는 Global Average Pooling을 이용하여 각 Feature Map($f_k(x,y)$)의 가로 세로 값을 평균하여 특징변수($F_k) k개를 생성합니다.
예를들어 위 그림에서는 총 4개의 Feature Map이 존재하므로 총 4개의 특징변수가 생성됩니다.

<center>$\sum_{x,y} f_k(x,y) = F_k$</center>
<center>$S_c=\sum_k w_k^c F_k$</center>
<center>$P_c=\frac{exp(S_c)}{\sum_c exp(S_c)}$</center>
$f_k(x,y)$ : Feature Map k의 가로(x), 세로(y)에 해당하는 값  
$F_k$ : 특징변수 k  
$k$ : Feature Map의 index  
$x, y$ : Featrue Map의 가로, 세로 위치  
$w_k^c$ 특징변수 k가 클래스 c에 기여하는 Weight  

이 특징변수($F_k$)와 FC Layer의 Weight($w_k^c$)를 곱하여 더하면 각 클래스의 점수($S_c$)를 계산할 수 있습니다.
각 특징변수에 곱해진 Weight는 각 Feature Map이 해당 클래스에 얼마나 기여하는 지를 나타냅니다.
마지막으로 클래스 점수에 SoftMax 함수를 취하면 각 클래스로 분류될 확률($P_c$)을 계산할 수 있습니다.

#### Class Activation Mapping(CAM)
위 식을 응용하면 각 클래스 분류될 확률에 영향을 미친 객체의 위치 위치(x,y)를 추출할 수 있습니다.

<center>$S_c=\sum_k w_k^c F_k$</center>
<center>$S_c=\sum_k w_k^c \sum_{x,y} f_k(x,y)$</center>
<center>$S_c=\sum_k w_k^c \sum_{x,y} f_k(x,y)$</center>
<center>$S_c=\sum_{x,y} \sum_k w_k^c f_k(x,y)$</center>






식을 이용하여 표현하면 위의 내용과 동일합니다.







  



#### Global Average Pooling
<center>$S_c &=& \sum_k w_k^c F_k$</center>
<center>$&=& \sum_k w_c^k \sum_{x, y}f_k(x,y)$</center>
<center>$&=& \sum_{x, y} \sum_k w_k^c f_k(x,y)$</center>
 
 
  


위 그림은 일반적인 이미지 인식 모델(AlexNet, AGGnet, GoogLeNet)에서 Feature를 추출하는 부분에 Global Average Pooling Layer를 적용한 모습을 설명한 것입니다.
전반부는 여러 층의 Conv Layer로 이루어진 이미지 인식 모델이 Feature Map을 추출하는 부분입니다.
추출한 여러층의 Feature Map을 Global Average Pooling Layer를 이용하여 한 층마다  
 모형이고 Global Average Pooling Layer를 적용한 후 Fully Connected Layer를 적용합니다. 
 
 Max pooling도 위치정보를 보존하면서 특징정보를 추출할 수 있지만 모든 Feature Map의 정보를 활용하지 않으므로 객체의 Boundary에 집중하여 정보를 추출한다
 반면에 Global Average Pooling은 Feature Map의 모든정보를 포함하여 정보를 추출하기 때문에 객체의 위치정보를 더 명확히 구분할 수 있는 능력을 부여합니다.
  
  방법이지만 Global Average Pooling






 직역적 특징을 잘 포착하는지 여부를 확인할 수 있는   

이 꾸준히 발전하였고 CNN의 기능에 대해   

딥러닝을 공부하는 



해석이 가능한 AI를 Explainable AI 라고 부릅니다. Neural Network 