---
layout:     post
title:      "[코드리뷰]Face Recognition Using with KPCA"
subtitle:   "Face Recognition Using Kernel Principal Component Analysis"
mathjax: true
tags:
  - PCA
  - Vision
  - Kernel-based learning
---

# [코드리뷰] - [Face Recognition Using Kernel Principal Component Analysis](https://arxiv.org/abs/1207.3538), 2002

오늘 리뷰는 이미지에서 Kernel-PCA를 사용하여 얼굴의 특징점을 추출하고 SVM을 이용하여 서로다른 얼굴을 분류하는 논문을 리뷰하겠습니다.
이 글은 [Face Recognition Using Kernel Principal Component Analysis 논문](https://arxiv.org/abs/1502.04681) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 sklearn 라이브러리를 이용하여 <u>코드를 구현</u>한 후 자세하게 설명드리겠습니다.
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 2가지는 아래와 같습니다.

1. 이미지로부터 특징점을 추출하는 방법으로 Kernel-PCA를 적용합니다.
2. SVM을 활용하여 이미지 특징점을 학습하고 얼굴을 분류하는 알고리즘을 구축합니다. 

## 모델 학습 과정
논문에서 제시하는 얼굴인식 학습 과정은 크게 2가지로 나뉩니다.
1. **FACE FEATURE EXTRACTION** : Kerenl-PCA를 이용하여 이미지 특징점 추출과정
2. **FACE RECOGNITION** : Linear-SVM을 이용하여 이미지를 분류하는 과정

#### [1] FACE FEATURE EXTRACTION
본 논문에서는 이미지 특징점을 추출하기 위하여 Kernel-PCA를 이용합니다. 
따라서 PCA와 Kernel-PCA에 대해 간단하게 설명드리겠습니다.

PCA는 데이터의 분산을 최대한 보존하면서 저차원 공간으로 변환하는 기법입니다. 
따라서 의미없는 정보를 버리고 의미있는 정보만을 추출하는 방법으로 사용됩니다.
일반적으로 공분산에서 고유벡터(eigenvector)와 고유값(eigenvalue)를 추출한 뒤 n개의 고유벡터만을 활용하여 입력을 재구성함으로써 PCA를 적용할 수 있습니다.   

Kernel-PCA는 PCA에 kernel trick을 적용한 알고리즘 입니다.
Kernel-PCA는 input $x$를 non-linear 매핑함수 $\pi$를 활용하여 고차원 공간으로 매핑한 다음 일번적인 linear PCA를 적용하는 것입니다.

##### Kernel PCA Procedure
Kernel Covariance Matrix는 다음과 같습니다.
<center>$C = \frac{1}{M} \sum^M_{i=1} \pi(x_i) \cdot \pi(x_i)$</center>

<center>$m^{\pi} = \frac{1}{N} \sum^N_{i=1} \pi(x_i) = 0$</center>



## Reference
- [[PAPER]](https://arxiv.org/abs/1207.3538) Face Recognition Using Kernel Principal Component Analysis, 2002

