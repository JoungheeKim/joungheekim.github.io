---
layout:     post
title:      "[코드리뷰]Face Recognition Using KPCA"
subtitle:   "Face Recognition Using Kernel Principal Component Analysis"
mathjax: true
tags:
  - PCA
  - Vision
  - Kernel-based learning
---

# [코드리뷰] - [Face Recognition Using Kernel Principal Component Analysis](https://arxiv.org/abs/1207.3538), 2002

오늘 리뷰는 이미지에서 Kernel-PCA를 사용하여 얼굴의 특징점을 추출하고 SVM을 이용하여 서로다른 얼굴을 분류하는 논문을 리뷰하겠습니다.
이 글은 [**Face Recognition Using Kernel Principal Component Analysis 논문**](https://arxiv.org/abs/1502.04681) 과 [**고려대학교 강필성 교수님의 강의**](https://www.youtube.com/watch?v=6Et6S03Me4o&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=14) 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 sklearn 라이브러리를 이용하여 <u>코드를 구현</u>한 후 자세하게 설명드리겠습니다.
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 2가지는 아래와 같습니다.

1. 이미지로부터 특징점을 추출하는 방법으로 Kernel-PCA를 적용합니다.
2. SVM을 활용하여 이미지 특징점을 학습하고 얼굴을 분류하는 알고리즘을 구축합니다. 

## 모델 학습 과정
논문에서 제시하는 얼굴인식 학습 과정은 크게 2가지로 나뉩니다.
1. **FACE FEATURE EXTRACTION** : kerenl-PCA를 이용하여 이미지 특징점 추출과정
2. **FACE RECOGNITION** : linear-SVM을 이용하여 이미지를 분류하는 과정

#### [1] Face Feature Extraction
본 논문에서는 이미지 특징점을 추출하기 위하여 kernel-PCA를 이용합니다. 
따라서 PCA와 kernel-PCA에 대해 간단하게 설명드리겠습니다.

PCA는 데이터의 분산을 최대한 보존하면서 저차원 공간으로 변환하는 기법입니다. 
따라서 의미없는 정보를 버리고 의미있는 정보만을 추출하는 방법으로 사용됩니다.
일반적으로 공분산에서 고유벡터(eigenvector)와 고유값(eigenvalue)를 추출한 뒤 n개의 고유벡터만을 활용하여 입력을 재구성함으로써 PCA를 적용할 수 있습니다.   

kernel-PCA는 PCA에 kernel trick을 적용한 알고리즘 입니다.
kernel-PCA는 non-linear 매핑함수 $\phi$를 활용하여 input $x$를 고차원 공간으로 매핑한 다음 일반적인 linear-PCA를 적용합니다.

##### Kernel PCA Procedure
covariance의 정의에 따라 kernel covariance matrix는 아래과 같이 표현됩니다.
covariance matrix는 feature M×M 차원으로 표현됩니다.  

<center>$C^{\phi} = \frac{1}{M} \sum^M_{i=1} ( \phi(x_i) - m^{\phi} ) (\phi(x_i) - m^{\phi} )^T$</center>
$\phi$ : 매핑함수  
$C^{\phi}$ : kernel covariance matrix    
$x_i$ : i 번째 데이터  
$\phi(x_i)$ : 고차원으로 공간으로 매핑된 i 번째 데이터  
$m^{\phi}$ : 고차원으로 공간으로 매핑된 데이터의 평균  
$M$ : feature space의 차원  
$N$ : 데이터 갯수  

기존 covariance matrix의 형태에서 x를 고차원 공간으로 매핑한 함수 $\phi$를 적용한 모습과 같습니다.
사영된 공간에서 평균은 0이라는 가정을 하면 좀 더 쉬운 수식으로 아래와 같이 표현할 수 있습니다.

<center>$m^{\phi} = \frac{1}{N} \sum^N_{i=1} \phi(x_i) = 0$</center>
<center>$C^{\phi} = \frac{1}{N} \sum^N_{i=1} \phi(x_i) \phi(x_i)^T$</center>

수학적 정의에 따라 covariance matrix($C^{\phi}$)의 eigenvalue($\lambda_k$)와 eigenvectors($v_k$)는 아래와 같이 식이 성립합니다.

<center>$C^{\phi} v_k = \lambda_k v_k$</center>

위 두 수식을 이용하여 다음과 같은 식을 구성할 수 있습니다.
또한 주성분 $v_k$ 는 데이터의 선형결합으로 표현되므로 아래와 같은 식으로 표현할 수 있습니다.
 
<center>$\frac{1}{N} \sum^N_{i=1} \phi(x_i) (\phi(x_i)^T v_k) = \lambda_k v_k$</center>
<center>$v_k = \frac{1}{N} \sum^N_{i=1} \alpha_{ki} \phi(x_i)$</center>

$v_k$를 정리하면 식을 아래와 같이 정리할 수 있습니다.

<center>$\frac{1}{N} \sum^N_{i=1} \phi(x_i) \phi(x_i)^T \sum^N_{j=1} \alpha_{kj} \phi(x_j) = \lambda_k \sum^N_{i=1} \alpha_{kj} \phi(x_i)$</center>

위 식에 kernel fuctnion을 적용하기 위하여 $\phi(x_l)$를 양변에 곱하고 약간의 변형하여 식을 재구성합니다.

<center>$\frac{1}{N} \sum^N_{i=1} \phi(x_l)^T \phi(x_i) \sum^N_{j=1} \alpha_{kj} \phi(x_i)^T \phi(x_j) = \lambda_k \sum^N_{i=1} \alpha_{kj} \phi(x_l)^T \phi(x_i)$</center>

이제 kernel trick을 이용하여 매핑함수 $\phi$와 관련된 식을 정리하면 아래와 같이 표현할 수 있습니다.

<center>$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$</center>





위 식을 



<center> \frac{1}{N} \sum^N_{i=1} \phi(x_i) \phi(x_i)^T \sum^N_{j=1} \alpha_{kj} \phi(x_j) = \lambda_k \sum^N_{i=1} \alpha_{kj} \phi(x_i) $</center>



<center>$m^{\phi} = \frac{1}{N} \sum^N_{i=1} \phi(x_i) = 0$</center>



## Reference
- [[PAPER]](https://arxiv.org/abs/1207.3538) Face Recognition Using Kernel Principal Component Analysis, 2002
- [[BLOG]](https://www.geeksforgeeks.org/ml-face-recognition-using-pca-implementation/) Face Recognition Using PCA Implementation, 
- [[YOUTUBE]](https://www.youtube.com/watch?v=6Et6S03Me4o&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=14) Kernel-based Learning - KPCA, 강필성



