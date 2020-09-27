---
layout:     post
title:      "[논문리뷰]U-Net"
subtitle:   "Convolutional Networks for Biomedical Image Segmentation"
mathjax: true
tags:
  - Image Segmentation
  - Vision
  - Deep Learning
---

이미지로부터 객체(Object)를 추출하는 것은 컴퓨터 비전 분야에서 매우 중요한 Task 중 하나입니다.
예를들어 의료분야에서 종양 및 세포를 구분하고 표시하는 것은 의사들이 병을 진단을 할 때 큰 도움을 줄 수 있습니다.
또한 자율주행과 같이 Task에서는 물체를 구분하는 기능이 선 수행되어야 가능합니다.

![](https://divamgupta.com/assets/images/posts/imgseg/image14.png?style=centerme)
<center>Image Segmentation 예시<a href="https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html">(출처)</a></center>

오늘 포스팅은 객체를 인식하고 분류하는 다양한 방법 중에서 **픽셀 기반으로 이미지를 분할**하여 구분하는 모델인 `U-net` `UNet` 에 대해 상세하게 리뷰하도록 하겠습니다.
이 글은 [U-net 논문](https://arxiv.org/abs/1505.04597)과 [MEDIUM](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문 그대로를 리뷰하기보다는 *생각을 정리하는 목적으로 제작*하고 있기 때문에 실제 내용과 다른점이 존재할 수 있습니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. 넓은 범위의 이미지 픽셀로부터 의미정보를 추출하고 의미정보를 기반으로 각 픽셀마다 객체를 분류하는 U 모양의 아키텍처를 제시합니다.
2. 적은 학습 데이터로 모델의 성능을 올리기 위하여 생물 의학분야에서 효과적인 이미지 데이터 증가 방법을 제시합니다. 
3. 서로 근접한 객체 경계를 잘 구분하도록 학습하기 위하여 Weighted Loss를 제시합니다.

## 모델 구조

모델 구조는 크게 2가지로 나눌 수 있습니다.
1. 점진적으로 넓은 범위의 이미지 픽셀을 보며 의미정보(Context Information)을 추출하는 **수축 경로(contracting Path)**
2. 의미정보를 픽셀 위치정보와 결합(Localization)하여 각 픽셀마다 어떤 객체에 속하는지를 구분하는 **확장 경로(Expanding Path)**

모델의 Input은 이미지의 픽셀별 RGB 데이터이고 모델의 Output은 이미지의 각 픽셀별 객체 구분 정보(Class)입니다.
Convolution 연산과정에서 패딩을 사용하지 않으므로 모델의 Output Size는 Input Size보다 작습니다.
예를들어 100X100 크기의 이미지는    












 

## Reference
- [[BLOG]]([https://kuklife.tistory.com/118?category=872136) [Semantic Segmentation] Semantic Segmentation 목적
- [[BLOG]](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28an)U-Net 논문 리뷰 — U-Net: Convolutional Networks for Biomedical Image Segmentation, 강준영
- [[PAPER]](https://arxiv.org/abs/1505.04597) U-Net: Convolutional Networks for Biomedical Image Segmentation(2015), Olaf Ronneberger