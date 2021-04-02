---
layout:     post
title:      "[코드리뷰]타코트론2 TTS 시스템 2/2"
subtitle:   "타코트론2 개인화 TTS 시스템 만들기 2/2"
mathjax: true
audio_support: true
tags:
  - Text-to-Speech
  - Speech Synthesis
  - Deep Learning
---

# [코드리뷰] - 타코트론2 TTS 시스템 2/2

[**지난 글**](/2021/04/01/code-review/)에서는 TTS 시스템을 개발하기 위하여 데이터를 전처리하는 방법에 대해서 다루었습니다.
이번 글에서는 전처리된 데이터를 활용하여 Tacotron2 모델과 WaveGlow 모델을 학습시키는 방법에 대해서 말씀드리도록 하겠습니다.

딥러닝 아키텍처(타코트론2)에 대해 궁금하신 분이 계시다면 [**이전 글**](/2020/10/08/paper-review/) 또는 [**세미나 영상**](https://www.youtube.com/watch?v=BmD8OA9FGR0&list=PLetSlH8YjIfWk_PBAXKWqQM4pqzMMENrb&index=8) 을 참조하시기 바랍니다.

TTS 시스템을 개발하기 위하여 어떤 모델을 선택하느냐에 따라 내부 구성요소들이 크게 달라질 수 있습니다.
이 글처럼 Tacotron2를 활용하여 TTS 시스템을 구성할 때에는 Vocoder가 필요합니다.
Vocoder는 Tacotron2를 통해 추출된 Mel-Specotrogram 또는 Spectrogram을 음성(Raw Audio)로 변환하는 프로그램을 의미합니다.
이 글에서는 WavGlow를 Vocoder로 활용하는 방법에 대해서 다루도록 하겠습니다. 

![](/img/in-post/2021/2021-04-02/architecture_overview.png)
<center><b>TTS 아키텍처 Overview</b></center>

#### Short Summary
개인화 TTS 시스템을 만드는 과정을 크게 나누면 아래와 같습니다.

1. 데이터 수집
2. 음성 데이터 전처리
3. 스크립트 전처리
4. **Tacotron2 모델 학습**
5. WaveGlow 모델 학습

## 4. Tacotron2 모델
TTS 시스템을 만들기 위해 다양한 모델을 선택할 수 있지만 이 글에서는 Tacotron2 모델을 활용하는 방법에 대해 다루고 있습니다.
Tacotron2 모델을 직접 개발하는 것도 좋지만 현재 개발되어 <u>공개저장소(github)에 공개</u>된 프로그램을 활용하는 것이 더 빠르고 신뢰성 있는 모델을 개발할 수 있는 방법입니다.
대부분 영어만 지원하지만 한글 프로그램도 존재합니다.

Github에 공개된 유용한 프로그램을 알아보겠습니다. 

[**[1] Tacotron2-Wavenet-Korean-TTS(Pytorch)**](https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS)   
hccho2님께서 개발하신 Tacoton2(한글 지원) 모델입니다. 
다중화자를 지원할 수 있도록 기본 Taconton2 모델을 변경하였으며, Vocoder로 Wavenet을 구현하여 제공하고 있습니다.
개발된 모델에서 성능을 간접적으로 평가할 수 있는 샘플로 손석희 아나운서와 문재인 대통령의 음성을 제공하고 있는데, 매우 깔끔하게 음성이 생성된 것을 확인할 수 있습니다.
Pytorch로 개발되었기 때문에 Pytorch에 익숙한 유저분들에게 추천드립니다. 

![](/img/in-post/2021/2021-04-02/tacotron2_hccho2.png)
<center><b>hccho2님의 Tacotron2</b></center>

[**[2] Multi-Speaker-Tacotron(Tensorflow)**](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow)   
carpedm20님께서 개발하신 Tacotron(한글 지원) 모델입니다. 다중화자를 지원할 수 있도록 Tacotron 모델을 변경하였으며, 개발을 간편하게 할 수 있도록 프로그램을 잘 정리한 것이 특징입니다.
[DEVIEW 2017](https://youtu.be/klnfWhPGPRs) 에서 해당 내용을 발표하였으니 영상도 함께 참고하시기 바랍니다.
pre-trained Model Checkpoint를 함께 제공하고 있기 때문에 Checkpoint를 Base로 Transfer Learning하여 모델을 빠르게 학습시킬 수 있습니다.
Tensorflow로 개발되었기 때문에 Tensorflow에 익숙한 유저분들께 추천드립니다. 

![](/img/in-post/2021/2021-04-02/tacotron2_carpedm20.png)
<center><b>carpedm20님의 Tacotron</b></center>

[**[3] Nvidia-Tacotron2(Pytorch)**](https://github.com/NVIDIA/tacotron2)   
영어만 지원하는 Tacotron2 모델입니다. NVIDIA에서 만들었기 때문에 Apex, AMP 등을 지원하도록 프로그램을 구성하였습니다.
따라서 비교적 빠른 속도로 모델을 학습할 수 있으며 사용하기 간편합니다. 영어 Checkpoint(학습된 모델)을 제공하고 있으므로 영어 데모를 테스트해보기 간편합니다.
간단하게 활용하는 방법에 대해 리뷰를 제공하는 [블로그 글](https://jybaek.tistory.com/811) 이 있으니 해당글을 참고하시면 프로그램을 이해하기 더 수월합니다.

![](/img/in-post/2021/2021-04-02/nvidia_demo.png)
<center><b>NVIDIA TTS 데모 사이트</b></center>

## Tacotron2 모델을 이용하여 학습 예시
Tacotron2를 이용하여 모델을 학습하기 위해 아래 작업이 필요합니다.

1. 타코트론2 리소스(코드) 다운
2. 데이터 정보 파일 생성
3. 모델 파라미터 설정
4. 모델 학습하기

##### Step4.1 타코트론2 리소스 다운
이 글에서는 NVIDIA에서 제공하는 타코트론2 모델을 활용하여 한국어를 학습하는 방법에 대해 다루고자 합니다.
>그 이유는 NVIDIA에서 Tacotron2와 WavGlow를 결합하여 프로그램을 만드는 방법에 대해서 자세히 가이드를 제공하고 있으며, 데모 페이지에 업로드한 샘플 음성이 퍽 인상깊었기 때문입니다.
>또한 대부분의 깃허브에서 공유되는 Tacotron2모델은 Wavenet을 Vocoder로 활용하는 방법에 대해서 다루고 있는데, Wavenet 모델은 너무 무겁기(파라미터가 많음) 때문에 학습 및 추론하는데 오랜 시간이 필요합니다.
>WaveGlow는 Wavenet과 비교하면 매우 가벼운 모델이므로 활용성 측면에서 WaveGlow를 활용하고자 해당 Github 리소스를 선택하였습니다.  

원본 NVIDIA 타코트론2 코드를 [Git 관리 프로그램](https://git-scm.com/) 을 이용하여 다운 받는 방법은 아래와 같습니다. 
```bash
git clone https://github.com/NVIDIA/tacotron2.git
```

원본 NVIDIA 코드는 한국어를 지원하지 않기 때문에 약간의 코드 수정이 필요합니다.
코드 수정이 필요한 부분은 바로 한국어를 전처리하는 부분입니다.
한국어 전처리에는 숫자, 특수문자, 영어 등을 한글로 변환하는 작업 뿐만아니라 음절로 되어 있는 단위를 음소로 변환하는 작업을 포함합니다.
음소로 변환다는 것을 쉽게 설명하면 초성,중성,종성 으로 한글문자를 잘게 자르는 작업을 의미합니다.

![](/img/in-post/2021/2021-04-02/preprocess_overview.png)
<center><b>한글 전처리 과정</b></center>

일반적으로 음절은 초성 중성 종성으로 구성되어 있는데 음절을 단위로 보면 
그 이유는 초성

![](/img/in-post/2021/2021-04-02/character_combination.png)
<center><b>NVIDIA TTS 데모 사이트</b></center>

음절(Character) 단위로 다루면 

기본적으로  




