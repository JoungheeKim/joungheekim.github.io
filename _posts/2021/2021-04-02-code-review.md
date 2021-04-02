---
layout:     post
title:      "[코드리뷰]타코트론2 TTS 시스템 2/2"
subtitle:   "타코트론2 개인화 TTS 시스템 만들기"
mathjax: true
audio_support: true
tags:
  - Text-to-Speech
  - Speech Synthesis
  - Deep Learning
---

# [코드리뷰] - 타코트론2 TTS 시스템 2/2

[**지난 글**](/2021/04/01/code-review/)에서는 TTS 시스템을 개발하기 위하여 데이터를 전처리하는 방법에 대해서 다루었습니다.
이번 글에서는 전처리된 데이터를 활용하여 **Tacotron2 모델과 WaveGlow 모델을 학습**시키는 방법에 대해서 말씀드리도록 하겠습니다.

딥러닝 아키텍처(타코트론2)에 대해 궁금하신 분이 계시다면 [**이전 글**](/2020/10/08/paper-review/) 또는 [**세미나 영상**](https://www.youtube.com/watch?v=BmD8OA9FGR0&list=PLetSlH8YjIfWk_PBAXKWqQM4pqzMMENrb&index=8) 을 참조하시기 바랍니다.

TTS 시스템을 개발하기 위하여 어떤 모델을 선택하느냐에 따라 내부 구성요소들이 크게 달라질 수 있습니다.
이 글처럼 Tacotron2를 활용하여 TTS 시스템을 구성할 때에는 Vocoder가 필요합니다.
**Vocoder**는 Tacotron2를 통해 추출된 Mel-Specotrogram 또는 Spectrogram을 <u>음성(Raw Audio)로 변환하는 프로그램</u>을 의미합니다.
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
TTS 시스템을 만들기 위해 다양한 모델을 선택할 수 있지만 이 글에서는 **Tacotron2 모델을 활용**하는 방법에 대해 다루고 있습니다.
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
2. 학습 정보 파일 생성
3. 모델 학습하기

##### Step4.1 타코트론2 리소스 다운
이 글에서는 NVIDIA에서 제공하는 타코트론2 모델을 활용하여 한국어를 학습하는 방법에 대해 다루고자 합니다.
>그 이유는 NVIDIA에서 Tacotron2와 WavGlow를 결합하여 프로그램을 만드는 방법에 대해서 자세히 가이드를 제공하고 있으며, 데모 페이지에 업로드한 샘플 음성이 퍽 인상깊었기 때문입니다.
>또한 대부분의 깃허브에서 공유되는 Tacotron2모델은 Wavenet을 Vocoder로 활용하는 방법에 대해서 다루고 있는데, Wavenet 모델은 너무 무겁기(파라미터가 많음) 때문에 학습 및 추론하는데 오랜 시간이 필요합니다.
>WaveGlow는 Wavenet과 비교하면 매우 가벼운 모델이므로 활용성 측면에서 WaveGlow를 활용하고자 해당 Github 리소스를 선택하였습니다.  

원본 NVIDIA 타코트론2 코드를 [Git 관리 프로그램](https://git-scm.com/) 을 이용하여 다운 받는 방법은 아래와 같습니다. 
```bash
git clone https://github.com/NVIDIA/tacotron2.git
```

원본 NVIDIA 코드는 한국어를 지원하지 않기 때문에 약간의 <u>코드 수정이 필요</u>합니다.
코드 수정이 필요한 부분은 바로 **한국어를 전처리**하는 부분입니다.
한국어 전처리에는 <u>숫자, 특수문자, 영어 등을 한글로 변환</u>하는 작업 뿐만아니라 음절로 되어 있는 단위를 <u>음소로 변환</u>하는 작업을 포함합니다.
음소로 변환다는 것을 쉽게 설명하면 초성,중성,종성 으로 한글문자를 잘게 자르는 작업을 의미합니다.

![](/img/in-post/2021/2021-04-02/preprocess_overview.png)
<center><b>한글 전처리 과정</b></center>

음절은 19개의 자음으로 구성된 초성, 21개의 모음으로 구성된 중성, 27개의 자음으로 구성된 종성으로 이루어져 있습니다.
즉 음소 조합으로 만들 수 있는 음절은 $19 \times 21 \times 27 = 10,773$개 입니다.
따라서 데이터가 충분하지 않다면 학습데이터에 많은 음절들이 누락되기 때문에 새로운 음절에 대해 모델이 대응하지 못하는 현상이 발생합니다.
이를 방지하고자 음소를 기본 입력 단위로 모델을 학습시킵니다. 

![](/img/in-post/2021/2021-04-02/character_combination.png)
<center><b>음절과 음소의 관계</b></center>

앞서 학습용 데이터(스크립트)는 직접(수동) 숫자, 특수문자, 영어등을 한글로 변경 했기 때문에 추가적인 전처리가 필요하지 않지만 
추론(inference) 단계에서 모델을 잘 작동시키기 위해 **전처리 코드를 추가**가 필요합니다.

<b>어떻게 한국어 전처리 코드를 추가해야 하는가?</b>

다행히 한국어 버전 Tacotron2인 hccho2 리소스와 영어 버전 Tacotron2인 Nvidia 리소스는 모두
[keithito 코드](https://github.com/keithito/tacotron) 를 기반으로 개발되었기 때문에 서로 호환이 가능합니다.
즉 hccho2 리소스에서 한국어 전처리 코드 부분을 복사하여 Nvidia 리소스에 붙여넣기 하는 방식으로 간단하게 한국어 전처리 코드를 추가할 수 있습니다.

![](/img/in-post/2021/2021-04-02/preprocess_korean.png)
<center><b>한국어 전처리 리소스 활용 방법</b></center>

NVIDIA Tacotron2 코드를 forked 한 뒤 위 그림과 같은 방법으로 전처리 리소스를 추가한 코드는 아래의 명령어를 통해 다운받을 수 있습니다. 
```bash
git clone https://github.com/JoungheeKim/tacotron2
```

##### Step4.2 학습 정보 파일 생성
학습 정보 파일이란 학습에 필요한 정보(음성 파일 위치, 스크립트)를 타코트론2 모델에게 전달하기 위해 정리한 문서를 의미합니다.
음성 파일의 위치와 해당 음성의 발화 스크립트로 구성된 병렬 데이터이며, 학습 및 검증용으로 train_data와 valid_data 2개가 필요합니다.

![](/img/in-post/2021/2021-04-02/train_information_example.png)
<center><b>학습 정보 파일 Format 예시</b></center>

생성한 학습 정보 파일을 타코트론2 프로젝터 내 filelists 폴더에 위치시킵니다.
> 설정파일(hparams.py)에서 학습 파일 위치를 설정하기 때문에 파일 위치는 변경 가능합니다.

![](/img/in-post/2021/2021-04-02/filelist_information_example.png)
<center><b>학습 정보 파일 생성 예시</b></center>

##### Step4.3 모델 학습하기
학습 정보 파일을 기반으로 타코트론2 모델을 학습시키기 간단한 명령어는 아래와 같습니다.

```bash
python train.py \
    --output_directory=output \
    --log_directory=log \
    --n_gpus=1 \
    --training_files=filelists/train_filelist.txt \ 
    --validation_files=filelists/train_filelist.txt \
    --epochs=500
```

