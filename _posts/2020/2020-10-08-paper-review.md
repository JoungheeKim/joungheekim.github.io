---
layout:     post
title:      "[논문리뷰]Tacotron2"
subtitle:   "Natural TTS Synthesis By Conditioning WAVENET On Mel Spectrogram Predictions"
mathjax: true
tags:
  - Text-to-Speech
  - Speech Synthesis
  - Deep Learning
---

# [논문리뷰] - [TACOTRON2 : Natural TTS Synthesis By Conditioning WAVENET On Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884v2), ICASSP 2018

WaveNet, Tacotron 등 딥러닝 방법론이 적용되면서 최근 몇년간 TTS(Text to Speech)은 빠르게 발전하였습니다.
따라서 이제는 복잡한 작업 절차 없이 데이터를 학습하여 텍스트로부터 고품질의 음성을 생성할 수 있게 되었습니다.

그에 따라 오늘날 음성합성은 다양한 방향으로 적용되고 있습니다.
예를들어 네이버에서는 "유인나가 읽어주는 오디오북" 서비스를 출시하였고 이제는 배우 유인나의 목소리로 다양한 이야기를 들을 수 있게 되었습니다.
또한 유투버들은 각각 개인의 특성을 살린 음성합성 프로그램(TTS)을 만들어 음성 도네이션 목소리로 활용하고 있습니다.

오늘 포스팅에서는 현재시점으로 대중적인 TTS(Text to Speech)모델인 `타코트론2` or `Tacotron2`에 대해 상세하게 리뷰하겠습니다. 
이 글은 [Tacotron2 논문](https://arxiv.org/abs/1712.05884v2) 을 참고하여 정리하였음을 먼저 밝힙니다.
논문 그대로를 리뷰하기보다는 **생각을 정리하는 목적으로 제작**하고 있기 때문에 <u>논문 이외의 내용을 포함하고 있으며</u> 사실과 다른점이 존재할 수 있습니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. **Attention 기반 Seq-to-Seq** TTS 모델 구조를 제시합니다.
2. <문장, 음성> 쌍으로 이루어진 데이터만으로 <u>별도의 작업없이</u> 학습이 가능한 **End-to-End 모델**입니다.  
3. 음성합성 품질 테스트(MOS)에서 높은 점수를 획득하였습니다. **합성품질**이 뛰어납니다.


### 모델 전체 구조
![](/img/in-post/2020/2020-10-08/train_process.png)
<center><b>모델 학습 단계 예시</b></center>

모델은 텍스트를 받아 음성을 합성합니다. 따라서 최종 Input과 Output은 텍스트와 음성입니다.
하지만 텍스트로부터 바로 음성을 생성하는 것은 어려운 Task이므로 타코트론2 논문에서는 TTS를 두단계로 나누어 처리합니다.

1. 텍스트로부터 Mel-spectrogram을 생성하는 단계
2. Mel-spectrogram으로부터 음성을 합성하는 단계

첫번째 단계는 Sequence to Sequence 딥러닝 구조의 **타코트론2 모델**이 담당하고 있습니다.
두번째 단계는 Vocoder로 불리며 타코트론2 논문에서는 **WaveNet 모델**을 변형하여 사용합니다.

### 1 타코트론2(Seq2Seq)
타코트론2 모델의 input은 character이고, output은 Mel-Spectrogram 입니다.
모델은 크게 Encoder, Decoder, Attention 모듈로 구성되어 있습니다.
Encoder는 character를 일련 길이의 hidden 벡터로 변환하는 작업을 담당합니다.
Attention은 Encoder에서 생성된 일정길이의 hidden 벡터로부터 시간순서에 맞게 정보를 추출하여 Decoder에 전달하는 역할을 합니다.
Decoder는 Attention에서 얻은 정보를 이용하여 Mel-spectrogram을 생성하는 역할을 담당합니다.

#### 1.1 전처리
모델을 학습하기 위해서 input과 output(label)이 한쌍으로 묶인 데이터가 필요합니다.
텍스트와 음성이 쌍으로 묶인 데이터가 있다면 이를 모델의 input과 output(label)의 형태로 가공하여야 합니다.
즉 텍스트는 character로 만들어야 하고 음성은 Mel-spectrogram으로 변형해야 합니다.

우선 영어라면 띄어쓰기가 포함하여 알파벳으로 나누는 작업을 통해 텍스트를 character로 변경할 수 있습니다.
예를들어 'I love you' 텍스트는 character sequence 즉 'i', ' ', 'l', 'v'..., 'u' 로 변경 됩니다.
이후 one-hot Encodding을 적용하여 character sequence를 정수열로 변경한 뒤 모델의 input으로 활용합니다. 
>타코트론2에 한글을 적용한 구현체의 대부분이 텍스트를 character로 변활할 때에는 음절을 이용합니다. 
>즉 초성(19개), 중성(21개), 종성(27개)으로 나눕니다.
>초성과 종성은 자음으로 구성되어 있어 동일하게 표기될 수 있으나 구분하여 활용합니다.  
>[[한국어 타코트톤 적용 참고자료]](https://aifrenz.github.io/present_file/Tacotron-AIFrenz-20190424.pdf)   

![](/img/in-post/2020/2020-10-08/audio_preprocess.png)
<center><b>오디오 데이터 전처리 예시</b></center>

음성데이터로부터 Mel-Spectrogram을 추출하여 output으로 활용하기 위하여 총 3가지 전처리 작업이 필요합니다.

1. Short-time Fourier Transform(STFT)
2. 80개의 mel filterbank를 이용하여 Mel scaling
3. Log Transform

오디오데이터에는 여러개의 오디오(frequency)가 섞여 있으므로 여러개의 오디오를 분리하여 표시하기 위하여 Fourier Transform을 활용합니다.
다만 모든데이터에 Fourier Transform을 적용하면 시간에 따른 오디오의 변화를 반영할 수 없으므로 sliding window를 이용하여 오디오를 특정길이로 잘라 Fourier Transform을 적용합니다. 
이 결과물을 spectrogram이라고 하며 오디오로부터 spectrogram을 만드는 과정을 short-time Fourier Transform이라고 지칭합니다.

두번째 단계는 spectrogram에 mel-filter bank라는 비선형 함수를 적용하여 저주파(low frequency) 영역을 확대하는 작업입니다.
사람의 귀는 고주파보다 저주파에 민감하므로 저주파의 영역을 확대하고 고주파의 영역을 축소하여 Feature로 사용합니다.
이는 더 명료한 음성을 생성하기 위하여 Feature를 사람이 쉽게 인지 가능한 scale로 변환하는 작업입니다.  
 
이후 log를 취해 amplitude 영역에서의 log scaling을 진행하면 mel-spectrogram이 생성되며 모델의 output(label)으로 활용합니다.

#### 1.2 Encoder
![](/img/in-post/2020/2020-10-08/encoder.png)
<center><b>Encoder 상세 구조 예시</b></center>

Encoder는 Character 단위의 one-hot verctor를 encoded feature로 변환하는 역할을 합니다. 
Encoder는 Character Embedding, 3 Convolution Layer, Bidirectional LSTM으로 구성되어 있습니다.

input으로 one-hot vector로 변환된 정수열이 들어오면 Embedding matrix를 통해 512차원의 embedding vector로 변환됩니다.
embedding vecotor는 3개의 conv-layer(1d convolution layer + batch norm)를 지나 bi-LSTM layer로 들어가서 encoded feature로 변환됩니다. 

#### 1.3 Attention
![](/img/in-post/2020/2020-10-08/attention.png)
<center><b>Attention 상세 구조 예시</b></center>

Attention 구조는 기본 Bandau Attetnion에 alingment 정보를 추가한 형태입니다.

<center>$s_{t, i} = w^{T}\tanh\left(Wd_{t-1} + Vh_{i} + b\right)$</center>
<center>$a_{t, i}=\frac{exp(s_{t, i})}{\sum_{i=1}^n exp(s_{t, i})}$</center>
<center>$a_{t}=[a_{t, 1}, a_{t, 2}, ... a_{t, n}]$</center>
$W, V$ : 학습이 가능한 Matrix Weights  
$w, b$ : 학습이 가능한 Vector Weights  
$$
 


 







## Reference
- [[BLOG]](https://medium.com/spoontech/tacotron2-voice-synthesis-model-explanation-experiments-21851442a63c) Tacotron2 voice synthesis model explanation & experiments, Ellie Kang
- [[BLOG]](https://m.blog.naver.com/PostView.nhn?blogId=designpress2016&logNo=221183754859&proxyReferer=https:%2F%2Fwww.google.com%2F) Tacotron2 Practical Use
- [[PAPER]](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE07614110&mark=0&useDate=&bookmarkCnt=0&ipRange=N&accessgl=Y&language=ko_KR) Tacotron2 기반 한국어 음성 합성 모델 개발과 한국어에 맞는 Hyper-parameter 탐색
- [[BLOG]](https://medium.com/@rajanieprabha/tacotron-2-implementation-and-experiments-832695b1c86e) Tacotron-2 : Implementation and Experiments

- [[BLOG]](https://hcnoh.github.io/2018-12-11-bahdanau-attention) Bahdanau Attention 개념 정리
- [[BLOG]](https://hcnoh.github.io/2019-01-01-luong-attention) Luong Attention 개념 정리


(https://medium.com/a-paper-a-day-will-have-you-screaming-hurray/day-7-natural-tts-synthesis-by-conditioning-wavenet-on-mel-spectogram-predictions-tacotron-2-bbcce354a3e3)