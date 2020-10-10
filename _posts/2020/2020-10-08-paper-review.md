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

- **Task1** :  텍스트로부터 Mel-spectrogram을 생성하는 단계
- **Task2** : Mel-spectrogram으로부터 음성을 합성하는 단계

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

Attention은 매 시점 Deocder에서 사용할 정보를 Encoder에서 추출하여 가져오는 역할을 합니다.
즉 Attention Mechanism은 Encoder의 LSTM에서 생성된 feature와 Decoder의 LSTM에서 전 시점에서 생성된 feature를 이용하여 Encoder로 부터 어떤 정보를 가져올지 alignment하는 과정을 의미합니다.  
타코트론2 모델은 [Location Sensitive Attention](https://paperswithcode.com/method/location-sensitive-attention#) 을 사용합니다.
Location Sensitive Attention 이란 Additive attention mechanism([Bandau Attetnion]((https://hcnoh.github.io/2018-12-11-bahdanau-attention)))에 attention alignment 정보를 추가 한 형태입니다.

##### Additive Attention
<center>$s_{t, i} = w^{T}\tanh\left(Wd_{t-1} + Vh_{i} + b\right)$</center>
<center>$\alpha_{t, i} = \frac{exp(s_{t, i})}{\sum_{i=1}^n exp(s_{t, i})}$</center>
<center>$\alpha_{t} = [\alpha_{t, 1}, \alpha_{t, 2}, ... \alpha_{t, n}]$</center>
<center>$c_{t} = \sum a_{t, i}h_{i} = \alpha_{t}h$</center>
$W, V$ : 학습이 가능한 matrix weights  
$w, b$ : 학습이 가능한 bector weights    
$h_{i}$ : Encoder bi-LSTM에서 생성된 $i$번째 feature
$d_{t}$ : Decoder LSTM에서 생성된 $t$번째 feature
$s_{t, i}$ : $t$ 시점에서 hidden $i$ 에 대한 attention score
$\alpha_{t, i}$ : $t$ 시점에서 hidden $i$ 에 대한 alignment(0~1)
$c_{t}$ : $t$시점에서 Attetnion 모듈로 부터 추출한 context vector

Addictive Attention 은 Encoder RNN으로부터 생성된 feature($h$)와 Decoder RNN의 한 step 전 결과물($d_{t-1}$) 을 이용하여 attention alignment($\alpha_{t}$)를 구합니다. 

##### Location Sensitive Attention
<center>$s_{t, i} = w^{T}\tanh\left(Wd_{t-1} + Vh_{i} + Uf_{t, i} + b\right)$</center>
<center>$f_{i} = F ∗ \alpha_{i−1}$</center>
$U$ : 학습이 가능한 matrix weights  
$U$

Location Sentitive Attention은 이전 시점($t-1$)에서 생성된 attention alignment($\alpah_{t-1}$)를 이용하여 다음 시점($t$) Attention alignment($\alpah_{t}$)를 구할 때 추가로 고려한 형태입니다. 
k개의 filter를 갖고 있는 1D convolution을 이용하여 Attention alignment($\alpah_{t-1}$)를 확장하여 $f_{i}$ matrix를 생성합니다.
이후 학습이 가능한 weights($U$)와 내적한 후 Addictivae attention의 구성에 포함하여 계산합니다.

#### 1.4 Decoder
![](/img/in-post/2020/2020-10-08/decoder.png)
<center><b>Decoder 상세 구조 예시</b></center>

Decoder는 Attention을 통해 얻은 alignment feature와 이전 시점에서 생성된 mel-spectrogram 정보를 이용하여 다음 시점 mel-spectrogram을 생성하는 역할을 합니다.
Decoder는 Pre-Net, Decoder LSTM, Projection Layer, Post-Net으로 구성됩니다.

Pre-Net은 2개의 Fully Connected Layer(256 dim) + ReLU 으로 구성되어 있습니다.
이전 시점에서 생성된 mel-spectrogram이 decoder의 input으로 들어오면 가장먼저 Pre-Net을 통과합니다.
Pre-Net은 bottle-neck 구간으로써 중요 정보를 거르는 역할을 합니다.

Decoder LSTM은 2개의 uni-directional LSTM Layer(1024 dim) 으로 구성되어 있습니다. 
Pre-Net을 통해 생성된 vector와 이전 시점($t-1$)에서 생성된 context vector($c_{t-1}$)를 합친 후 Decoder LSTM을 통과합니다.
Decoder LSTM은 Attention Layer의 정보와 Pre-Net으로부터 생성된 정보를 이용하여 특정 시점($t$)에 해당하는 정보를 생성합니다.

Decoder LSTM에서 생성된 매 시점($t$) vector는 두개로 분기되어 처리됩니다.
1. 종료 조건의 확률을 계산하는 경로
2. mel-spectrogram을 생성하는 경로

종료 조건의 확률을 계산하는 경로는 Decoder LSTM으로부터 매 시점 생성된 vector를 Fully Connected layer를 통과시킨 후 sigmoid 함수를 취하여 0에서 1사이의 확률로 변환합니다.
이 확률이 Stop 조건에 해당하며 사용자가 설정한 threshold를 넘을 시 inference 단계에서 mel-spectrogram 생성을 멈추는 역할을 합니다. 

mel-spectrogram을 생성하는 경로는 Decoder LSTM으로부터 매 시점 생성된 vector와 Attention에서 생성된 context vector를 합친 후 Fully Connected Layer를 통과시킵티다.
이렇게 생성된 mel-vector는 inference 단계에서 Decoder의 다음 시점의 input이 됩니다.

Post-Net은 5개의 1D Convolution Layer로 구성되어 있습니다.
Convolution Layer는 512개의 filter와 5×1 kernel size를 가지고 있습니다.
이전 단계에서 생성된 mel-vector는 Post-Net을 통과한 뒤 다시 mel-vector와 구조(Residual Connection)로 이루어져 있습니다.
Post-Net은 mel-vector를 보정하는 역할을 하며 타코트론2 Task1의 최종 결과물인 mel-spectrogram의 품질을 높이는 역할을 합니다.

      


 
  
 



>논문에서 Attention과 관련하여 자세한 구조를 설명하고 있지 않습니다.
>따라서 [구현체](https://github.com/BogiHsu/Tacotron2-PyTorch/blob/master/model/model.py) 를 보고 자료를 구성하였습니다.
>


 







## Reference
- [[BLOG]](https://medium.com/spoontech/tacotron2-voice-synthesis-model-explanation-experiments-21851442a63c) Tacotron2 voice synthesis model explanation & experiments, Ellie Kang
- [[BLOG]](https://m.blog.naver.com/PostView.nhn?blogId=designpress2016&logNo=221183754859&proxyReferer=https:%2F%2Fwww.google.com%2F) Tacotron2 Practical Use
- [[PAPER]](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE07614110&mark=0&useDate=&bookmarkCnt=0&ipRange=N&accessgl=Y&language=ko_KR) Tacotron2 기반 한국어 음성 합성 모델 개발과 한국어에 맞는 Hyper-parameter 탐색
- [[BLOG]](https://medium.com/@rajanieprabha/tacotron-2-implementation-and-experiments-832695b1c86e) Tacotron-2 : Implementation and Experiments

- [[BLOG]](https://hcnoh.github.io/2018-12-11-bahdanau-attention) Bahdanau Attention 개념 정리
- [[BLOG]](https://hcnoh.github.io/2019-01-01-luong-attention) Luong Attention 개념 정리

- [[GITHUB]](https://github.com/BogiHsu/Tacotron2-PyTorch) Tacotron2 Pytorch Implementation


(https://medium.com/a-paper-a-day-will-have-you-screaming-hurray/day-7-natural-tts-synthesis-by-conditioning-wavenet-on-mel-spectogram-predictions-tacotron-2-bbcce354a3e3)