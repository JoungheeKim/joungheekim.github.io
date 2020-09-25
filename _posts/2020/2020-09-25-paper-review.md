---
layout:     post
title:      "[논문리뷰]Tacotron"
subtitle:   "Towards End-to-End Speech Synthesis"
mathjax: true
tags:
  - Text-to-Speech
  - Speech Synthesis
  - Deep Learning
---

# [논문리뷰] - [TACOTRON : Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135), DeepMind

TTS(Text to Speech)는 매우 복잡하며 긴 작업절차가 필요한 어려운 문제입니다. 
그 이유는 문장으로부터 음성을 생성하기 위하여 '문장을 음소로 나누고, 음소의 발음을 찾고, 음소와 음성의 위치를 매핑하는 등...' <u>다양한 작업이 필요</u>하기 때문입니다.
또한 각 <u>작업은 난이도가 높아</u> 전문가의 지식이 필요하고 <u>작업이 분리되어 따로 설계</u>되므로 이들을 합쳤을 때 발생하는 품질저하 사유를 찾기가 매우 힘들기 때문입니다.

하지만 드디어 이러한 복잡한 문제를 극복한 End-to-End 모델인 **타코트론** 논문이 공개되었습니다.
이 타코트론 모델은 문장으로부터 음성을 생성하기 위하여 <u>별도의 작업 없이</u> <문장, 음성> 쌍으로 이루어진 데이터가 있으면 학습이 가능하다는 특징을 갖고 있습니다.
따라서 이 논문이 나온 이후로 이제는 <u>전문가의 도움 없이도 개인이 음성을 합성</u>할 수 있게 되었습니다.

오늘 포스팅은 `타코트론` 또는 `Tacotron`라고 불리는 End-to-End 모델 대해 상세하게 리뷰하도록 하겠습니다.
이 글은 [Tacotron 논문](https://arxiv.org/abs/1703.10135)과 [Tacotron을 한국어에 적용한 논문](http://www.ndsl.kr/ndsl/search/detail/article/articleSearchResultDetail.do?cn=JAKO201811648108967) 을 참고하여 정리하였음을 먼저 밝힙니다. 
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 3가지는 아래와 같습니다.

1. **Attention 기반 Seq-to-Seq** TTS 모델 구조를 제시합니다.
2. <문장, 음성> 쌍으로 이루어진 데이터만으로 <u>별도의 작업없이</u> 학습이 가능한 **End-to-End 모델**입니다.  
3. 음성합성 품질 테스트(MOS)에서 높은 점수를 획득하였습니다. **합성품질**이 뛰어납니다.

## 모델 구조
![](/img/in-post/2020/2020-09-25/model_structure.gif)

모델은 크게 문장을 Input으로 받아 정보를 추출하는 **인코더**, 인코더로부터 추출된 정보를 이용하여 멜 스펙토그램을 생성하는 **디코더**, 
인코더의 정보를 디코더에 매핑해주는 *어텐션*, 마지막으로 디코더에서 생성된 <u>멜 스펙토그램</u>을 이용하여 Linear 스펙토그램을 생성하는 **후처리** 부분으로 나뉠 수 있습니다.
그리고 추가적으로 모델로부터 나온 최종 결과물인 Linear 스펙토그램을 오디오로 바꿔주는 **Grifin-Lim 알고리즘**이 있습니다.
인코더와 디코더 안에는 공통적으로 반복되는 CBHG 공통 모듈이 존재합니다. 

### 1) Input & Output

모델의 학습(Training) 및 추론(Inference) 단계에서 Input은 캐릭터 단위의 One-hot 벡터 입니다. 
따라서 영어 문장을 모델에 넣기 위해서는 문장을 캐릭터 단위로 나누고 One-hot Encoding하는 작업이 필요합니다.
예를들어 'I love you' 문장은 각각 한개의 캐릭터 'i', ' ', 'l', 'v'..., 'u' 로 나누고 One-hot Encoding을 통해 각 캐릭터에 맞는 숫자열[8, 6, 13, ..., 2]로 변형한 뒤 모델의 Input으로 사용합니다. 
한글의 경우 문장을 초성, 중성, 종성, 그리고 문장 부호로 나누어 총 80개의 캐릭터로 문자를 나누고 난 뒤 One-hot Encoding을 통해 숫자열로 변형합니다.
> 한글의 경우 초성과 종성의 자음은 각각 다른 캐릭터로 임베딩하여 처리합니다.
> 예를들어 나는 -> [ㄴ,ㅏ,ㄴ,ㅡ,ㄴ] -> [2, 4, 2, 5, 8]  으로 표현된 것처럼 'ㄴ'은 초성, 종성에 따라 다르게 임베딩됩니다.
> [[한국어 타코트톤 적용 참고자료]](https://aifrenz.github.io/present_file/Tacotron-AIFrenz-20190424.pdf)   

모델의 추론(Inference) 단계에서 Output은 Linear 스펙트로그램 입니다.
모델을 학습할 때에는 **후처리** 부분에서의 Ouput인 Linear 스펙토그램 뿐만 아니라 **디코더** 부분에서의 Output인 멜 스펙토그램을 함께 사용합니다.
즉 손실함수를 `Linear 스펙토그램 Loss + 멜 스펙토그램 Loss` 로 구성하여 학습합니다. 

> 모델에 사용하는 Linear 스펙트로그램은 Short-Time Fourier Transform 뿐만아니라 로그스케일, 노말라이징, 데시벨 스케일링 등의 다양한 전처리를 통해 추출됩니다. 
> 멜 스펙토그램은 Linear 스펙토그램을 Mel filter Bank라는 필터에 통과시켜 얻을 수 있습니다.
> 자세한 전처리 과정은 [[오디오 데이터 전처리]](https://hyunlee103.tistory.com/54) 에서 참고 부탁드립니다.

### 2) CBHG 모듈

CBHG 모듈은 인코더와 디코더에 공통적으로 존재하는 모듈로써 순차적인(Sequence) 데이터를 처리하는데 특화되어 있습니다.
**CBHG** 모듈은 1D **C**onvolution **B**ank, **H**ighway 네트워크, Bidirectional **G**RU로 구성되어 있습니다.
모듈은 Sequence 벡터를 Input으로 사용하며 Sequence 벡터가 Output으로 추출됩니다.
모듈의 상세 프로세스는 아래와 같습니다.

1. Sequence 데이터를 1부터 K개의 필터를 갖고 있는 **1D Convolution bank**에 통과시켜 Feature 벡터를 생성합니다.
2. Feature 벡터를 **Max polling Layer**에 통과시켜 Sequence에 따라 변하지 않는 부분(local invariance)을 추출합니다.
3. 고정된 폭을 갖은 몇개의 **1D Convolution Network**을 통과시켜 Sequence 데이터의 벡터 사이즈와 일치하는 벡터를 생성합니다.
4. 3)에서 생성된 벡터와 1)의 Sequence Input 벡터를 더하여 **Residual Connection**을 구성합니다.
5. 4)에서 생성된 벡터를 **Highway 네트워크**에 통과시켜 high-level features를 생성합니다.
6. high-level features를 **GRU**의 입력으로 사용합니다.

CBHG 모듈 안에 있는 4) Residual Connection은 모델의 성능 뿐만 아니라 빠르게 수렴할 수 있도록 돕는 역할을 합니다.
모든 1D Convolution Network는 Batch Normalization을 포함하고 있어 정규화 작용을 합니다.
참고로 모든 1D Convolution 안에는 Batch Normalization을 적용합니다.

#### Highway 네트워크
Hightway 네트워크는 Gate 구조를 사용하는 Residual 네트워크 입니다.


### 3) 인코더(Encoder)

인코더는 문장으로부터 고정된 길이의 특징(벡터)를 추출하는 것이 목적입니다.
따라서 앞서 설명한 것 처럼 캐릭터 단위로 나뉜 캐릭터 One-Hot 벡터가 인코더의 Input으로 들어와 어텐션 모듈에서 사용될 Sequence 벡터로 변환되는 과정은 아래와 같습니다.

1. 임베딩 매트릭스(Embedding Matrix)를 이용하여 One-Hot 벡터로 표현된 캐릭터 Input을 임베딩 벡터로 변환합니다.
2. 임베딩 벡터를 (FC layer + Lelu + Dropout)으로 구성된 Pre-net 모듈에 통과시킵니다.
3. Pre-Net을 통과시켜 생성된 벡터를 CBHG 넣으면 어텐션 모듈에서 활용될 Sequence 벡터가 생성됩니다.

Pre-Net에는 2층의 Fully Connected Layer(FC Layer) 입니다. 이 모듈은 과적합을 막기위한 목적으로 Dropout이 적용되어 있습니다.

### 4) 디코더(Decoder)

디코더는 인코더에서 생성된 Sequence 벡터와 $t-1$ 시점까지 생성된 디코더의 멜 스펙토그램을 Input으로 받아 $t$ 시점의 멜 스펙토그램을 생성합니다.
어텐션 기반 모델이므로 인코더에서 생성된 Sequence 벡터는 어텐션 모듈에서 계산된 가중치에 따라 가중합 되어 Decoder-RNN 모듈에 사용됩니다.
자세한 디코더 프로세스는 아래와 같습니다.

1. 디코더의 Input은 $t-1$ 시점까지 디코더에서 생성된 멜 스펙토그램입니다.
처음 시점에는 생성된 멜 스펙토그램이 없으므로 모든 값이 0인 멜 스펙토그램<Go 프레임>을 Input으로 사용합니다.
2. 멜 스펙토그램을 Pre-Net 모듈에 통과시켜 벡터를 생성 한 후 Attention-RNN의 Input으로 사용합니다.
3. Attention-RNN으로 부터 추출된 Sequence hidden 벡터($h_1, h_2, ..., h_{t-1}$)를 어텐센 모듈에 넣어 인코더의 벡터의 각 시점과 관련된 벡터의 가중합인 Context 벡터($c_1, c_2, ..., c_{t-1}$)를 추출합니다.
4. Attention-RNN hidden 벡터($h_1, h_2, ..., h_{t-1}$)와 Context 벡터($c_1, c_2, ..., c_{t-1}$)를 Concatenate 하여 Decoder-RNN의 Input으로 사용합니다.
5. Decoder-RNN에서 추출된 결과가 디코더의 Output인 $t$ 시점의 멜 스펙토그램입니다.

오디오로부터 추출한 멜 스펙토그램은 시간에 따라 연속한 프레임으로 구성되어 있습니다.
따라서 **겹치는 정보가 많으므로**(연속시점의 멜 스펙토그램은 반복되는 프레임으로 구성된 경우가 많음) 디코더에서 한 시점에서 여러개($\tau$)의 멜 스펙트그램을 추출하여 학습 및 수렴 속도를 상승시킵니다.
즉 총 $p$길이의 멜 스펙토그램이 존재하면 디코더에서 예측하는 총 시점은 $t=p/\tau$

&t& 시점에서 추출된 멜 스펙토그램







인코더는 문장으로부터 고정된 일련의 특징을 추출하기 위



우선 Short-Time Fourier Transform을 통해 오디오로부터  





모델의 Output은 디코더에서 생성되는 80 밴드 멜 스케일 스펙토그램과 후처리에서 생성되는 선형 스케일 스펙토그램으로 구성됩니다.













하지만 오늘 포스팅할 논문인 타코트론은 기존 TTS의 복잡한 문제를 End-to-End 모델로 만듬으로써 학습이 쉽고 좋은 음성품질을  

단점을 극복한 End-to-End 입니다.
즉 타코트론은 문장으로부터 음성을 생성하기 위하여 별도의 작업 없이 <문장, 음성> 쌍으로 이루어진 데이터가 있으면 학습이 가능하다는 특징을 갖고 있습니다.
  




WaveNet, DeepVoice와 같은  
하지만 이런 모델들과 비교하여 타코트론의 가장 큰 특징은 End-to-End 모델이라는 점 입니다.


이 논문이 나온 이후로 이제는 큰 전문가의 도움 없이도 머신러닝을 할 수 있는 개인이 음성합성을 할 수 있다는 점이 가장 매력적인 것 같습니다.
그것이 바로 오늘 포스팅할 모델인 `Tacotron` 또는 `타코트론` 이라고 End-to-End 음성합성 모델입니다.
이 글은 N-beats 논문과 Medium 글을 참고하여 정리하였음을 먼저 밝힙니다. 혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.













## Reference
- [[BLOG]](https://hcnoh.github.io/2018-12-11-tacotron) [Speech Synthesis] Tacotron 논문 정리
- [[BLOG]](https://medium.com/@shwetagargade216/text-to-speech-detailed-explanation-bfa021b5ca55) Text to speech Detailed Explanation
- [[SLIDE]](https://www.slideshare.net/carpedm20/deview-2017-80824162) 책 읽어주는 딥러닝: 배우 유인나가 해리포터를 읽어준다면 DEVIEW 2017
- [[PAPER]](http://www.ndsl.kr/ndsl/search/detail/article/articleSearchResultDetail.do?cn=JAKO201811648108967) 한국어 text-to-speech(TTS) 시스템을 위한 엔드투엔드 합성 방식 연구
- [[https://brightwon.tistory.com/11]](https://brightwon.tistory.com/11) MFCC 이해하기





   