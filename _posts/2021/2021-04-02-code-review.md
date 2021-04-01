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

[지난 글](/2021/04/01/code-review/)에서는 TTS 시스템을 개발하기 위하여 데이터를 전처리하는 방법에 대해서 다루었습니다.
이번 글에서는 전처리된 데이터를 활용하여 Tacotron2 모델과 WaveGlow 모델을 학습시키는 방법에 대해서 말씀드리도록 하겠습니다.

딥러닝 아키텍처(타코트론2)에 대해 궁금하신 분이 계시다면 [**이전 글**](https://joungheekim.github.io/2020/10/08/paper-review/) 또는 [**세미나 영상**](https://www.youtube.com/watch?v=BmD8OA9FGR0&list=PLetSlH8YjIfWk_PBAXKWqQM4pqzMMENrb&index=8) 을 참조하시기 바랍니다.

#### Short Summary
개인화 TTS 시스템을 만드는 과정을 크게 나누면 아래와 같습니다.

1. 데이터 수집
2. 음성 데이터 전처리
3. 스크립트 전처리
4. Tacotron2 모델 개발
5. WaveGlow 모델 개발

