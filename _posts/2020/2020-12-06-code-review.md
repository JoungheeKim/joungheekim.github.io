---
layout:     post
title:      "[코드리뷰]Bootstrapped DQN"
subtitle:   "Deep Exploration via Bootstrapped DQN"
mathjax: true
tags:
  - Ensemble Model 
  - Reinforcement Learning
  - Deep Learning
---

# [코드리뷰] - [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621), NIPS 2016

딥러닝, 강화학습 등이 사회에 보편적으로 알려져 있지 않던 2016년 3월 구글 딥마인드에서 만든 <b>알파고</b>와 당시 세계 바둑 랭킹 2위인 이세돌 9단과의 세기의 대결이 펼쳐졌습니다.
5국의 바둑 대결에서 놀랍게도 알파고가 이세돌 9단을 4대1로 이기고 우승을 차지하였습니다.
이후 많은 사람들이 당시 대결로부터 큰 충격과 영감을 받았으며 딥러닝과 강화학습의 가능성에 큰 기대를 갖고 투자를 시작하여 현재 해당 분야는 많은 발전을 이루었습니다.

![](https://lh3.googleusercontent.com/kFsqNQX_cQ2bLof_G_2UKCuSwDT34PcZpC8nNHDwaiGFteedmYbJODRnUXz8t_zdCemoWPAX_JxtkjfFHdBKhf819GDxiruo4HYYug=w1440-rw-v1)
<center><b>다양한 게임에 도전하는 알파고<a href="https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go">[(출처 : DeepMind Blog)]</a></b></center>

현재는 다양한 강화학습 모델이 존재하고 있으며 다양한 분야에 활용되고 있습니다. 강화학습의 종류는 **[여기](https://dreamgonfly.github.io/blog/rl-taxonomy/)** 를 참조하시기 바랍니다.
오늘 포스팅에서는 강화학습 모델 중 하나인 DQN(Deep Q Network)에 bootstrapping 방법을 적용한 Ensemble 모델인 `Bootstrapped DQN` 에 대해 다루도록 하겠습니다.
이 글은 **[Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)** 논문을 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 pytorch 라이브러리를 이용하여 코드를 구현한 내용을 자세히 설명드리겠습니다.
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 2가지는 아래와 같습니다.

1. 강화학습 모델 DQN을 변형한 앙상블 모델인 **Bootstrapped DQN**의 아키텍처를 제시합니다.
2. Unbalanced label 시계열 데이터에 **Unsupervised Anomaly Detection 방법론**을 적용하는 방법에 대해 제시합니다.

## 논문 리뷰

### DQN 개념
논문에서 활용한 base 모델인 DQN(Deep Q Network)에 대한 짧은 개념을 먼저 소개하고 논문리뷰를 시작하겠습니다.
DQN 모델을 이해하기 위해서는 먼저 강화학습이 다루는 문제가 어떤 것인지 살펴보겠습니다.

![](/img/in-post/2020/2020-12-06/reinforcement_learning_example.png)

강화학습에는 환경(Environment)과 에이전트(Agent)가 있습니다.
이 포스팅에서는 게임을 예제로 다룰 것이니 환경을 게임기라고 하고 에이전트를 사람 또는 딥러닝 에이전트라고 생각해 봅시다.
게임기는 매 시점 마다 현재 게임에서 어떤 상황인지를 파악할 수 있는 정보인 상태(State) 와 보상(Reward) 에 대한 정보를 제공합니다.
게임에서 상태는 게임화면이라고 볼 수 있고 보상은 획득점수 또는 깨진 블럭수(블럭깨기 게임)로 볼 수 있습니다. 

그 정보를 보고 에이전트는 매 시점에서 어떤 행위를 취해야 할지 결정하고 행동을 환경에 전달하게 됩니다.
게이머가 방향키 키보드를 누르는 행위를 행동의 예로 들 수 있습니다.

![](/img/in-post/2020/2020-12-06/reinforcement_learning_example2.png)

매 시점 보상이 생성되는 게임도 있지만 일반적으로 이벤트(블럭이 감소)가 발생한 특정시점에 생성됩니다.
강화학습의 목표는 매 시점 주어진 환경에서 보상을 최대한 많이 받을 수 있도록 에이전트를 만드는 것(학습)입니다.

강화학습 모델 중 DQN은 상태와 행동을 가치(Value)로 치환하는 함수 `Q`를 만들어 강화학습에 적용하는 방법론입니다.
이 방법은 매 시점 상태(게임화면)를 보고 특정 행위(왼쪽, 오른쪽)를 선택했을 때 얻을 수 있는 가치(Value)를 모델링 하는 방법입니다.
즉 특정 상태(게임화면)를 보고 특정 행위(왼쪽, 오른쪽)을 선택했을 때 얻을 수 있는 보상을 알 수 있다면 매 시점 가장 큰 미래의 보상을 보장하는 행위를 선택하면 되는 문제로 바뀔 수 있습니다.

![](/img/in-post/2020/2020-12-06/q_function.png)

현재 시점의 상태를 $s$라고 하고, 현재 시점에 취한 행동을 $a$라고 한다면 지금 상태($s$)에서 행동($a$)를 취했을 때 가치 `Q함수`는 $Q(s, a)$로 표현합니다.
현재 상태($s$)에서 행동($a$)을 취한 후 도달한 다음 시점의 상태를 $\grave{s}$ 라고 하고 다음 시점의 행동을 $\grave{a}$ 라고 한다면 상태($\grave{s}$)에서 행동($\grave{a}$)을 취했을 때 가치는 $Q(\grave{s}, \grave{a})$ 입니다.
현재 시점의 가치와 미래 시점의 가치를 이용하여 아래와 같이 표현할 수 있습니다.

<center>$Q(s,a) \cong R + Q(\grave{s}, \grave{a})$</center>

$R$ 은 상태($s$)에서 행동($s$)를 취했을 때 받을 수 있는 즉각적인 보상을 의미합니다.
보상은 게임에서는 블럭 한개를 깻을 때 받을 수 있는 점수와 같습니다.
즉 위의 식은 상태($s$)에서 행동($s$)을 취했을 때 받을 수 있는 가치는 즉각적인 가치와 미래 가치를 더한 것이라는 것을 의미합니다.

<center>가치 $\cong$ 즉각적인 보상 + 미래 가치</center>

다만 게임의 경우 빠른 시간에 보상을 얻어 게임을 끝내는 것이 더 좋은 결과이므로 미래 가치에는 시간에 따른 감가율($\gamma$)이 적용되어야 합니다.
또한 다음 시점의 상태($\grave{s}$)에서 일반적으로 가장 이득을 취할 수 있는 행동을 취할 것이므로 이를 반영하면 위의 식을 변형하여 아래와 같이 표현할 수 있습니다.   

<center>$Q(s,a) \cong R + \gamma \cdot max Q(\grave{s}, \grave{a})$</center>

`Q함수` 대한 정의를 하였고, 이제 위에서 정의한 것처럼 $Q(s,a)$ 가 $R + \gamma \cdot max Q(\grave{s}, \grave{a})$ 가까워지게 만들면 됩니다.
현재 $Q(s,a)$와 $R + \gamma \cdot max Q(\grave{s}, \grave{a})$ 의 차이에 학습율 $\alpha$를 곱하여 점진적으로 `Q함수`를 근사하는 것이 바로 DQN 강화학습의 목표입니다.
이는 아래와 같이 표현 할 수 있습니다.

<center>Q함수 = Q함수 + 비율 * 차이</center>
<center>$Q(s,a) = $Q(s,a) + \alpha( R + \gamma \cdot max Q(\grave{s}, \grave{a}) - Q(s, a) )$</center>

이제 실제 게임과 연결지어 앞서 설명한 내용을 적용하는 방법에 대해 생각해 보겠습니다. 

![](/img/in-post/2020/2020-12-06/game_data.png)

컴퓨터 또는 사람이 게임한 내용을 저장하고 있다고 가정합니다. 
게임한 내용은 특정 시점에서 화면($s$), 그 시점에서 조작한 행동($a$), 그 행동을 통해 생성된 보상($R$), 그 행동을 통해 다음 시점 변경된 게임화면($\grave{s}$)을 포함하고 있습니다.

![](/img/in-post/2020/2020-12-06/q_sample.png)

이 데이터를 이용하면 다음과 같이 쉽게 식을 구성할 수 있습니다.
이제 앞서 설명한 것처럼 `Q함수`를 잘 근사하는 모델을 만들고 학습하면 됩니다.

DQN에서는 `Q함수`를 Deep Neural Network를 이용하여 구성합니다.
DQN은 게임 화면($s$)를 입력으로 받고 각 행동($s$)에 따라 얼마만큼 가치가 있는지 추출할 수 있어야 합니다.

![](/img/in-post/2020/2020-12-06/q_architecture.png)




 `Q함수`



 $t$ 시점의 화면($s$)과 그것을 
즉 게임을 한 과거 데이터() 




감가율을 적용한 위의 식으
다만 미래 가치는 시간에 따라 









게임의 경우 


이를 이용하여 `Q함수`를 

다음 시점의 상태를 $\grave{s}$ 라고 하고 다음 시점의 행동을 $\grave{a}$ 라고 한다면 그 행동으로 

`Q함수`를 통해 도출 된 값은 현재 $s$에 대해 $a$를 선택했을 때 가치로 해석 할 수 있다.



`Q함수`가 특정 상태에서 특정 행위에 대한 가치를 반영하기 위해서는 미래의 가치를 고려하여야 한다.
즉 




미래 보상의 기댓값은




이를 모델링하는 방법은 다음과 같습니다.
<center>실제갗</center>


<center>$Q(s,a) = Q(s,a) + \alpha (R + \gamma max Q(\grave{s}, \grave{a} - Q(s,a)))$</center>




>좀 더 자세한 강화학습에 대해 알기 원하시는 분은 [강화학습 알아보기 BLOG](https://greentec.github.io/reinforcement-learning-second/) 를 방문하시기 바랍니다.


##### 1. 라이브러리 Import & 설치
``` python
import gym
import torch
```
모델을 구현하는데 필요한 기본 라이브러리를 Import 합니다.
Import 에러가 발생하면 해당 **라이브러리를 설치한 후 진행**해야 합니다.

강화학습에 환경에 해당하는 게임은 `GYM` 이라는 라이브러리를 사용함으로써 해결 할 수 있습니다.
`GYM`은 **OpenAI** 에서 제공하고 있는 가상환경 라이브러리 입니다. 
이 라이브러리는 다양한 환경을 제공하고 있으나 본 튜토리얼에서는 아타리 게임 중 하나인 breakout_4 를 활용하겠습니다.

`GYM` 라이브러리 설치 방법은 [[OpenAI 공식 사이트]](https://gym.openai.com/docs/) 에서 가이드를 제공 받을 수 있습니다.
하지만 해당 라이브러리는 Linux OS를 기반으로 Build 되었기 때문에 Windows OS 에서 활용하기 위해서는 Wrapper가 필요합니다.
Windows OS에서 GYM 라이브러리 설치하는 방법은 [[GYM 설치방법 안내]](https://talkingaboutme.tistory.com/entry/RL-Windows-10%EC%97%90%EC%84%9C-OpenAI-Gym-Baselines-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0) 를 참고하시기 바랍니다.

`GYM` 라이브러리를 정상정으로 설치한 후 게임이 잘 작동하는지 확인해 봅니다.
``` python
import gym
from PIL import Image

## 벽돌깨기 게임 환경 생성
env = gym.make('BreakoutDeterministic-v4')

## 환경 초기화
state = env.reset()

frames_game = []
for _ in range(30):
    ## render 함수를 이용하여 화면 불러오기
    img = env.render(mode='rgb_array')
    img = Image.fromarray(img)
    frames_game.append(img)
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    state = next_state

## 이미지를 gif로 저장
frames_game[0].save('play_breakout.gif', format='GIF', append_images=frames_game[1:], save_all=True, duration=0.0001, loop=0)

## 환경 종료
env.close()
```
![](/img/in-post/2020/2020-12-06/play_breakout.gif)
<center><b>GYM을 활용한 벽돌깨기 게임 예시</b></center>















## Reference
- [[PAPER]](https://arxiv.org/abs/1607.00148) Deep Exploration via Bootstrapped DQN, Osband at el
- [[BLOG]](https://greentec.github.io/reinforcement-learning-second/) 강화학습 알아보기(DQN)
- [[GITHUB]](https://github.com/johannah/bootstrap_dqn) Bootstrap DQN(Pytorch Implementation)