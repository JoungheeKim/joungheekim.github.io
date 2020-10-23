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

# [코드리뷰] - [Face Recognition Using Kernel Principal Component Analysis](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=991133), 2002

오늘 리뷰는 이미지에서 Kernel-PCA를 사용하여 얼굴의 특징점을 추출하고 SVM을 이용하여 서로다른 얼굴을 분류하는 논문을 리뷰하겠습니다.
이 글은 [**Face Recognition Using Kernel Principal Component Analysis 논문**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=991133) 과 [**고려대학교 강필성 교수님의 강의**](https://www.youtube.com/watch?v=6Et6S03Me4o&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=14) 참고하여 정리하였음을 먼저 밝힙니다.
논문을 간단하게 리뷰하고 sklearn 라이브러리를 이용하여 <u>코드를 구현</u>한 후 자세하게 설명드리겠습니다.
혹시 제가 잘못 알고 있는 점이나 보안할 점이 있다면 댓글 부탁드립니다.

#### Short Summary
이 논문의 큰 특징 2가지는 아래와 같습니다.

1. 이미지로부터 특징점을 추출하는 방법으로 Kernel-PCA를 적용합니다.
2. SVM을 활용하여 이미지 특징점을 학습하고 얼굴을 분류하는 알고리즘을 구축합니다. 

## 모델 구조
![](/img/in-post/2020/2020-11-01/model_flow.png)
<center><b>모델 상세 구조</b></center>

논문에서 제시하는 얼굴인식 분류 과정은 크게 2가지로 나뉩니다.

1. **Face Feature Extraction** : kerenl-PCA를 이용하여 <u>이미지 특징점 추출</u>과정
2. **Face Recognition** : linear-SVM을 이용하여 이미지를 <u>class로 분류</u>하는 과정

이미지 특징점 추출하는 단계에서는 **kernel-PCA**를 활용합니다.
분류단계에서는 **linear-SVM**과 **Neural Network**를 활용합니다.

### [1] Face Feature Extraction
본 논문에서는 이미지 특징점을 추출하기 위하여 **kernel-PCA**를 이용합니다. 
따라서 PCA와 kernel-PCA에 대해 간단하게 설명드리겠습니다.

![](/img/in-post/2020/2020-11-01/pca_example.png)
<center><b>PCA 예시</b></center>

PCA는 데이터의 분산을 <u>최대한 보존하면서 저차원 공간으로 변환</u>하는 기법입니다. 
따라서 의미없는 정보를 버리고 **의미있는 정보만을 추출**하는 방법으로 사용됩니다.
일반적으로 공분산에서 고유벡터(eigenvector)와 고유값(eigenvalue)를 추출한 뒤 n개의 고유벡터만을 활용하여 입력을 재구성함으로써 PCA를 적용할 수 있습니다.   

kernel-PCA는 PCA에 kernel trick을 적용한 알고리즘 입니다.
kernel-PCA는 non-linear 매핑함수 $\phi$를 활용하여 input $x$를 고차원 공간으로 매핑한 다음 일반적인 linear-PCA를 적용합니다.

##### Kernel PCA Procedure
![](/img/in-post/2020/2020-11-01/kpca_example.png)
<center><b>Kernel-PCA 예시</b></center>

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
<center>$\frac{1}{N} \sum^N_{i=1} K(x_l, x_i) \sum^N_{j=1} \alpha_{kj} K(x_i, x_j) = \lambda_k \sum^N_{i=1} \alpha_{kj} K(x_l, x_i)$</center>
<center>$K^2 = \lambda N K \alpha_k$</center>

이로써 Kernel PCA는 아래와 같이 구할 수 있습니다.

<center>$y_k(x) = \phi(x)^T v_k = \sum^N_{i=1} \alpha_{ki} K(x, x_i)$</center>

### [2] Face Recognition
![](/img/in-post/2020/2020-11-01/face_recognition_example.png)
<center><b>분류 단계 상세 구조</b></center>

kernel-PCA를 통해 추출한 특징점을 이용하여 이미지를 분류하는 단계입니다.
본 논문에서는 2가지 모델을 차례로 이용하여 특징점을 각 class로 분류합니다.

1. **점수 추출** : Linear-SVM으로부터 각 <u>class에 대한 점수를 추출</u>하는 단계
2. **정규화 과정** : Neural Network 를 활용하여 각 **class에 대한 점수를 Normalize**하고 최종 결과를 도출하는 단계

각 class에 대한 점수를 추출하기 위하여 Linear-SVM을 구성합니다.
분류할 대상은 2개 이상의 class를 갖고 있으므로 SVM을 class 갯수(N)만큼 구성합니다.
kernel-PCA로부터 추출한 특징을 N개의 SVM에 넣어 각 class에 대한 점수를 추출합니다.

추출한 점수를 (Fully Connected Layer + Tanh) 로 구성된 2개 층의 Neural Network에 넣습니다.
Neural Network로부터 최종 결과인 각 Class에 대한 Normalize 점수가 추출됩니다. 
이미지가 해당 class에 속할 경우 1로 해당하지 않을 경우 -1로 label을 구성하고 Neural Network의 output과의 차이로부터 **MSE Loss로 계산하여 학습**하면 SVM으로부터 추출된 점수보다 정규화한 점수를 Neural Network에서 추출할 수 있습니다.

## 코드 구현

<p style="text-align: center;"><b><i class="fa fa-exclamation-triangle" aria-hidden="true"></i> 주의 <i class="fa fa-exclamation-triangle" aria-hidden="true"></i></b></p>  
튜토리얼은 pytorch, numpy, torchvision, easydict, tqdm, matplotlib, celluloid, tqdm 라이브러리가 필요합니다.
2020.11.01 기준 최신 버전의 라이브러리를 이용하여 구현하였고 이후 **업데이트 버전에 따른 변경은 고려하고 있지 않습니다.**
<u>Jupyter로 구현한 코드를 기반</u>으로 글을 작성하고 있습니다. 따라서 tqdm 라이브러리를 python 코드로 옮길때 주의가 필요합니다.

#### 데이터
![](/img/in-post/2020/2020-11-01/olivetti_sample.png)
<center><b>Olivetti faces 데이터 예시</b></center>

튜토리얼에서 사용하는 데이터는 [The Labeled Faces in the Wild face recognition dataset](http://vis-www.cs.umass.edu/lfw/) 입니다.
Olivetti 데이터는 총 40명의 인물이 등장하며 각 인물에 대해 10개의 이미지로 구성되어 있습니다.
각 이미지는 서로 다른 시간에 촬영되었습니다.
또한 촬영시 제약사항이 없으므로 이미지에는 다양한 얼굴표정, 안경 착용 등 독특한 특징이 나타날 수 있습니다.
데이터는 `sklearn` package를 통해서 다운 받을 수 있으므로 편의상 라이브러리를 활용합니다.

##### 1. 라이브러리 Import
``` python


```
모델을 구현하는데 필요한 라이브러리를 Import 합니다.
Import 에러가 발생하면 반드시 해당 **라이브러리를 설치한 후 진행**해야 합니다.

##### 2. 데이터 불러오기
``` python
olive_data = fetch_olivetti_faces()

## 이미지 데이터
n_samples, h, w = olive_data.images.shape
data_size = h*w
X = olive_data.images.reshape(-1, data_size)
print("이미지 갯수[{}],  이미지 가로[{}],  이미지 세로[{}]  가로X세로[{}]".format(n_samples, h, w, data_size))

## Label 데이터
y = olive_data.target
n_classes = len(set(y))
print("클래스 개수[{}]".format(n_classes))
```
이미지의 갯수는 400개 입니다. 1개의 channel과 64×64 픽셀 크기의 데이터입니다.

``` python
## 데이터 시각화  
def plot_gallery(images, item_index, n_col = 3, h=64, w=64, title=None): 
    if len(item_index) < n_col:
        n_row = len(item_index)
    
    plt.figure(figsize =(1.8 * n_col, 2.4))
    item_index = item_index[:n_col]
    #plt.subplots_adjust(bottom = 0, left =.01, right =.99, top =.90, hspace =.35) 
    for i in range(n_col): 
        plt.subplot(1, n_col, i + 1) 
        plt.imshow(images[item_index[i]].reshape(h, w), cmap = plt.cm.gray) 
        plt.xticks(()) 
        plt.yticks(())
    if title is not None:
        plt.suptitle(title)
        
## 얼굴 선택
sample_index = 1
n_col = 3
itemindex = np.where(y==sample_index)[0]

## 얼굴 plot
plot_gallery(X, itemindex, n_col, h, w)
```
![](/img/in-post/2020/2020-11-01/visulization_sample.png)
<center><b>Olivetti faces 데이터 시각화</b></center>

이미지를 시각화하여 API를 통해 데이터를 잘 불러왔는지 확인합니다.

##### 3. Kernel-PCA & Linear-SVM

이미지를 input으로 사용하여 각 class로 구분하기 까지 kernel-PCA, SVM, Neural Network 알고리즘을 구성해야 합니다.
각 알고리즘은 다양한 hyper-parameter 갖고 있으므로 알맞는 hyper-parameter 탐색이 필요합니다.
우선적으로 Kernel-PCA와 SVM 두개의 알고리즘을 이용하여 Pipe-Line을 구성하고 Grid Search를 활용하여 각 알고리즘의 hyper-paramter 탐색을 진행합니다. 

``` python
## Pipe Line 구성
pipe = Pipeline([('pca', KernelPCA(kernel="poly")), ('svc', SVC(class_weight ='balanced'))])

## 탐색 할 hyper-paramter 를 setting
param_grid = {
    'pca__n_components': [20, 40, 60, 80, 100, 120, 140], ## Kernel-PCA 파라미터(n_components)
    'pca__degree': [2, 3, 4, 5], ## Kernel-PCA 파라미터(degree)
    'svc__C': [1e3, 5e3, 1e4, 5e4, 1e5], ## SVM 파라미터
} 
```
kernel-PCA에서 탐색해야 할 hyper-paramter는 degree와 n_components 입니다.
degree는 polynomial kernel 의 승수를 의미합니다.
n_components는 몇개의 eigenvector를 활용하여 차원을 축소할 것인지를 나타내는 hyper-parameter입니다.

Support Vector Machine(SVM)에서 탐색해야 할 hyper-parameter는 C입니다.
C는 패널티 정도를 의미하며 SVM을 fitting하는 과정에서 정답 class로 분류되지 않을 때 부여되는 값과 비례합니다.

``` python
clf = GridSearchCV(pipe, param_grid=param_grid)
clf = clf.fit(X_train, y_train)
print(clf.best_estimator_)
```
![](/img/in-post/2020/2020-11-01/grid_search_result.png)
Grid-Search를 통해 앞서 setting 탐색범위를 확인하고 가장 좋은 성능의 모델을 추출합니다.

``` python
## kenel-PCA & SVM 성능평가
y_pred = clf.predict(X_test) 
print(classification_report(y_test, y_pred)) 
```
탐색을 통해 얻은 hyper-parameter를 활용하여 Kernel-PCA와 SVM 만을 활용했을 때 성능을 측정합니다.
성능측정 결과 **0.97의 정확도**가 도출되었습니다.
즉 Kernel-PCA와 SVM만으로도 충분히 40개의 얼굴 이미지는 분류가 가능하다는 것을 확인할 수 있습니다. 

##### 3. Neural Network





얼굴을 Input으로 받아 구분하기 까지 총 3가지  

Kernel-PCA와 Linear-SVM을 활용하여 얼굴을 구분하는 알고리즘을 구축할 수 있습니다.




Kernel-PCA를 이용하여 이미지로부터 특징점을 추출하는 단계입니다.
본 논문에서는 **Polynomial Kernel을 사용**합니다. 
Polynomial Kernel-PCA는 Hyper-parameter로 degree를 설정해야 합니다.
논문에서는 **degree를 4**로 설정했을 때 가장 성능이 좋았다고 기술하고 있습니다.
또한 알고리즘으로부터 추출된 **eigenvalues 중 큰 순서대로 120개를 뽑아 특징점**으로 활용했을 때 가장 성능이 높다고 기술하고 있습니다.   







 
 






## Reference
- [[PAPER]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=991133) Face Recognition Using Kernel Principal Component Analysis, 2002
- [[BLOG]](https://www.geeksforgeeks.org/ml-face-recognition-using-pca-implementation/) Face Recognition Using PCA Implementation, 
- [[YOUTUBE]](https://www.youtube.com/watch?v=6Et6S03Me4o&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=14) Kernel-based Learning - KPCA, 강필성



