---
title: "Review \"Attention Is All You Need\""
date: 2021-04-25 00:00:00 -0400
categories: AI
use_math: true
---

# [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
## 서론
BERT, ELECTRA를 접하면서 Transformer라는 단어를 많이 접했다. Transformer를 여러번 배웠지만 정확하게 알고 있지 않은 느낌이 계속 들어서 Transformer 논문을 정리해보려고 한다.

### Attention
논문을 읽기 전, 제목에 적혀있는 **Attention**이 어떤 의미인지 확인했다.

[딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/22893)에 설명되어 있고 [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)에서도 확인할 수 있다.

어텐션(attention): 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위한 기법

## 논문
### Abstract
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

recurrence와 convolutions는 layer를 의미한다. recurrent layer와 convolutional layer를 제외하고 오로지 Self-Attention Layer만으로 만든 Transformer라는 의미가 된다.

### 1. Introduction
Attention mechanisms have become an integral part of compelling sequence modeling and trasduction models in various tasks, allowing modeling of dependencies without regard to their distnace in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw gloabl dependencies between input and output.

기존의 Attention mechanisms이 사용됐던 한계에 대해 설명하고 Transformer를 제안하고 있다.

### 2. Background
To the best of our knowledge, however, the Transformer is the transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

### 3. Model Architecture
The encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of continuous repesentations $z = (z_1, ..., z_n)$. Given z, the decoder then generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

<img src="https://user-images.githubusercontent.com/12611645/115994730-23ca9300-a613-11eb-9842-f8b83641e517.JPG" width="40%" height="30%" title="The Transformer diagram" alt="Transformer architecture">

<details>
<summary>Encoder/Decoder</summary>
<div markdown="1">

세부 내용 Encoder and Decoder Stacks를 읽기 전, 내가 정확하게 Encoder와 Decoder에 대해 알고 있는 것인지 의문이 들었다. 문장을 입력받는 부분이 encoder, 출력하는 부분이 decoder라는 것은 알지만 RNN에서 정확히 어느 부분이 encoder인지 decoder인지 구분하라고 하면 하지 못 할 것 같다. 그래서 **ENCODER/DECODER**도 다시 짚고 넘어가기로 했다.

이미지를 찾아봤을 때, 아래처럼 Encoder와 Decoder는 항상 나눠져있다.

<img src="https://user-images.githubusercontent.com/12611645/115995895-15cb4100-a618-11eb-809b-46186a5edbbd.png" width="40%" height="30%" title="Encoder and Decoder" alt="Encoder Decoder">

문장을 입력하면 문장이 출력되는 구조에서, 문장을 입력받는 부분이 encoder, 출력하는 부분이 decoder라고 되어있다. 그런데 RNN에서 하나의 입력에 대해 다음 단어를 예상하는 경우가 의문인데 이 때 RNN은 입력을 받기도 하고 출력을 하기도 한다. 그렇다면 encoder와 decoder를 어떻게 나누는 것일까?

위에 적었던 [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) 페이지를 보면 EncoderRNN 함수와 DecoderRNN 함수가 구현되어 있다. ~~코드를 조각내보면 뭐라도 나오겠지.~~

먼저 EncoderRNN 함수를 보면 super로 nn.Module 함수를 상속받는다. 
동일한 명칭으로 [IBM github](https://github.com/IBM/pytorch-seq2seq/tree/master/seq2seq/models) repository에 EncoderRNN과 DecoderRNN 함수가 구현되어 있다. 정작 pytorch 공식 페이지의 [rnn 모듈](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#RNN) 페이지에 들어가보면 딱히 encoder, decoder가 명시되어 있지 않다.


**잘 몰랐던 파이썬 상속**

파이썬에서는 부모 클래스의 생성자를 자식 클래스에서 호출하려면 super 함수 내부에 자기 자신의 클래스명으로 호출해줘야 한다. 따라서 pytorch 페이지에 있는 EncoderRNN 클래스 내부에서 ```super(EncoderRNN, self).__init__()```를 정의함으로써 nn.Module 생성자를 호출하는 것이고, IBM github의 함수에서는 ```super(EncoderRNN, self).__init__()``` 정의해 자신과 동일한 경로에 존재하는 BaseRNN 클래스의 생성자를 호출하게 된다.
[관련 Q&A](https://hashcode.co.kr/questions/965/%EB%B6%80%EB%AA%A8-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%9D%98-%EC%83%9D%EC%84%B1%EC%9E%90%EB%A5%BC-%ED%98%B8%EC%B6%9C%ED%95%98%EB%A0%A4%EB%A9%B4-%EA%BC%AD-%EC%9E%90%EA%B8%B0-%EC%9E%90%EC%8B%A0%EC%9D%84-%EC%8D%A8%EC%A4%98%EC%95%BC-%ED%95%98%EB%82%98%EC%9A%94)


(IBM github에 있는) 두 함수를 비교했을 때 확실히 encoder와 decoder는 forward 함수가 return 하는 것부터 달랐다. decoder 함수 부분에는 teacher forcing에 관한 조건도 붙어있다.

</div>
</details>

#### 3.1 Encoder and Decoder Stacks
**Encoder:** The encoder is composed of a stack of *N* = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each sub-layers is LayerNorm($x$ + Sublayer($x$)), where Sublayer($x$) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model}$ = 512

**Decoder:** The decoder is also composed of a stack of *N* = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less that $i$.
