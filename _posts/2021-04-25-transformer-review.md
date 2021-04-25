---
title: "Review \"Attention Is All You Need\""
date: 2021-02-17 00:00:00 -0400
categories: AI
use_math: true
---

# [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
## 서론
BERT, ELECTRA를 접하면서 Transformer라는 단어를 많이 접했다. Transformer를 여러번 배웠지만 정확하게 알고 있지 않은 느낌이 계속 들어서 Transformer 논문을 정리해보려고 한다.

### Attention
논문을 읽기 전, 제목에 적혀있는 **Attention**가 어떤 의미인지 확인했다.

[딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/22893)에 설명되어 있고 [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)에서도 확인할 수 있다.

어텐션(attention): 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위한 기법

## 논문
### Abstract
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

recurrence와 convolutions를 제거하고 attention mechanism만을 이용한 방법.

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
