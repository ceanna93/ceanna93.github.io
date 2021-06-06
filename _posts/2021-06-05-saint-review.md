---
title: "Review \"Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing\""
date: 2021-06-05 00:00:00 -0400
categories: AI
use_math: true
---

SAINT: Separated Self-AttentIve Neural Knowledge Tracing
===================================

"Deep Knowledge Tracing" task를 위해 Riiid 대회에서 사용된 모델 Saint 간단 리뷰

제공되는 정보는 userID(사용자 ID), assessmentItemID(과제 ID. testID에 포함), testID(시험지 ID), answerCode(정답 여부), Timestamp(답을 제출한 시간), KnowledgeTag(시험의 유형)가 있다.

Riiid와는 제공되는 데이터와 다르기 때문에 모델에 수정이 필요하다.

# ABSTRACT
베이지안 지식 추적(Bayesian knowledge tracing)과 협업 필터링(collaborative filtering)보다 RNN과 Transformer를 이용한 방법이 훨씬 좋은 성능을 보인다. 하지만 attention 메카니즘에도 두 가지 문제가 존재한다.
1. 지식 추적을 위해 심도있는 자기주의 계산을 활용하지 못한다. 모델은 시간이 지남에 따라 연습과 응답 간의 복잡한 관계를 포착하지 못하게 된다.
2. 지식 추적을 위한 자기주의 계층에 대한 쿼리, 키 및 값을 구성하는 데 적합한 feature들이 널리 탐색되지 않았다. 연습과 상호 작용 (운동-응답 쌍)을 각각 쿼리 및 키 / 값으로 사용하는 일반적인 관행은 아직 경험이 부족하다.
대강 해석한 내용으로 생각해보면 지식 추적을 하다보니 Self attention에 더 집중하지 못 하고 시간의 흐름에 따라 변화하는 관계를 잘 포착하지 못하는 것 같다. 그리고 DKT 분야가 아직 흔하지 않아 어떤 feature들이 가장 적절한 지 확실하지 않은 문제도 있다는 내용으로 보인다.

그래서 논문에서는 transformer 기반의 SAINT 모델을 제안하고 있다. SAINT 모델에서는 encoder와 decoder가 있는데 각각에 연습과 응답 임베딩 시퀀스가 들어가게 된다. Encoder는 연습 임베딩 시퀀스에 self attention layer를 적용하고, decoder는 응답 임베딩 시퀀스에 self attention layer와 인코더-디코더 attention layer를 교대로 적용한다. 이러한 입력 분리를 통해 attention layer를 여러 번 쌓아 AUC(DKT의 점수 측정 방법) 아래 영역을 개선할 수 있다. 이것은 연습과 응답에 개별적으로 깊은 self attention layer을 적용하는 지식 추적을 위한 인코더-디코더 모델을 제안하는 첫 번째 작업이다.

저자는 활발한 모바일 교육을 통해 수집된 대규모 지식 추적 Dataset인 EdNet에서 SAINT를 경험적으로 평가했다고 설명한다. 그리고 결과는 SAINT가 현재의 최신 모델(추측하기론 RNN, Transformer를 의미하는 것 같다)에 비해 AUC에서 1.8 %의 개선으로 지식 추적에서 최첨단 성능을 달성했음을 보여준다.


[Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing](https://arxiv.org/pdf/2002.07033.pdf)
