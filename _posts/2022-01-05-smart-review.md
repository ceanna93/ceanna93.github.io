# SMART
Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization

### 1. Introduction
To fully harness the power of fine-tuning in a more principled manner, we propose a new learning framework for robust and efficient fine-tuning on the pre-trained language models through regularized optimization techniques. Specifically, our framework consists of two important
ingredients for preventing overfitting:

(I) To effectively control the extremely high complexity of the model, we propose a Smoothnessinducing Adversarial Regularization technique. Our proposed regularization is motivated by local shift sensitivity in existing literature on robust statistics. Such regularization encourages the output of the model not to change much, when injecting a small perturbation to the input. Therefore, it enforces the smoothness of the model, and effectively controls its capacity (Mohri et al., 2018).

(II) To prevent aggressive updating, we propose a class of Bregman Proximal Point Optimization methods. Our proposed optimization methods introduce a trust-region-type regularization (Connet al., 2000) at each iteration, and then update the model only within a small neighborhood of the previous iterate. Therefore, they can effectively prevent aggressive updating and stabilize the fine-tuning process.


---

- https://arxiv.org/pdf/1911.03437.pdf
- https://paperswithcode.com/paper/smart-robust-and-efficient-fine-tuning-for
