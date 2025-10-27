# Improving Attention Mechanisms via the GWO Framework

This repository contains the code and experimental results for improving the standard attention mechanism in a GPT-2 model by applying principles from the paper **"Window is Everything: A Grammar for Neural Operations"**. We introduce *FactorizedAttention*, a novel attention mechanism derived from this framework, and demonstrate its superior performance across several datasets.

## Abstract

This project demonstrates the practical application of the Generalized Windowed Operation (GWO) framework for designing novel neural operations. We introduce and implement **FactorizedAttention**, an alternative to the standard self-attention mechanism. This repository allows for the training and evaluation of a standard GPT-2 small model versus a GPT-2 model modified with FactorizedAttention, both fine-tuned using LoRA.

The GWO framework proposes a unified theory that decomposes any neural operation into three orthogonal components: **Path** (defining operational locality), **Shape** (defining geometric structure), and **Weight** (defining feature importance). This project leverages this grammar to enhance the standard attention mechanism.

*   **To run the experiments:** `python main.py`
*   **To change the dataset:** Modify the `dataset_name` parameter in `config.yaml`.
*   **Original GWO Paper:** [Window is Everything: A Grammar for Neural Operations](https://zenodo.org/records/17103133)

**Contact:**
*   **Author:** Youngseong Kim
*   **Email:** dafaafafaf33@gmail.com
*   **Discord:** [Discussion on Discord](https://discord.gg/tfuYKTGTk5)

---

## 1. Introduction

The Generalized Windowed Operation (GWO) paper provides a powerful theoretical grammar for decomposing, analyzing, and creating neural network operations. It posits that any operation can be understood as a combination of three components: Path (P), Shape (S), and Weight (W). This provides a principled approach to designing new architectures tailored to specific data properties.

This project aims to validate the utility of the GWO framework by designing a new attention mechanism, **FactorizedAttention**, and empirically demonstrating its effectiveness.

## 2. Methodology

### 2.1. Analyzing Standard Attention with GWO

In the GWO framework, the standard self-attention mechanism can be deconstructed as follows:
- The **Combination** function, which computes the attention matrix, is a simple inner product (dot product) between query (Q) and key (K) vectors.

This dot product can only capture first-order linear relationships between tokens. We hypothesized that enhancing this Combination function could allow the model to capture more complex, higher-order interactions, leading to improved performance.

### 2.2. FactorizedAttention

Instead of a simple dot product, FactorizedAttention implements a more expressive interaction function. Specifically, we replace the `QK^T` computation with a factorized quadratic form. This is achieved through the following steps:

1.  **Multiplicative Factorization:** The query and key vectors are passed through separate linear layers (`q_w1`, `q_w2` for query and `k_w3`, `k_w4` for key) to create intermediate representations. These are then combined with element-wise multiplication.
    
    ```
    q_interaction = q_w1(query) * q_w2(query)
    k_interaction = k_w3(key) * k_w4(key)
    ```
    
2.  **Projection and Activation:** The results are passed through a GELU activation and projected back to their original dimension.
    
3.  **Gated Residual Connection:** A learnable, sigmoidal gate (`alpha_q`, `alpha_k`) is used to add this new structured representation back to the original query and key vectors. This allows the model to learn how much of the higher-order interaction to incorporate.
    
4.  **Learnable Temperature:** A learnable temperature parameter is applied on a per-head basis, allowing the model to control the sharpness of the attention distribution.
    
5.  **Normalization:** LayerNorm is applied to the query and key vectors before the factorization for increased training stability.

## 3. Theoretical Analysis

The standard dot-product attention `(W_q * q) * (W_k * k)` is fundamentally a bilinear operation. By introducing the multiplicative interaction `(q * W_1 * W_2 * q)`, our FactorizedAttention approximates a quadratic form, allowing the model to learn second-order dependencies and more complex relationships between tokens.

From the perspective of the **GWO theory**, this increased expressiveness is a direct application of the **Principle of Structural Alignment**. Datasets involving complex reasoning, such as mathematics (`MetaMathQA`) or algorithmic coding (`code_contests`), are characterized by high-order, non-linear relationships between tokens. A simple variable's meaning might depend on a complex interplay of operators, function definitions, and logical structures that a simple bilinear dot product may not fully capture.

By modifying the `Combination` function of the attention mechanism to model these higher-order interactions, `FactorizedAttention` achieves a better structural alignment with the intrinsic complexity of the data. This alignment, according to the GWO framework, is the key to unlocking better generalization and performance, which is precisely what we observe in our experiments.

## 4. Experiments

### 4.1. Experimental Setup

We evaluated the performance of FactorizedAttention against the standard attention mechanism on a **GPT-2 small** model. To isolate the training to the attention blocks, we used **Low-Rank Adaptation (LoRA)**. For the baseline model, LoRA was applied to the `c_attn` and `c_proj` modules. For the `FactorizedAttention` model, in addition to applying LoRA to `c_attn` and `c_proj` for a fair comparison, all the newly introduced parameters specific to the factorized mechanism (such as projection layers and gating parameters) were also made fully trainable.

The experiments were conducted on three distinct datasets to test performance on different domains:
*   `codeparrot/codeparrot-clean`: A dataset of Python code.
*   `deepmind/code_contests`: A dataset of competitive programming problems and solutions.
*   `meta-math/MetaMathQA`: A dataset focused on advanced mathematical reasoning.

### 4.2. Results

The patched model using FactorizedAttention demonstrated consistent performance improvements over the baseline model across all three datasets. The results below are averaged over three runs with different random seeds. Perplexity (PPL) is used as the evaluation metric, where lower is better.

| Dataset | Model | Mean Perplexity (PPL) | Improvement |
| :--- | :--- | :--- | :--- |
| **codeparrot-clean** | Baseline | 5.3948 +/- 0.0138 | - |
| | **FactorizedAttention** | **5.2918 +/- 0.0039** | **1.91% +/- 0.20%** |
| **code_contests** | Baseline | 4.3243 +/- 0.0142 | - |
| | **FactorizedAttention** | **4.1876 +/- 0.0210** | **3.16% +/- 0.26%** |
| **MetaMathQA** | Baseline | 5.8064 +/- 0.0085 | - |
| | **FactorizedAttention** | **5.6083 +/- 0.0048** | **3.41% +/- 0.16%** |

Notably, the most significant improvement was observed on the **MetaMathQA** dataset, which supports our hypothesis that the enhanced expressive power of FactorizedAttention is particularly beneficial for complex reasoning tasks. It is also crucial to highlight that these gains were achieved even though the base GPT-2 model was pre-trained with standard attention, giving the baseline an inherent advantage. This underscores the effectiveness of our GWO-derived modifications.

While `FactorizedAttention` uses more trainable parameters (1.40M vs. 0.81M), this increase is marginal compared to the total model size of ~125M parameters. The additional overhead is less than 0.5% of the total parameters, yet it yields a significant performance boost. This demonstrates a highly favorable trade-off, indicating that the added parameters efficiently enhance the model's expressive power.

## 5. Conclusion and Future Work

### 5.1. Conclusion

This project successfully demonstrates that the GWO framework is not just a theoretical tool but a practical grammar for engineering novel and superior neural network operations. By analyzing the standard attention mechanism through the GWO lens and enhancing its Combination function, we created FactorizedAttention, which consistently outperforms the baseline. This serves as a strong proof-of-concept for the utility of using the GWO framework to guide future innovations in neural architecture design.

### 5.2. Future Work

While this work provides strong evidence for the effectiveness of FactorizedAttention, several avenues for future research remain:

*   **Scaling to Larger Models:** The experiments in this project were conducted on a GPT-2 small model. An important next step is to apply FactorizedAttention to larger-scale models (e.g., Llama, GPT-3) to investigate whether the performance improvements scale with model size.
*   **Evaluation on Diverse Metrics:** Perplexity is a useful measure of a language model's predictive ability, but it doesn't always correlate directly with downstream task performance. Future work should evaluate FactorizedAttention on a broader range of metrics, such as code generation benchmarks (e.g., HumanEval) and mathematical reasoning accuracy tests, to provide a more comprehensive understanding of its capabilities.