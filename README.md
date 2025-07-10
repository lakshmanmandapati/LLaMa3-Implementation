# **Llama 3 Implementation (From Scratch!)**

## **Features**
1.  Rotary Positional Embeddings (RoPE)
2.  Grouped Query Attention (GQA)
3.  Key-Value (KV) Cache
4.  RMS Normalization
5.  SwiGLU Activation
6.  Top-p Sampling

All features are implemented to match the original model architecture.

## **Structure**
* `model.py`: Llama 3 model implementation.
* `inference.py`: Code for running inference.

## **Testing**
Tested with official **Llama 3 8B Instruct** pretrained weights.

**To get Llama 3 weights:**
1.  **Request Access:** Visit [Meta Llama website](https://ai.meta.com/llama/) and accept license.
2.  **Hugging Face Login:** Log in via `huggingface-cli login` in your terminal.

`inference.py` automatically downloads model and tokenizer on first run.

## **Running Inference**

### **Requirements**
* `torch`
* `transformers`
* `sentencepiece`
* `tqdm`
* `accelerate`

Install using:

```bash
pip install -r requirements.txt
