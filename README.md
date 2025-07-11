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



### **Steps**
1.  **Prepare:** Ensure Llama 3 license accepted and Hugging Face logged in (`huggingface-cli login`).
2.  **Execute:** Run script:

    ```bash
    python inference.py
    ```
    (First run downloads large model weights; may take time.)

### **Important Note on Hardware:**

This implementation aims for architectural understanding. Running the full Llama 3 8B model (which is around 16GB in size in bfloat16 precision) on systems with limited unified or dedicated GPU memory might lead to "Out of Memory" errors. For practical deployment of such models on resource-constrained hardware, highly optimized tools like **Ollama** are often recommended.

### **Resources**
* **Official Llama 3:** [Meta Llama Website](https://ai.meta.com/llama/), [Hugging Face Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* **Practical Use:** [Ollama Website](https://ollama.com/)
* **General LLM Understanding:** Various online resources, open-source implementations, and tutorials.
