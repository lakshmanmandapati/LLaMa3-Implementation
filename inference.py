import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm 
import torch.nn.functional as F 
import traceback 

from model import Llama3Model 
from config import LLAMA3_8B_CONFIG 

# --- IMPORTANT PRE-REQUISITES ---
# 1. Accept Llama 3 License: Go to https://llama.meta.com/llama3/ or
#    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and accept the license.
# 2. Hugging Face Login: Open your terminal and run `huggingface-cli login`.
#    Paste your Hugging Face user access token when prompted.
#    This allows the `transformers` library to download gated models.
#    (Optional: You could also set `os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"` in your script,
#    but `huggingface-cli login` is generally more secure for local development).

class Llama3Inference:
    def __init__(self, model_size: str = "8B"):
        print("--- Initializing Llama3Inference ---") 
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if model_size == "8B":
            self.config = LLAMA3_8B_CONFIG
            self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError(f"Unsupported model size: {model_size}. Only 8B is configured for this example.")

        print(f"Instantiating custom Llama 3 {model_size} model architecture...") 

        self.model = Llama3Model(self.config).to(self.device)
        print("Custom Llama3Model instance created on device.") 

        print(f"Loading tokenizer from Hugging Face model ID: {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print("Tokenizer loaded successfully.")

        # Load the pre-trained weights into your custom model
        self._load_pretrained_weights(self.model_id)
        self.model.eval() 
        print("Model set to evaluation mode. Llama3Inference initialization complete.") # Debug point

    def _load_pretrained_weights(self, model_id: str):
        """
        Loads pre-trained Llama 3 weights from Hugging Face into the custom model.
        
        WARNING for 8GB M2 Macs:
        This method attempts to load the full-precision (bfloat16) Llama 3 8B model.
        The Llama 3 8B model is approximately 16GB in bfloat16.
        Your 8GB M2 Mac has unified memory, so the OS and other processes also use this RAM.
        Therefore, it is highly probable you will encounter an Out-Of-Memory (OOM) error here.
        If you get an OOM, this means the model simply does not fit in your available memory.
        """
        print(f"\n--- Attempting to load FULL-PRECISION weights for '{model_id}' ---")
        print("IMPORTANT WARNING: This Llama 3 8B model requires ~16GB of RAM in bfloat16.")
        print(f"Your Mac M2 has 8GB unified memory. An Out-Of-Memory error is VERY LIKELY here.")
        print("If it fails, consider using Ollama for practical Llama 3 8B usage on your hardware,")
        print("or try a much smaller model (e.g., TinyLlama 1.1B) with this 'from scratch' approach.")

        try:
            print("Calling AutoModelForCausalLM.from_pretrained to load source weights...") 
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16, 
                device_map="auto" 
            )
            print("Hugging Face source model loaded (might be partially on CPU).") 

            print("Extracting state_dict from Hugging Face model and preparing for custom model mapping...") # Debug point
            state_dict = hf_model.state_dict() 
            new_state_dict = {} 
            new_state_dict['tok_embeddings.weight'] = state_dict['model.embed_tokens.weight']
            for i in range(self.config.n_layers):
                # Attention block
                new_state_dict[f'layers.{i}.attention_norm.weight'] = state_dict[f'model.layers.{i}.input_layernorm.weight']
                new_state_dict[f'layers.{i}.attention.wq.weight'] = state_dict[f'model.layers.{i}.self_attn.q_proj.weight']
                new_state_dict[f'layers.{i}.attention.wk.weight'] = state_dict[f'model.layers.{i}.self_attn.k_proj.weight']
                new_state_dict[f'layers.{i}.attention.wv.weight'] = state_dict[f'model.layers.{i}.self_attn.v_proj.weight']
                new_state_dict[f'layers.{i}.attention.wo.weight'] = state_dict[f'model.layers.{i}.self_attn.o_proj.weight']
                
                # FeedForward block
                new_state_dict[f'layers.{i}.ffn_norm.weight'] = state_dict[f'model.layers.{i}.post_attention_layernorm.weight']
                new_state_dict[f'layers.{i}.feed_forward.w1.weight'] = state_dict[f'model.layers.{i}.mlp.gate_proj.weight']
                new_state_dict[f'layers.{i}.feed_forward.w2.weight'] = state_dict[f'model.layers.{i}.mlp.down_proj.weight']
                new_state_dict[f'layers.{i}.feed_forward.w3.weight'] = state_dict[f'model.layers.{i}.mlp.up_proj.weight']

            # Final normalization and output layer
            new_state_dict['norm.weight'] = state_dict['model.norm.weight']
            new_state_dict['output.weight'] = state_dict['lm_head.weight']

            print("Loading mapped weights into custom Llama3Model...") 
            self.model.load_state_dict(new_state_dict, strict=True) 
            print("Pre-trained weights successfully loaded into custom Llama3Model.")
            del hf_model
            if self.device == "mps":
                torch.mps.empty_cache() 
            print("Hugging Face source model instance cleared from memory.") 

        except Exception as e:
            print(f"\n!!! CRITICAL ERROR during weight loading !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\n--- Full Traceback ---")
            traceback.print_exc()
            print("--- End Traceback ---")
            print("\nCommon causes:")
            print("  - Out-Of-Memory (OOM): Llama 3 8B (16GB) is too large for 8GB RAM.")
            print("  - Hugging Face Access: Token issues or license not accepted.")
            print("  - Weight Mapping: Incorrect keys in `new_state_dict` mapping to `self.model`.")
            print("Exiting due to critical error.")
            exit(1)

    def generate_text(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.6, top_p: float = 0.9):
        """
        Generates text based on a given prompt using the Llama 3 model.
        """
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt" 
        ).to(self.device)

        generated_tokens = []
        start_pos = 0 
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [self.tokenizer.eos_token_id, eot_id]

        print("\n--- Generated Response ---")
        print(prompt, end="", flush=True) 
        with torch.no_grad():
            for i in tqdm(range(max_new_tokens), desc="Generating tokens"):
                output = self.model(input_ids, start_pos)
                logits = output[:, -1, :] 

                if temperature < 1e-5: # Pure greedy decoding (always pick the highest probability token)
                    next_token = torch.argmax(logits, dim=-1)
                else: # Sampling (introduces randomness for more diverse outputs)
                    probs = F.softmax(logits / temperature, dim=-1) # Apply temperature for softer probabilities
                    if top_p < 1.0: # Nucleus sampling (Top-P)
                        # Sort probabilities in descending order
                        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                        # Calculate cumulative probabilities
                        cumsum_probs = torch.cumsum(probs_sort, dim=-1)
                        # Mask out tokens whose cumulative probability exceeds top_p
                        mask = cumsum_probs - probs_sort > top_p
                        probs_sort[mask] = 0.0 # Set probabilities of masked tokens to 0
                        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # Re-normalize remaining probabilities
                        # Sample from the filtered probabilities
                        next_token = torch.multinomial(probs_sort, num_samples=1)
                        # Get the original token ID from the sorted index
                        next_token = torch.gather(probs_idx, -1, next_token)
                    else: # Pure temperature sampling (sample from all tokens according to temperature-adjusted probs)
                        next_token = torch.multinomial(probs, num_samples=1)

                # Decode and print the newly generated token in real-time
                decoded_token = self.tokenizer.decode(next_token.item(), skip_special_tokens=True)
                print(decoded_token, end="", flush=True) # `end=""` prevents newline, `flush=True` prints immediately
                
                # Check if the generated token is a termination token
                if next_token.item() in terminators:
                    break # Stop generation

                generated_tokens.append(next_token.item()) # Add token to history
                
                # For the next iteration, the input to the model is just the newly generated token
                # This is efficient for incremental decoding with KV caching.
                input_ids = next_token.unsqueeze(0) # Reshape (1) to (1, 1) for batching
                start_pos += 1 # Increment current position for KV cache

        print("\n--------------------------")
        # Return the full generated text for further use if needed
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("\n--- Executing Main Block (inference.py) ---") # Debug point
    
    # Initialize inference engine for Llama 3 8B
    # This is the point where the model download and loading will occur.
    inference_engine = Llama3Inference(model_size="8B")

    # --- Example Text Generations ---

    # Example 1: Explaining a complex concept
    text_prompt_1 = "Explain the concept of quantum computing in simple terms for a high school student."
    print(f"\nUser Query 1: {text_prompt_1}")
    generated_response_1 = inference_engine.generate_text(text_prompt_1, max_new_tokens=300)
    # The response is printed during generation, but you can also use the returned string:
    # print(f"Full Response (from return): {generated_response_1}") 

    # Example 2: Creative writing
    text_prompt_2 = "Write a short, inspiring haiku about the sunrise over the ocean."
    print(f"\nUser Query 2: {text_prompt_2}")
    generated_response_2 = inference_engine.generate_text(
        text_prompt_2,
        max_new_tokens=50, # Shorter output
        temperature=0.8, # Higher temperature for more creativity
        top_p=0.95 # Nucleus sampling
    )

    # Example 3: Factual question
    text_prompt_3 = "What is the capital of France?"
    print(f"\nUser Query 3: {text_prompt_3}")
    generated_response_3 = inference_engine.generate_text(
        text_prompt_3,
        max_new_tokens=20, # Short answer expected
        temperature=0.1, # Lower temperature for less creativity, more factual
        top_p=0.9 # Nucleus sampling
    )
    print("\n--- Script Finished ---") # Debug point