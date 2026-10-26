import os
import torch
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc
from typing import Optional, List
from PIL import Image

class Chatbot:
    def __init__(
        self,
        generation_model_path: str,
        vector_db,
        reranker,
        memory=None,
        gemma_device: str = "cpu",
        gemma_torch_dtype=torch.float32,
        vision_device: str = "cuda:0",
        vision_torch_dtype=torch.bfloat16,
        vision_attn_implementation: Optional[str] = None
    ):
        self.generation_model_path = generation_model_path
        self.vector_db = vector_db
        self.reranker = reranker
        self.memory = memory or ConversationBufferWindowMemory(k=5)  # Limit to last 5 exchanges
        self.gemma_device = gemma_device
        self.gemma_torch_dtype = gemma_torch_dtype
        self.vision_device = vision_device
        self.vision_torch_dtype = vision_torch_dtype
        self.vision_attn_implementation = vision_attn_implementation
        self.prompt_template = PromptTemplate.from_template(
            "Conversation history:\n{history}\n\n"
            "Relevant information:\n{retrieved_info}\n\n"
            "Web search results:\n{web_results}\n\n"
            "User question: {user_input}\n\n"
            "Answer:"
        )
        print(f"🧠 Loading generation model (Gemma) into memory (on {self.gemma_device})...")
        if self.gemma_device.startswith("cuda"):
            torch.set_float32_matmul_precision('high')
            print("Set torch.set_float32_matmul_precision('high') for Gemma on GPU.")
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(self.generation_model_path, local_files_only=True)
        self.gemma_model = AutoModelForCausalLM.from_pretrained(
            self.generation_model_path,
            torch_dtype=self.gemma_torch_dtype,
            device_map=self.gemma_device,
            attn_implementation=None,  # No attention implementation for CPU
            local_files_only=True
        ).eval()
        self.gemma_pipeline = pipeline(
            "text-generation",
            model=self.gemma_model,
            tokenizer=self.gemma_tokenizer,
            max_new_tokens=256,
            temperature=0.7,
        )
        print("✅ Gemma model and pipeline loaded.")

        try:
            print(f"🧠 Loading image-text pipeline with Gemma on {self.vision_device}...")
            device_id = 0 if self.vision_device.startswith("cuda") else -1
            self.vision_pipeline = pipeline(
                "image-text-to-text",
                model=generation_model_path,
                torch_dtype=self.vision_torch_dtype,
                device=device_id,  # GPU if cuda, else CPU
                attn_implementation=self.vision_attn_implementation if self.vision_device.startswith("cuda") else None
            )
            print(f"✅ Image-text pipeline loaded on {self.vision_device}.")
        except Exception as e:
            print(f"❌ Failed to load image-text pipeline: {e}")
            self.vision_pipeline = None

    def get_response(self, user_input: str, web_results: Optional[List[dict]] = None) -> str:
        print("\n--- Step 1: Searching Vector DB ---")

        # Check if vector DB is available and has documents
        if self.vector_db is None:
            print("Vector DB not available - using general knowledge mode")
            retrieved_info = "No document context available. Using general knowledge to answer."
        else:
            try:
                top_k_docs = self.vector_db.search(user_input, k=5)
                top_k_chunks = [doc.page_content for doc in top_k_docs]
                print(f"Retrieved {len(top_k_chunks)} chunks.")

                if top_k_chunks:
                    print("\n--- Step 2: Reranking Chunks ---")
                    reranked_chunks = self.reranker.rerank(user_input, top_k_chunks)
                    retrieved_info = reranked_chunks[0] if reranked_chunks else "No relevant information found."
                else:
                    print("No documents found in vector DB - using general knowledge mode")
                    retrieved_info = "No document context available. Using general knowledge to answer."
            except Exception as e:
                print(f"Error searching vector DB: {e} - using general knowledge mode")
                retrieved_info = "No document context available. Using general knowledge to answer."

        print("\n--- Step 3: Formatting Web Search Results ---")
        web_context = ""
        if web_results:
            web_context = "\n".join(
                f"[{result['title']}]({result['href']}): {result['body']}"
                for result in web_results
            ) if web_results else "No web search results found."
        else:
            web_context = "No web search performed."

        print("\n--- Step 4: Generating Response ---")
        try:
            history = self.memory.load_memory_variables({})["history"]
            prompt = self.prompt_template.format(
                history=history,
                retrieved_info=retrieved_info,
                web_results=web_context,
                user_input=user_input
            )
            print("\nGenerated Prompt:\n", prompt)

            response = self.gemma_pipeline(prompt)
            generated_output = response[0]["generated_text"]
            answer_split = generated_output.split("Answer:")
            response_text = answer_split[-1].strip() if len(answer_split) > 1 else generated_output.replace(prompt, "").strip()

            # Save the conversation to memory
            self.memory.save_context({"input": user_input}, {"output": response_text})
            print(f"Conversation saved to memory. Current history:\n{self.memory.load_memory_variables({})['history']}")

        except Exception as e:
            print(f"An error occurred during text generation: {e}")
            if "CUDA out of memory" in str(e):
                print("HINT: Gemma model might be too large for your GPU. Try setting `gemma_torch_dtype=torch.float16` or `gemma_device='cpu'` in main.py.")
            response_text = "Sorry, I didn't get that. Can you rephrase?"

        return response_text

    def describe_image(self, image_path: str, prompt_text: str = "Describe this image") -> str:
        if self.vision_pipeline is None:
            return "Image pipeline not loaded. Cannot process images."

        try:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert image analyst."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            output = self.vision_pipeline(text=messages, max_new_tokens=200)
            if isinstance(output, list) and len(output) > 0:
                response_text = output[0]["generated_text"]
                # Save image description to memory
                self.memory.save_context({"input": f"Image query: {prompt_text}"}, {"output": response_text})
                print(f"Image description saved to memory. Current history:\n{self.memory.load_memory_variables({})['history']}")
                return response_text
            else:
                return "Image analysis failed: Empty output."
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def cleanup(self):
        """Unloads the Gemma model and pipeline from memory."""
        print("🧹 Unloading generation model (Gemma) from memory...")
        if hasattr(self, 'gemma_model') and self.gemma_model is not None:
            if self.gemma_model.device.type == 'cuda':
                self.gemma_model.to("cpu")
            del self.gemma_model
        if hasattr(self, 'gemma_tokenizer') and self.gemma_tokenizer is not None:
            del self.gemma_tokenizer
        if hasattr(self, 'gemma_pipeline') and self.gemma_pipeline is not None:
            del self.gemma_pipeline
        if hasattr(self, 'vision_pipeline') and self.vision_pipeline is not None:
            if self.vision_device.startswith('cuda'):
                model = self.vision_pipeline.model
                if model is not None and model.device.type == 'cuda':
                    model.to("cpu")
                del model
            del self.vision_pipeline
        gc.collect()
        if self.vision_device.startswith('cuda') or self.gemma_device.startswith('cuda'):
            torch.cuda.empty_cache()
