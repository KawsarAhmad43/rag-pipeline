from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from config import config
import torch
import logging
import psutil
import gc
from transformers.utils import is_flash_attn_2_available
from utils.system import get_memory_usage, clean_gpu_cache

logger = logging.getLogger(__name__)



def get_gemma_llm(model_path):
    logger.debug(f"Loading Gemma LLM from: {model_path}")
    try:
        # Enable TensorFloat32 for better performance
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')

        clean_gpu_cache()

        cpu_mem, gpu_mem = get_memory_usage()
        logger.debug(f"Memory usage before loading Gemma: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=config.HUGGING_FACE_API_KEY,
            local_files_only=True,
            trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine device and dtype
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        attn_implementation = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=config.HUGGING_FACE_API_KEY,
            local_files_only=True,
            device_map="auto",
            dtype=dtype,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=attn_implementation
        ).eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

        pipeline_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.5,
            "do_sample": True,      # Enable sampling for more natural responses
            "repetition_penalty": 1.2,
        }

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
            clean_up_tokenization_spaces=True,
            device_map="auto",
            **pipeline_kwargs
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        cpu_mem, gpu_mem = get_memory_usage()
        logger.info(f"Gemma LLM initialized successfully. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        return llm
    except Exception as e:
        cpu_mem, gpu_mem = get_memory_usage()
        logger.error(f"Failed to load Gemma LLM: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        raise RuntimeError(f"Gemma LLM initialization failed: {str(e)}")
