"""로컬 오픈웨이트 모델(4-bit) 로딩 + chat 래퍼.

공용 GPU 를 고려해 프로세스 메모리 상한을 걸 수 있게 했고,
모델마다 다른 chat template 을 tokenizer.apply_chat_template 로 처리한다.
"""
import inspect
import re
from functools import wraps

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _install_exaone35_causal_mask_compat(masking_module=None):
    """Temporarily adapt EXAONE 3.5 remote code to newer Transformers APIs."""
    if masking_module is None:
        try:
            from transformers import masking_utils as masking_module
        except ImportError:
            return None

    original = masking_module.create_causal_mask
    parameters = inspect.signature(original).parameters
    if "input_embeds" in parameters or "inputs_embeds" not in parameters:
        return None

    @wraps(original)
    def compatible_create_causal_mask(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" not in parameters:
            kwargs.pop("cache_position", None)
        return original(*args, **kwargs)

    masking_module.create_causal_mask = compatible_create_causal_mask
    return masking_module, original, compatible_create_causal_mask


def _restore_causal_mask(patch_state):
    if patch_state is None:
        return
    masking_module, original, installed = patch_state
    if masking_module.create_causal_mask is installed:
        masking_module.create_causal_mask = original


def load_model(model_id, load_in_4bit=True, gpu_index=0, mem_fraction=None,
               trust_remote_code=False):
    """모델과 토크나이저를 로드한다. 4-bit 는 bitsandbytes(nf4) 사용."""
    if mem_fraction:
        # 내 프로세스가 잡을 수 있는 VRAM 을 전체의 fraction 으로 제한(공용 GPU 보호).
        torch.cuda.set_per_process_memory_fraction(mem_fraction, gpu_index)

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    patch_state = None
    if "EXAONE-3.5" in model_id.upper():
        patch_state = _install_exaone35_causal_mask_compat()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{gpu_index}"},
            trust_remote_code=trust_remote_code,
        )
    finally:
        _restore_causal_mask(patch_state)
    model.eval()
    return model, tokenizer


def model_context_window(model, tokenizer):
    """Return the smallest finite context limit advertised by model/tokenizer."""
    candidates = []
    for value in (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    return min(candidates) if candidates else None


def validate_context_budget(model, tokenizer, input_tokens, max_new_tokens):
    """Fail before generation when the prompt plus output budget exceeds context."""
    context_window = model_context_window(model, tokenizer)
    if context_window is None:
        return None
    required = input_tokens + max_new_tokens
    if required > context_window:
        raise ValueError(
            "context window exceeded: "
            f"input_tokens={input_tokens} + max_new_tokens={max_new_tokens} "
            f"> context_window={context_window}. "
            "Reduce the article/prompt or max_new_tokens; input is not truncated."
        )
    return context_window


def build_messages(user_content, system_prompt=None, history=None):
    """chat 메시지 리스트를 구성한다."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


@torch.no_grad()
def chat(model, tokenizer, messages, max_new_tokens=256, temperature=0.7,
         top_p=0.8, enable_thinking=None):
    """messages 를 모델 template 에 맞춰 생성한다. 프롬프트를 제외한 새 토큰만 반환."""
    template_kwargs = {}
    if enable_thinking is not None:  # Qwen3 계열만 이 인자를 받는다
        template_kwargs["enable_thinking"] = enable_thinking

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    validate_context_budget(
        model, tokenizer, inputs["input_ids"].shape[1], max_new_tokens
    )

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def strip_think(text):
    """Qwen thinking 모드의 <think>...</think> 블록을 제거한다."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
