"""Backend calls used by AnswerGenerator."""

from __future__ import annotations

import json
from typing import Any

from src import net
from src.model_cache import resolve_model_load_reference


def ollama_options(generator: Any) -> dict:
    options = {
        "think": False,
        "temperature": generator.temperature,
        "top_p": generator.top_p,
    }
    if generator.seed is not None:
        options["seed"] = generator.seed
    return options


def call_ollama_with_system(generator: Any, system_prompt: str, user_msg: str) -> str:
    payload = {
        "model": generator.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": ollama_options(generator),
    }
    response = net.post(
        f"{generator.ollama_base_url}/api/chat",
        json=payload,
        timeout=net.Timeout(600.0, connect=5.0),
    )
    response.raise_for_status()
    result = response.json()
    return result["message"]["content"].strip()


def stream_ollama_with_system(
    generator: Any,
    system_prompt: str,
    user_msg: str,
    cancel_event=None,
):
    payload = {
        "model": generator.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
        "options": ollama_options(generator),
    }
    with net.stream(
        "POST",
        f"{generator.ollama_base_url}/api/chat",
        json=payload,
        timeout=net.Timeout(600.0, connect=5.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if cancel_event is not None and cancel_event.is_set():
                break
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token


def mlx_generation_kwargs(generator: Any) -> dict:
    from mlx_lm.sample_utils import make_sampler

    kwargs = {
        "max_tokens": generator.max_tokens,
        "sampler": make_sampler(temp=generator.temperature, top_p=generator.top_p),
    }
    if generator.seed is not None:
        import mlx.core as mx

        mx.random.seed(generator.seed)
    return kwargs


def load_mlx_model(generator: Any, model_name: str | None = None) -> None:
    from mlx_lm import load

    target = model_name or generator.mlx_model_name
    if generator._mlx_model is None or target != generator.mlx_model_name:
        model_ref = resolve_model_load_reference(
            target,
            generator.allow_model_download,
            purpose="answer",
        )
        loaded = load(model_ref)
        generator._mlx_model = loaded[0]
        generator._mlx_tokenizer = loaded[1]
        generator.mlx_model_name = target


def build_prompt_nothink(generator: Any, system: str, user: str) -> str:
    if generator._mlx_tokenizer is None:
        raise RuntimeError("MLX tokenizer is not loaded")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return generator._mlx_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def stream_mlx(generator: Any, system: str, user: str, cancel_event=None):
    from mlx_lm import stream_generate

    prompt = build_prompt_nothink(generator, system, user)
    generation_kwargs = mlx_generation_kwargs(generator)
    if generator._mlx_model is None or generator._mlx_tokenizer is None:
        raise RuntimeError("MLX model is not loaded")
    for response in stream_generate(
        generator._mlx_model,
        generator._mlx_tokenizer,
        prompt=prompt,
        **generation_kwargs,
    ):
        if cancel_event is not None and cancel_event.is_set():
            break
        if response.text:
            yield response.text


def call_mlx(generator: Any, system: str, user: str) -> str:
    from mlx_lm import generate as mlx_generate

    prompt = build_prompt_nothink(generator, system, user)
    generation_kwargs = mlx_generation_kwargs(generator)
    if generator._mlx_model is None or generator._mlx_tokenizer is None:
        raise RuntimeError("MLX model is not loaded")
    return mlx_generate(
        generator._mlx_model,
        generator._mlx_tokenizer,
        prompt=prompt,
        verbose=False,
        **generation_kwargs,
    )


def call_claude_with_system(generator: Any, system_prompt: str, user_msg: str) -> str:
    if not generator.claude_api_key:
        raise RuntimeError("Claude backend requires claude_api_key or ANTHROPIC_API_KEY")
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Claude backend requires the 'anthropic' package") from e
    if generator._anthropic_client is None:
        generator._anthropic_client = anthropic.Anthropic(api_key=generator.claude_api_key)
    message = generator._anthropic_client.messages.create(
        model=generator.claude_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text.strip()
