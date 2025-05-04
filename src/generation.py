from ollama import ChatResponse, chat

from ollama_utils import (OllamaConfig, ensure_model_pulled, ensure_ollama_installed,
                          start_ollama_service)


def generate_prompt(prompt: str):
    resp: ChatResponse = chat(
        model=OllamaConfig.default_generation_model,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return resp.message.content


if __name__ == "__main__":
    ensure_ollama_installed()
    ensure_model_pulled()
    if not start_ollama_service():
        raise RuntimeError("Ollama server didn’t start in time")
    print(generate_prompt("Why is the sky blue?"))