import subprocess
import time

import ollama
import requests
from ollama import ChatResponse, chat

MODEL = "smollm2:135m"
HOST  = "http://127.0.0.1:11434"

def ensure_ollama_installed():
    try:
        subprocess.run(["ollama", "--version"], check=True)
    except subprocess.CalledProcessError:
        print("Installing Ollama…")
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, check=True)
        print("Ollama installed successfully!")
    else:
        print("Ollama already installed.")

def ensure_model_pulled(model=MODEL):
    """Ensure the embedding model is pulled locally using Ollama."""
    existing_models = ollama.list()
    existing_model_names = [m.model.split(':')[0] for m in existing_models['models']]
    print(f"Existing models: {existing_model_names}")
    
    if model.value not in existing_model_names:
        print(f"Pulling embedding model {model.value}…")
        ollama.pull(model.value)
    else:
        print(f"Embedding model {model.value} already present.")

def start_and_wait(host=HOST, timeout=60, interval=0.5):
    """Start Ollama server in background (if needed) and wait for it."""
    try:
        # Quick check if already running
        r = requests.get(f"{host}/api/tags", timeout=2)
        if r.ok:
            print("Ollama server is already running.")
            return True
    except requests.ConnectionError:
        print("Ollama server not running. Starting it...")

    # Start in background
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait loop
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags")
            if r.ok:
                print("Ollama server started successfully.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)

    print("Timed out waiting for Ollama server to start.")
    return False


def generate_prompt(prompt: str):
    resp: ChatResponse = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return resp.message.content


def delete_ollama_model(model_name: str = 'smollm2:135m'):
    print(f"Attempting to delete model: {model_name}...")
    
    try:
        ollama.delete(model_name)
        print(f"Successfully deleted model: {model_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    ensure_ollama_installed()
    ensure_model_pulled()
    if not start_and_wait():
        raise RuntimeError("Ollama server didn’t start in time")
    print(generate_prompt("Why is the sky blue?"))