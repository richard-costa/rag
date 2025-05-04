import subprocess
import time
from dataclasses import dataclass

import ollama
import requests
from loguru import logger


@dataclass
class OllamaConfig:
    default_generation_model: str = "smollm2:135m"
    default_embedding_model: str = 'mxbai-embed-large:latest'
    host: str = "http://127.0.0.1:11434"

def install_ollama():
    logger.info("Installing Ollama…")
    try:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        logger.success("Ollama installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Ollama: {e}")
def list_existing_models():
    try:
        existing_models = ollama.list()
        existing_model_names = [m.model for m in existing_models['models']]
        return existing_model_names
    except Exception as e:
        logger.error(f"Failed to fetch existing models: {e}")
        return []

def ensure_ollama_installed():
    try:
        subprocess.run(["ollama", "--version"], check=True)
    except FileNotFoundError:
        logger.info("Ollama is not installed.")
        install_ollama()
    except subprocess.CalledProcessError:
        logger.warning("Ollama is installed but not working correctly. Try reinstalling.")
        install_ollama()
    else:
        logger.info("Ollama already installed.")

def ensure_model_pulled(model_name):
    existing_model_names = list_existing_models()
    logger.info(f"Existing models: {existing_model_names}")
    
    if model_name not in existing_model_names:
        logger.info(f"Pulling model {model_name}…")
        ollama.pull(model_name)
        logger.success(f"Model {model_name} pulled successfully.")
    else:
        logger.info(f"Model {model_name} already present.")

def start_ollama_service(host, timeout=60, interval=0.5):
    """Start Ollama server in background (if needed) and wait for it."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        if r.ok:
            logger.info("Ollama server is already running.")
            return True
    except requests.ConnectionError:
        logger.info("Ollama server not running. Starting it...")

    # start in background
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{host}/api/tags")
            if r.ok:
                logger.success("Ollama server started successfully.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)

    logger.error("Timed out waiting for Ollama server to start.")
    return False

def delete_model(model_name: str):
    logger.info(f"Attempting to delete model: {model_name}…")
    
    try:
        existing_models = list_existing_models()
        if model_name not in existing_models:
            logger.info(f"Model {model_name} not found. Skipping deletion.")
            return
        
        ollama.delete(model_name)
        logger.success(f"Successfully deleted model: {model_name}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")



if __name__ == "__main__":
    config = OllamaConfig()
    ensure_ollama_installed()
    if not start_ollama_service(config.host):
        raise RuntimeError("Ollama server didn’t start in time")
    
    ensure_model_pulled(model_name=config.default_embedding_model)
    ensure_model_pulled(model_name=config.default_generation_model)
    delete_model(model_name=config.default_embedding_model)
    delete_model(model_name=config.default_generation_model)