import subprocess
import platform
import os
import sys

#CODE NOT TESTED, JUST FOR REFERENCE FOR AN IMPLEMENTATION ON DIFFERENT PLATFORMS

def ensure_ollama_installed():
    # 1) Check if Ollama is already installed
    try:
        subprocess.run(["ollama", "--version"], check=True, stdout=subprocess.DEVNULL)
        print("Ollama is already installed.")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Either the command returned non‐zero, or "ollama" wasn't found on PATH
        print("Ollama not found; installing…")

    # 2) Install
    system = platform.system()
    try:
        if system == "Windows":
            installer_url = "https://ollama.com/download/OllamaSetup.exe"
            installer_path = os.path.join(os.getcwd(), "OllamaSetup.exe")
            # Download the installer via PowerShell
            subprocess.run([
                "powershell", "-Command",
                f"Invoke-WebRequest -Uri '{installer_url}' -OutFile '{installer_path}'"
            ], check=True)
            # Launch the installer (this will prompt the user)
            subprocess.run([installer_path], check=True)
        else:
            # Unix / macOS
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True, check=True
            )
        print("Ollama installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed (exit code {e.returncode}): {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
