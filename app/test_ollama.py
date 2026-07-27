"""
Ollama connectivity test for the offline multimodal RAG system.

Sends a simple prompt to the locally running Ollama server and prints
the response. Run this to verify your Ollama setup before using the app.

Usage:
    python app/test_ollama.py
"""

import sys
import httpx
import ollama

# config.py lives at the project root; ensure it's importable when running
# this script directly from any working directory.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

TEST_PROMPT = "Say hello in one sentence."


def test_ollama() -> None:
    print(f"Connecting to Ollama at : {OLLAMA_BASE_URL}")
    print(f"Model                   : {OLLAMA_MODEL}")
    print(f"Prompt                  : {TEST_PROMPT}\n")

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )
        print("Response:", response["message"]["content"].strip())

    except httpx.ConnectError:
        print("ERROR: Cannot reach Ollama server.")
        print("  1. Start the server  : ollama serve")
        print(f"  2. Pull the model    : ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    except ollama.ResponseError as e:
        # Raised when the server is up but the model isn't available
        if "not found" in str(e).lower() or "pull" in str(e).lower():
            print(f"ERROR: Model '{OLLAMA_MODEL}' is not available locally.")
            print(f"  Pull it with: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"ERROR: Ollama returned an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_ollama()
