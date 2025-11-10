

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from utils.env import safe_load_env

def test_openai_connection():
    """
    Simple test to verify OpenAI API connectivity.
    Make sure OPENAI_API_KEY is set in your environment.
    """
    safe_load_env(
        paths=[
            Path(".env"),
            Path(__file__).resolve().parents[1] / ".env",
        ],
        override=False,
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please export it before running this test.")
        return

    client = OpenAI(api_key=api_key)

    print("✅ Testing OpenAI connection...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello and confirm the API connection works."}
            ],
            temperature=0.3,
        )
        message = response.choices[0].message.content
        print(f"✅ Connection successful!\n\nLLM says:\n{message}\n")
    except Exception as e:
        print(f"❌ Error connecting to OpenAI API:\n{e}\n")

if __name__ == "__main__":
    test_openai_connection()