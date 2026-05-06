import subprocess
import json

OPENCLAW_BIN = "/opt/homebrew/bin/openclaw"

def ask_openclaw(session_id: str, prompt: str):
    result = subprocess.run(
        [
            OPENCLAW_BIN,
            "agent",
            "--local",
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--json"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {
            "success": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    try:
        output = result.stdout if result.stdout.strip() else result.stderr
        data = json.loads(output)

        text = ""
        for payload in data.get("payloads", []):
            text += payload.get("text", "") + "\n"

        return {
            "success": True,
            "response": text.strip(),
            "raw": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "returncode": result.returncode,
            "output": output
        }
