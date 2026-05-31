# STT

A FastAPI-based service for real-time speech-to-text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and WebRTC VAD.



https://github.com/user-attachments/assets/83c49dcb-e263-43f4-96fc-1430650a5689



## Installation

```bash
# Install system requirements
sudo apt install portaudio19-dev

# Install python dependencies
python3 src/setup.py
source src/stt-venv/bin/activate
```

## Usage

**Start the service:**

```bash
cd src/
python app.py
```

**Python example:**

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:47102/transcribe",
        files={"file": f}
    )
    print(response.json()["text"])
```
