import subprocess
import os
import signal
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PORT = config["streamlit"]["port"]

def start_fastapi():
    fastapi_cmd = f"uvicorn app:app --reload --port {PORT}"
    subprocess.Popen(fastapi_cmd, shell=True)


def start_streamlit():
    streamlit_cmd = "streamlit run streamlit_app.py"
    subprocess.Popen(streamlit_cmd, shell=True)

if __name__ == "__main__":
    try:
        fastapi_process = start_fastapi()
        streamlit_process = start_streamlit()

    except KeyboardInterrupt:
        fastapi_process.terminate()
        streamlit_process.terminate()