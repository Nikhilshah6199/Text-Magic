import os 
from datetime import datetime

def get_current_time():
    fmt = "%Y-%m-%d %H%M%S"
    return f"{datetime.now().strftime(fmt)}"


ROOT_DIR = os.getcwd()
CURRENT_TIME_STAMP = get_current_time()

# Application configuration
APPLICATION_NAME = "text_to_speech"
ARTIFACT_DIR_KEY = "artifact"
AUDIO_DIR = "tts_audio"
TEXT_DIR = "tts_text"
TEXT_FILENAME = "input_texts.txt"
