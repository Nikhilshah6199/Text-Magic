from dataclasses import dataclass
from text_to_speech.constants import *
from from_root import from_root


CURRENT_DIR = os.getcwd()
@dataclass
class TTSConfig:
    app_name: str = APPLICATION_NAME
    artifact_dir: str = os.path.join(CURRENT_DIR, APPLICATION_NAME, ARTIFACT_DIR_KEY)
    audio_dir: str = os.path.join(artifact_dir, AUDIO_DIR)
    text_dir: str = os.path.join(artifact_dir, TEXT_DIR)

