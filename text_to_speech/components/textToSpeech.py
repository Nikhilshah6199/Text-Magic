import os
import sys
import base64
import datetime
from gtts import gTTS
from text_to_speech.constants import *
from text_to_speech.entity.config_entity import TTSConfig

from text_to_speech.entity.config_entity import TTSConfig

class TTSapplication:
    def __init__(self, app_config=TTSConfig()):
        """
        Initializes the application by loading the application configuration.
        """
        
        self.app_info = app_config
        self.artifact_dir = self.app_info.artifact_dir
        self.audio_dir = self.app_info.audio_dir
        self.text_dir = self.app_info.text_dir
    
        
    def text2Speech(self, data, accent):
        """
        It takes in a Text string and an accent, writes the string to a text file, creates a gTTS object, saves
        the gTTS object as an mp3 file, and then returns the mp3 file as a base64 encoded string
        
        Args:
          data: The text that you want to convert to speech.
          accent: This is the accent of the voice.
        
        Returns:
          The return value is a base64 encoded string.
        """
   
        my_text = data
        
        text_dir = self.text_dir
        txt_filename = TEXT_FILENAME
        
        txt_file_path = os.path.join(text_dir, txt_filename)
        
        with open(txt_file_path, "a+") as file:
            file.write(f'\n{my_text}')
            
        # Create object for Text to speech
        tts = gTTS(text=my_text, lang='en', slow=False, tld=accent)
        
        audio_dir = self.audio_dir
        
        filename = f"converted_file_{CURRENT_TIME_STAMP}.mp3"
        
        audiopath = os.path.join(audio_dir, filename)
        
        # save tts object as mp3
        tts.save(audiopath)

        with open(audiopath, "rb") as file:
            my_string = base64.b64encode(file.read())
        return my_string
