import os
import time
import logging
import argparse
import RPi.GPIO as GPIO
import textwrap
import re
import sys
from pathlib import Path
#log_file_path = str(Path(__file__).resolve().parent.joinpath('', 'bot_log.txt'))
#sys.stdout = open(log_file_path, 'w')

from dotenv import load_dotenv
load_dotenv()

# Translations
trans_hu = {"listening" : "FÜLEL", "thinking" : "GONDOL", "speaking" : "BESZÉL", "silent":"CSENDBEN", "lang": "Hungarian"}
trans_en = {"listening" : "LISTENING", "thinking" : "THINKING", "speaking" : "SPEAKING", "silent":"SILENT", "lang": "English"}
trans_de = {"listening" : "HÖREN", "thinking" : "DENKEN", "speaking" : "SPRECHEN", "silent":"STILL", "lang": "German"}
translation = {}
translation["hu"] = trans_hu
translation["en"] = trans_en
translation["de"] = trans_de

lang_switch_phrases = [
    {"language": "English", 
        "stt_language": "en", 
        "phrases": ["válts angolra", "beszélj angolul", "válaszolj angolul", "angolul válaszolj", "angolul beszélj"]},
    {"language": "Hungarian", 
        "stt_language": "hu", 
        "phrases": ["switch to hungarian", "respond in hungarian", "use hungarian", "talk in hungarian", "talk hungarian", "speak in hungarian", "speak hungarian"]},
    {"language": "German", 
        "stt_language": "de", 
        "phrases": ["válts németre", "beszélj németül", "válaszolj németül", "németül válaszolj", "németül beszélj", "switch to german", "respond in german", "use german", "talk in german", "talk german", "speak in german", "speak german"]  }
]

# Local classes
from utils import Utils
utils = Utils()

from botconfig import BotConfig
bot_config = BotConfig()

from gptchatservice import GPTChatService
from groq_audio_service import GroqAudioService

from lcdservice import LCDServiceColor
lcd_service = LCDServiceColor()

# Audio HW settings
output_device_name="sysdefault:CARD=UACDemoV10"
input_device_name="hw:CARD=WEBCAM"
mute_mic_during_tts = True

# Global variable for stopping execution
done = False 
listening = True
thinking = False
speaking = False
ui_lang = "en"
assistant_language = "English"
groq_audio_service = None

# Statistics
total_tts_duration = 0
total_stt_chars = 0
program_start_time = 0

def check_single_char_dot(string):
    pattern = r'^[a-zA-Z0-9]\.'
    match = re.search(pattern, string)
    if match:
        return True
    else:
        return False
    
# Handle a completed Groq transcription
def recognized(stt_text):

    try:

        global thinking
        global speaking
        global total_stt_chars

        if (speaking == True or thinking == True):
            return;
        
        if stt_text == "" or check_single_char_dot(stt_text): return
 
        recognized_text_log = f"Recognized speech: {stt_text}"
        print(recognized_text_log, flush=True)
        log.info(recognized_text_log)

        total_stt_chars += len(stt_text)

        if (mute_mic_during_tts): utils.mute_mic(device_name=input_device_name)
        
        if (bot_config.change_face == True):
            change_mood_thinking(stt_text)
           
        thinking = True
        start = time.time()

        if (bot_config.exp_lang_autoswitch == True):
            lang_switcher = check_lang_switch_phrases(stt_text)
            if (lang_switcher != None):
                change_language(lang_switcher)
                print(f"Language switched to {lang_switcher['language']}")
                log.info(f"Language switched to {lang_switcher['language']}")
                stt_text = f" From now on, you will have to respond in {lang_switcher['language']}! So please respond in {lang_switcher['language']}. Acknowledge this by saying that you will speak now in {lang_switcher['language']}"

        response_text = gpt_service.ask(stt_text)
        groq_call_duration = f'Groq API call ended: {time.time() - start} ms'
        print(groq_call_duration, flush=True)
        log.debug(groq_call_duration)
        thinking = False
        
        if (bot_config.change_face == True):
            change_mood_talking(response_text)
            time.sleep(0.5) 
        
        start = time.time()
        speak_text(response_text)

        if (listening == False):
            return
        
        print("Speak!")
        if (bot_config.auto_mute_mic == True): 
            toggle_mute(False)
        else:
            toggle_mute(True)
            
    except Exception as e:
        log.error(e)
        if hasattr(e, 'message'):
            print(e.message)
            log.error(e.message)
        else:
            print(e)          
        return "" 
    finally:
        thinking = False
        if listening and not bot_config.auto_mute_mic and mute_mic_during_tts:
            utils.unmute_mic(device_name=input_device_name)

def change_mood_thinking(top_text):
    wrapper = textwrap.TextWrapper(width=70)
    text_wrapped = wrapper.fill(text=top_text)
    if (bot_config.show_recognized == False): 
        text_wrapped = ''
    lcd_service.draw_face(face=LCDServiceColor.FACE_THINK, icon=LCDServiceColor.ICON_LOAD, additional_text=translation[ui_lang]['thinking'], top_small_text=text_wrapped)

def change_mood_talking(top_text):
    wrapper = textwrap.TextWrapper(width=70)
    text_wrapped = wrapper.fill(text=top_text)
    if (bot_config.show_gpt_response == False):
        text_wrapped = ''    
    lcd_service.draw_face(face=LCDServiceColor.FACE_TALK, icon=LCDServiceColor.ICON_SPEAKER, additional_text=translation[ui_lang]['speaking'], top_small_text=text_wrapped)

def speak_text(text):
    global speaking
    global total_tts_duration
    speaking = True
    try:
        duration = groq_audio_service.speak(text, should_stop=lambda: not listening)
        total_tts_duration += duration
        print(f"AI response ({duration:.1f} sec): {text}")
    finally:
        speaking = False

def run_ai():

    log.info("Started bot...")

    global program_start_time

    if (mute_mic_during_tts): utils.unmute_mic(device_name=input_device_name)
    groq_audio_service.start()

    program_start_time = time.time()
  
    if (bot_config.auto_mute_mic == True): 
        toggle_mute(False)
    else:
        print("Speak!")
        lcd_service.draw_face(face=LCDServiceColor.FACE_LISTEN, icon=LCDServiceColor.ICON_MIC, additional_text=translation[ui_lang]['listening'])

    while not done:
        if groq_audio_service.error is not None:
            raise RuntimeError("Groq audio listener stopped") from groq_audio_service.error
        time.sleep(.5)

def init_logging():
    global log
    log = logging.getLogger("bot_log")
    logging.basicConfig(filename='gpt_service.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def check_internet():
        
    if (utils.has_internet() == True):
        return
        
    time.sleep(1)

    while not utils.has_internet():
        lcd_service.draw_large_icon(LCDServiceColor.ICON_ERROR, "Waiting for internet connect...")
        time.sleep(1)
    
    lcd_service.draw_large_icon(LCDServiceColor.ICON_WIFI, "Internet connection found!")
    time.sleep(1)
    lcd_service.clear_screen()    

def check_lang_switch_phrases(input_text):
    for phrase_item in lang_switch_phrases:
        for phrase_text in phrase_item["phrases"]:
            if re.search(phrase_text, input_text, re.IGNORECASE):
                lang_switcher = {
                    "language": phrase_item["language"],
                    "stt_language": phrase_item["stt_language"],
                }
                return lang_switcher
    return None

def change_language(lang_switcher):
    global assistant_language
    assistant_language = lang_switcher["language"]
    groq_audio_service.language = lang_switcher["stt_language"]
    gpt_service.change_language(lang_switcher["language"])
    tts_language = (
        "Arabic"
        if groq_audio_service.tts_model == "canopylabs/orpheus-arabic-saudi"
        else "English"
    )
    if assistant_language != tts_language:
        log.warning(
            "Groq Orpheus TTS speaks %s, but the assistant language was changed to %s",
            tts_language,
            assistant_language,
        )


def init_groq_audio(voice):
    global ui_lang
    global assistant_language
    global groq_audio_service
    ui_lang = "en"
    tts_model = os.getenv(
        "GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"
    )
    assistant_language = (
        "Arabic"
        if tts_model == "canopylabs/orpheus-arabic-saudi"
        else "English"
    )
    groq_audio_service = GroqAudioService(
        input_device=input_device_name,
        output_device=output_device_name,
        voice=voice,
        on_transcription=recognized,
        is_listening=lambda: listening and not thinking and not speaking,
        logger=log,
    )
    
def init_gpio():
    GPIO.setmode(GPIO.BCM) # Use physical pin numbering
    GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
    GPIO.add_event_detect(15,GPIO.RISING,callback=button_pushed, bouncetime=500) # Setup event on pin 10 rising edge
    

def button_pushed(channel):

    global thinking
    if (thinking == True):
        return
    
    global listening
    listening = not listening
    toggle_mute(listening)

def toggle_mute(listening_local):
    global listening
    if (listening_local == True):
        listening = True
        if (bot_config.change_face == True):
            lcd_service.draw_face(face=LCDServiceColor.FACE_LISTEN, icon=LCDServiceColor.ICON_MIC, additional_text=translation[ui_lang]['listening'])
        if (mute_mic_during_tts): utils.unmute_mic(device_name=input_device_name)
    if (listening_local == False):
        listening = False
        if (bot_config.change_face == True):
            lcd_service.draw_face(face=LCDServiceColor.FACE_SILENT, icon=LCDServiceColor.ICON_MIC_OFF, additional_text=translation[ui_lang]['silent'])  
        if (mute_mic_during_tts): utils.mute_mic(device_name=input_device_name)


def init_ai():
    global gpt_service
    gpt_service = GPTChatService(assistant_language)
    print(assistant_language)

def end_program(write_stats = True):

    if groq_audio_service is not None:
        groq_audio_service.stop()
    lcd_service.clear_screen()
    GPIO.cleanup()     

    if (write_stats):
        global gpt_service
        global program_start_time
        global total_stt_chars
        global total_tts_duration

        program_end_time = time.time()
        program_run_duration = program_end_time - program_start_time
        print(f'STATS: program duration: {program_run_duration} seconds')
        print(f'STATS: total TTS duration: {total_tts_duration} sec')
        print(f'STATS: total STT characters: {total_stt_chars} chars')    
        if "gpt_service" in globals():
            print(f'STATS: total Groq API tokens: {gpt_service.get_stats()}')
    


def main():
    try:
        init_gpio()
        init_logging()
        check_internet()
        init_groq_audio(bot_config.voice_name)
        init_ai()        
        run_ai() 
    except KeyboardInterrupt:
        end_program()   
    finally:
        end_program()       
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--audio_input", help='Audio input device')
    parser.add_argument("-o", "--audio_output", help='Audio output device')
    parser.add_argument("-r", "--record_card", help='Audio record card')
    args = parser.parse_args()
    if (args.audio_input is not None):
        input_device_name=args.audio_input
        print(f"Audio input override: {input_device_name}")
    if (args.audio_output is not None):
        output_device_name=args.audio_output  
        print(f"Audio output override: {output_device_name}")  
        
    main()
