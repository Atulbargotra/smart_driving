from gtts import gTTS
import playsound
def speak(text_input,language='hi'):
    tts = gTTS(text=text_input, lang=language)
    tts.save("sounds/output.mp3")
    playsound.playsound('sounds/output.mp3')
