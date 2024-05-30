from gtts import gTTS
from pydub import AudioSegment

# The first three sentences of the Brazilian National Anthem
text = """
Ouviram do Ipiranga as margens plácidas
De um povo heroico o brado retumbante,
E o sol da liberdade, em raios fúlgidos,
"""

# Create a gTTS text-to-speech object
tts = gTTS(text, lang='pt-br')

# Save the audio to a temporary MP3 file
tts.save("temp_audio_file.mp3")

# Load the MP3 file
audio = AudioSegment.from_mp3("temp_audio_file.mp3")

# Convert to WAV and save
audio.export("audio_file.wav", format="wav")