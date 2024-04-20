from transformers import AutoProcessor, BarkModel
import soundfile as sf

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("Brazil! It's not for amateurs", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

# save the audio
sf.write("output.wav", audio_array, 22050)
print("Audio saved to output.wav")
