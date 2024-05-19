from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)
torchaudio.save("80s_pop_track.wav", audio_values[0], 16000)
print("Saved 80s_pop_track.wav")