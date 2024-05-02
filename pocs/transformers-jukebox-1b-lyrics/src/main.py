from transformers import AutoTokenizer, JukeboxModel, set_seed
import torch
import soundfile as sf

metas = dict(artist="The Rolling Stones", genres="Rock", lyrics="Agile training, we're learning fast, coding in the zone at last")
tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()

labels = tokenizer(**metas)["input_ids"]
set_seed(42)
zs = [torch.zeros(1, 0, dtype=torch.long) for _ in range(3)]
zs = model._sample(zs, labels, [0], sample_length=264600, save_results=False)

# Decode the tensor to audio
audio = model.decode(zs)

# Convert the tensor to a numpy array
audio_np = audio[0].data.cpu().numpy()

# Save the audio to a .wav file
sf.write('song.mp3', audio_np, 44100)