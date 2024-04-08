import torch
import torchaudio
import os
import IPython
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchaudio.__version__)

_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()

dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

def plot_wave(i=1):
    waveform, sample_rate, label = dataset[i]
    plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    IPython.display.Audio(waveform, rate=sample_rate)

plot_wave()
plot_wave(3)
plot_wave(5)

