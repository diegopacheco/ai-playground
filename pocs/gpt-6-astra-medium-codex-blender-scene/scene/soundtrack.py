import math
import random
import struct
import sys
import wave
from pathlib import Path


def compose(path):
    rate = 44100
    duration = 16
    samples = [0.0] * (duration * rate)
    rng = random.Random(27)

    def note(start, length, frequency, volume, voice):
        for index in range(int(length * rate)):
            target = int(start * rate) + index
            if target >= len(samples):
                break
            t = index / rate
            envelope = min(1, t * 180) * math.exp(-t * (11 if voice == 'pluck' else 5))
            if voice == 'hat':
                value = rng.uniform(-1, 1) * math.exp(-t * 65)
            elif voice == 'kick':
                value = math.sin(2 * math.pi * (48 * t + 2 * (1 - math.exp(-t * 35)))) * math.exp(-t * 19)
            else:
                value = math.sin(2 * math.pi * frequency * t) + 0.3 * math.sin(4 * math.pi * frequency * t) + 0.14 * math.sin(6 * math.pi * frequency * t)
            samples[target] += value * envelope * volume

    melody = [74, 78, 81, 78, 76, 74, 69, 71, 74, 78, 83, 81, 78, 76, 74, 69]
    bass = [38, 38, 43, 45]
    for beat in range(32):
        start = beat * 0.5
        note(start, 0.45, 440 * 2 ** ((bass[(beat // 4) % 4] - 69) / 12), 0.29, 'bass')
        note(start, 0.22, 0, 0.36, 'kick')
        note(start + 0.25, 0.12, 0, 0.12, 'hat')
        for off in range(2):
            pitch = melody[(beat * 2 + off) % len(melody)]
            note(start + off * 0.25, 0.3, 440 * 2 ** ((pitch - 69) / 12), 0.2, 'pluck')
    peak = max(abs(value) for value in samples)
    pcm = bytearray()
    for index, value in enumerate(samples):
        fade = min(1, index / (rate * 0.05), (len(samples) - index) / (rate * 0.4))
        pcm.extend(struct.pack('<h', int(value / peak * 27000 * fade)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)
    print(f'Saved original 16-second instrumental: {path}')


if __name__ == '__main__':
    compose(sys.argv[1])
