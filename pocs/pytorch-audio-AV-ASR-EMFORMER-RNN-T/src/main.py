import numpy as np
import sentencepiece as spm
import torch
import torchaudio
import torchvision

device = torch.device('cpu')

def stream(q, format, option, src, segment_length, sample_rate):
    print("Building StreamReader...")
    streamer = torchaudio.io.StreamReader(src=src, format=format, option=option)
    streamer.add_basic_video_stream(frames_per_chunk=segment_length, buffer_chunk_size=500, width=600, height=340)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length * 640, sample_rate=sample_rate)

    print(streamer.get_src_stream_info(0))
    print(streamer.get_src_stream_info(1))
    print("Streaming...")
    print()
    for (chunk_v, chunk_a) in streamer.stream(timeout=-1, backoff=1.0):
        q.put([chunk_v.to(device), chunk_a.to(device)])


class ContextCacher:
    def __init__(self, segment_length: int, context_length: int, rate_ratio: int):
        self.segment_length = segment_length
        self.context_length = context_length

        self.context_length_v = context_length
        self.context_length_a = context_length * rate_ratio
        self.context_v = torch.zeros([self.context_length_v, 3, 340, 600])
        self.context_a = torch.zeros([self.context_length_a, 1])

    def __call__(self, chunk_v, chunk_a):
        if chunk_v.size(0) < self.segment_length:
            chunk_v = torch.nn.functional.pad(chunk_v, (0, 0, 0, 0, 0, 0, 0, self.segment_length - chunk_v.size(0)))
        if chunk_a.size(0) < self.segment_length * 640:
            chunk_a = torch.nn.functional.pad(chunk_a, (0, 0, 0, self.segment_length * 640 - chunk_a.size(0)))

        if self.context_length == 0:
            return chunk_v.float(), chunk_a.float()
        else:
            chunk_with_context_v = torch.cat((self.context_v, chunk_v))
            chunk_with_context_a = torch.cat((self.context_a, chunk_a))
            self.context_v = chunk_v[-self.context_length_v :]
            self.context_a = chunk_a[-self.context_length_a :]
            return chunk_with_context_v.float(), chunk_with_context_a.float()

import sys

sys.path.insert(0, "../../examples")

from avsr.data_prep.detectors.mediapipe.detector import LandmarksDetector
from avsr.data_prep.detectors.mediapipe.video_process import VideoProcess


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class Preprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess()
        self.video_transform = torch.nn.Sequential(
            FunctionalModule(
                lambda n: [(lambda x: torchvision.transforms.functional.resize(x, 44, antialias=True))(i) for i in n]
            ),
            FunctionalModule(lambda x: torch.stack(x)),
            torchvision.transforms.Normalize(0.0, 255.0),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def forward(self, audio, video):
        video = video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video).permute(0, 3, 1, 2).float()
        video = self.video_transform(video)
        audio = audio.mean(axis=-1, keepdim=True)
        return audio, video

class SentencePieceTokenProcessor:
    def __init__(self, sp_model):
        self.sp_model = sp_model
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens, lstrip: bool = True) -> str:
        filtered_hypo_tokens = [
            token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list
        ]
        output_string = "".join(self.sp_model.id_to_piece(filtered_hypo_tokens)).replace("\u2581", " ")

        if lstrip:
            return output_string.lstrip()
        else:
            return output_string


class InferencePipeline(torch.nn.Module):
    def __init__(self, preprocessor, model, decoder, token_processor):
        super().__init__()
        self.preprocessor = preprocessor
        self.model = model
        self.decoder = decoder
        self.token_processor = token_processor

        self.state = None
        self.hypotheses = None

    def forward(self, audio, video):
        audio, video = self.preprocessor(audio, video)
        feats = self.model(audio.unsqueeze(0), video.unsqueeze(0))
        length = torch.tensor([feats.size(1)], device=audio.device)
        self.hypotheses, self.state = self.decoder.infer(feats, length, 10, state=self.state, hypothesis=self.hypotheses)
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript


def _get_inference_pipeline(model_path, spm_model_path):
    model = torch.jit.load(model_path)
    model.eval()

    sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)
    token_processor = SentencePieceTokenProcessor(sp_model)

    decoder = torchaudio.models.RNNTBeamSearch(model.model, sp_model.get_piece_size())

    return InferencePipeline(
        preprocessor=Preprocessing(),
        model=model,
        decoder=decoder,
        token_processor=token_processor,
    )

from torchaudio.utils import download_asset


def main(device, src, option=None):
    print("Building pipeline...")
    model_path = download_asset("tutorial-assets/device_avsr_model.pt")
    spm_model_path = download_asset("tutorial-assets/spm_unigram_1023.model")

    pipeline = _get_inference_pipeline(model_path, spm_model_path)

    BUFFER_SIZE = 32
    segment_length = 8
    context_length = 4
    sample_rate = 19200
    frame_rate = 30
    rate_ratio = sample_rate // frame_rate
    cacher = ContextCacher(BUFFER_SIZE, context_length, rate_ratio)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")

    @torch.inference_mode()
    def infer():
        num_video_frames = 0
        video_chunks = []
        audio_chunks = []
        while True:
            chunk_v, chunk_a = q.get()
            num_video_frames += chunk_a.size(0) // 640
            video_chunks.append(chunk_v)
            audio_chunks.append(chunk_a)
            if num_video_frames < BUFFER_SIZE:
                continue
            video = torch.cat(video_chunks)
            audio = torch.cat(audio_chunks)
            video, audio = cacher(video, audio)
            pipeline.state, pipeline.hypotheses = None, None
            transcript = pipeline(audio, video.float())
            print(transcript, end="", flush=True)
            num_video_frames = 0
            video_chunks = []
            audio_chunks = []

    q = ctx.Queue()
    p = ctx.Process(target=stream, args=(q, device, option, src, segment_length, sample_rate))
    p.start()
    infer()
    p.join()


if __name__ == "__main__":
    main(
        device="avfoundation",
        src="0:1",
        option={"framerate": "30", "pixel_format": "rgb24"},
    )
