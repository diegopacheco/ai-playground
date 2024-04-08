import torch
import torchaudio
import os
import time
import matplotlib.pyplot as plt
from torchaudio.io import StreamReader
from torchaudio.utils import ffmpeg_utils

print(torch.__version__)
print(torchaudio.__version__)

print("FFmpeg Library versions:")
for k, ver in ffmpeg_utils.get_versions().items():
    print(f"  {k}:\t{'.'.join(str(v) for v in ver)}")

print("Available NVDEC Decoders:")
for k in ffmpeg_utils.get_video_decoders().keys():
    if "cuvid" in k:
        print(f" - {k}")

print("Avaialbe GPU:")
print(torch.cuda.get_device_properties(0))

src = torchaudio.utils.download_asset(
    "tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
)

s = StreamReader(src)
s.add_video_stream(5, decoder="h264_cuvid")
s.fill_buffer()
(video,) = s.pop_chunks()
print(video.shape, video.dtype)
print(video.device)

s = StreamReader(src)
s.add_video_stream(5, decoder="h264_cuvid", hw_accel="cuda:0")
s.fill_buffer()
(video,) = s.pop_chunks()
print(video.shape, video.dtype, video.device)

# Video data is sent to CUDA device 0, decoded and
# converted on the same device.
s.add_video_stream(
    ...,
    decoder="h264_cuvid",
    decoder_option={"gpu": "0"},
    hw_accel="cuda:0",
)

# Video data is sent to CUDA device 0, and decoded there.
# Then it is transfered to CUDA device 1, and converted to
# CUDA tensor.
s.add_video_stream(
    ...,
    decoder="h264_cuvid",
    decoder_option={"gpu": "0"},
    hw_accel="cuda:1",
)

def test_decode(decoder: str, seek: float):
    s = StreamReader(src)
    s.seek(seek)
    s.add_video_stream(1, decoder=decoder)
    s.fill_buffer()
    (video,) = s.pop_chunks()
    return video[0]
timestamps = [12, 19, 45, 131, 180]

cpu_frames = [test_decode(decoder="h264", seek=ts) for ts in timestamps]
cuda_frames = [test_decode(decoder="h264_cuvid", seek=ts) for ts in timestamps]

def yuv_to_rgb(frames):
    frames = frames.cpu().to(torch.float)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], -1)
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rgb.numpy()

def plot():
    n_rows = len(timestamps)
    fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
    for i in range(n_rows):
        axes[i][0].imshow(yuv_to_rgb(cpu_frames[i]))
        axes[i][1].imshow(yuv_to_rgb(cuda_frames[i]))

    axes[0][0].set_title("Software decoder")
    axes[0][1].set_title("HW decoder")
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()


plot()

def test_options(option):
    s = StreamReader(src)
    s.seek(87)
    s.add_video_stream(1, decoder="h264_cuvid", hw_accel="cuda:0", decoder_option=option)
    s.fill_buffer()
    (video,) = s.pop_chunks()
    print(f"Option: {option}:\t{video.shape}")
    return video[0]
original = test_options(option=None)
resized = test_options(option={"resize": "480x270"})
cropped = test_options(option={"crop": "135x135x240x240"})
cropped_and_resized = test_options(option={"crop": "135x135x240x240", "resize": "640x360"})

def plot():
    fig, axes = plt.subplots(2, 2, figsize=[12.8, 9.6])
    axes[0][0].imshow(yuv_to_rgb(original))
    axes[0][1].imshow(yuv_to_rgb(resized))
    axes[1][0].imshow(yuv_to_rgb(cropped))
    axes[1][1].imshow(yuv_to_rgb(cropped_and_resized))

    axes[0][0].set_title("Original")
    axes[0][1].set_title("Resized")
    axes[1][0].set_title("Cropped")
    axes[1][1].set_title("Cropped and resized")
    plt.tight_layout()
    return fig


plot()

# ffmpeg -y -f lavfi -t 12.05 -i mptestsrc -movflags +faststart mptestsrc.mp4
test_src = torchaudio.utils.download_asset("tutorial-assets/mptestsrc.mp4")

def decode_resize_ffmpeg(mode, height, width, seek):
    filter_desc = None if mode is None else f"scale={width}:{height}:sws_flags={mode}"
    s = StreamReader(test_src)
    s.add_video_stream(1, filter_desc=filter_desc)
    s.seek(seek)
    s.fill_buffer()
    (chunk,) = s.pop_chunks()
    return chunk

def decode_resize_cuvid(height, width, seek):
    s = StreamReader(test_src)
    s.add_video_stream(1, decoder="h264_cuvid", decoder_option={"resize": f"{width}x{height}"}, hw_accel="cuda:0")
    s.seek(seek)
    s.fill_buffer()
    (chunk,) = s.pop_chunks()
    return chunk.cpu()

params = {"height": 224, "width": 224, "seek": 3}

frames = [
    decode_resize_ffmpeg(None, **params),
    decode_resize_ffmpeg("neighbor", **params),
    decode_resize_ffmpeg("bilinear", **params),
    decode_resize_ffmpeg("bicubic", **params),
    decode_resize_cuvid(**params),
    decode_resize_ffmpeg("spline", **params),
    decode_resize_ffmpeg("lanczos:param0=1", **params),
    decode_resize_ffmpeg("lanczos:param0=3", **params),
    decode_resize_ffmpeg("lanczos:param0=5", **params),
]

def plot():
    fig, axes = plt.subplots(3, 3, figsize=[12.8, 15.2])
    for i, f in enumerate(frames):
        h, w = f.shape[2:4]
        f = f[..., : h // 4, : w // 4]
        axes[i // 3][i % 3].imshow(yuv_to_rgb(f[0]))
    axes[0][0].set_title("Original")
    axes[0][1].set_title("nearest neighbor")
    axes[0][2].set_title("bilinear")
    axes[1][0].set_title("bicubic")
    axes[1][1].set_title("NVDEC")
    axes[1][2].set_title("spline")
    axes[2][0].set_title("lanczos(1)")
    axes[2][1].set_title("lanczos(3)")
    axes[2][2].set_title("lanczos(5)")

    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()


plot()

def test_decode_cuda(src, decoder, hw_accel="cuda", frames_per_chunk=5):
    s = StreamReader(src)
    s.add_video_stream(frames_per_chunk, decoder=decoder, hw_accel=hw_accel)

    num_frames = 0
    chunk = None
    t0 = time.monotonic()
    for (chunk,) in s.stream():
        num_frames += chunk.shape[0]
    elapsed = time.monotonic() - t0
    print(f" - Shape: {chunk.shape}")
    fps = num_frames / elapsed
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return fps

def test_decode_cpu(src, threads, decoder=None, frames_per_chunk=5):
    s = StreamReader(src)
    s.add_video_stream(frames_per_chunk, decoder=decoder, decoder_option={"threads": f"{threads}"})

    num_frames = 0
    device = torch.device("cuda")
    t0 = time.monotonic()
    for i, (chunk,) in enumerate(s.stream()):
        if i == 0:
            print(f" - Shape: {chunk.shape}")
        num_frames += chunk.shape[0]
        chunk = chunk.to(device)
    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return fps

def run_decode_tests(src, frames_per_chunk=5):
    fps = []
    print(f"Testing: {os.path.basename(src)}")
    for threads in [1, 4, 8, 16]:
        print(f"* Software decoding (num_threads={threads})")
        fps.append(test_decode_cpu(src, threads))
    print("* Hardware decoding")
    fps.append(test_decode_cuda(src, decoder="h264_cuvid"))
    return fps

src_qvga = torchaudio.utils.download_asset("tutorial-assets/testsrc2_qvga.h264.mp4")
fps_qvga = run_decode_tests(src_qvga)

src_vga = torchaudio.utils.download_asset("tutorial-assets/testsrc2_vga.h264.mp4")
fps_vga = run_decode_tests(src_vga)

src_xga = torchaudio.utils.download_asset("tutorial-assets/testsrc2_xga.h264.mp4")
fps_xga = run_decode_tests(src_xga)

def plot():
    fig, ax = plt.subplots(figsize=[9.6, 6.4])

    for items in zip(fps_qvga, fps_vga, fps_xga, "ov^sx"):
        ax.plot(items[:-1], marker=items[-1])
    ax.grid(axis="both")
    ax.set_xticks([0, 1, 2], ["QVGA (320x240)", "VGA (640x480)", "XGA (1024x768)"])
    ax.legend(
        [
            "Software Decoding (threads=1)",
            "Software Decoding (threads=4)",
            "Software Decoding (threads=8)",
            "Software Decoding (threads=16)",
            "Hardware Decoding (CUDA Tensor)",
        ]
    )
    ax.set_title("Speed of processing video frames")
    ax.set_ylabel("Frames per second")
    plt.tight_layout()

plot()

def test_decode_then_resize(src, height, width, mode="bicubic", frames_per_chunk=5):
    s = StreamReader(src)
    s.add_video_stream(frames_per_chunk, decoder_option={"threads": "8"})

    num_frames = 0
    device = torch.device("cuda")
    chunk = None
    t0 = time.monotonic()
    for (chunk,) in s.stream():
        num_frames += chunk.shape[0]
        chunk = torch.nn.functional.interpolate(chunk, [height, width], mode=mode, antialias=True)
        chunk = chunk.to(device)
    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed
    print(f" - Shape: {chunk.shape}")
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return fps

def test_decode_and_resize(src, height, width, mode="bicubic", frames_per_chunk=5):
    s = StreamReader(src)
    s.add_video_stream(
        frames_per_chunk, filter_desc=f"scale={width}:{height}:sws_flags={mode}", decoder_option={"threads": "8"}
    )

    num_frames = 0
    device = torch.device("cuda")
    chunk = None
    t0 = time.monotonic()
    for (chunk,) in s.stream():
        num_frames += chunk.shape[0]
        chunk = chunk.to(device)
    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed
    print(f" - Shape: {chunk.shape}")
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return fps

def test_hw_decode_and_resize(src, decoder, decoder_option, hw_accel="cuda", frames_per_chunk=5):
    s = StreamReader(src)
    s.add_video_stream(5, decoder=decoder, decoder_option=decoder_option, hw_accel=hw_accel)

    num_frames = 0
    chunk = None
    t0 = time.monotonic()
    for (chunk,) in s.stream():
        num_frames += chunk.shape[0]
    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed
    print(f" - Shape: {chunk.shape}")
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    return fps

def run_resize_tests(src):
    print(f"Testing: {os.path.basename(src)}")
    height, width = 224, 224
    print("* Software decoding with PyTorch interpolate")
    cpu_resize1 = test_decode_then_resize(src, height=height, width=width)
    print("* Software decoding with FFmpeg scale")
    cpu_resize2 = test_decode_and_resize(src, height=height, width=width)
    print("* Hardware decoding with resize")
    cuda_resize = test_hw_decode_and_resize(src, decoder="h264_cuvid", decoder_option={"resize": f"{width}x{height}"})
    return [cpu_resize1, cpu_resize2, cuda_resize]

fps_qvga = run_resize_tests(src_qvga)
fps_vga = run_resize_tests(src_vga)
fps_xga = run_resize_tests(src_xga)

def plot():
    fig, ax = plt.subplots(figsize=[9.6, 6.4])

    for items in zip(fps_qvga, fps_vga, fps_xga, "ov^sx"):
        ax.plot(items[:-1], marker=items[-1])
    ax.grid(axis="both")
    ax.set_xticks([0, 1, 2], ["QVGA (320x240)", "VGA (640x480)", "XGA (1024x768)"])
    ax.legend(
        [
            "Software decoding\nwith resize\n(PyTorch interpolate)",
            "Software decoding\nwith resize\n(FFmpeg scale)",
            "NVDEC\nwith resizing",
        ]
    )
    ax.set_title("Speed of processing video frames")
    ax.set_xlabel("Input video resolution")
    ax.set_ylabel("Frames per second")
    plt.tight_layout()


plot()




