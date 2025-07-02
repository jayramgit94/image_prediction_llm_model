import sys
import os
import json
import cv2
import torch
import numpy as np
import tempfile
import ffmpeg
import whisper
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.cnn_model import load_model, predict_image
from utils.image_utils import preprocess_pil_image

VIDEO_FOLDER = 'dataset/sample_videos'
OUTPUT_FILE = 'results_video.json'
FRAME_INTERVAL = 1  # seconds

def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    timestamps = []
    if not cap.isOpened() or fps == 0:
        return frames, timestamps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    sec = 0
    while sec < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        timestamps.append(sec)
        sec += interval
    cap.release()
    return frames, timestamps

def extract_audio(video_path, out_wav_path):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(out_wav_path, ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception:
        return False

def transcribe_audio(wav_path, model=None):
    if model is None:
        model = whisper.load_model("base", device="cpu")
    try:
        result = model.transcribe(wav_path, fp16=False)
        return result.get("text", "").strip()
    except Exception:
        return ""
    
def process_video(video_path, model, whisper_model):
    frames, timestamps = extract_frames(video_path, interval=FRAME_INTERVAL)
    frame_results = []
    for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess_pil_image(pil_img)
        label, debug_info = predict_image(model, img_tensor)
        # Only keep top-3 predictions for brevity
        top3 = debug_info.get("top5", [])[:3]
        thought_process = [
            f"Frame timestamp: {ts:.1f}s",
            "Top-3 predictions:",
            *[f"  {t['label']}: {t['prob']:.4f}" for t in top3],
            f"Final prediction: {label}"
        ]
        frame_results.append({
            "timestamp": ts,
            "label": label,
            "thought_process": thought_process
        })
    # Audio
    transcript = ""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        audio_ok = extract_audio(video_path, tmp_wav.name)
        tmp_wav.close()
        if audio_ok:
            transcript = transcribe_audio(tmp_wav.name, whisper_model)
    finally:
        with contextlib.suppress(Exception):
            os.unlink(tmp_wav.name)
    return frame_results, transcript

def main():
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Video folder '{VIDEO_FOLDER}' does not exist. Creating it now.")
        os.makedirs(VIDEO_FOLDER)
        print("Please add .mp4 or .avi files to this folder and rerun the script.")
        return
    model = load_model()
    whisper_model = whisper.load_model("base", device="cpu")
    results = []
    for filename in os.listdir(VIDEO_FOLDER):
        if filename.lower().endswith(('.mp4', '.avi')):
            video_path = os.path.join(VIDEO_FOLDER, filename)
            print(f"Processing {filename} ...")
            frame_results, transcript = process_video(video_path, model, whisper_model)
            results.append({
                "video_filename": filename,
                "frames": frame_results,
                "audio_transcript": transcript
            })
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")

    # Additional frame processing example
    cap = cv2.VideoCapture('your_video.mp4')
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Example: pretend we scan for an object
        detected = False  # Replace with your detection logic

        # Print concise info per frame
        print(f"Frame {frame_num}: {'Object detected' if detected else 'No object'}")

        # Optional: break after a few frames for demo
        if frame_num >= 10:
            break

    cap.release()

if __name__ == "__main__":
    main()