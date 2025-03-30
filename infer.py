from faster_whisper import WhisperModel

model_path = "models/faster-whisper-tiny-vn-1"

# Load model on CPU with FP16
model = WhisperModel(model_path, device="cpu", compute_type="int8")

# Transcrive a wav file
segments, info = model.transcribe("test.wav", beam_size=3, language='ar', task="translate")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# Print transcript with timestamps
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))