import torch
import spaces
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else "cpu"
# Whisper's full language ID mapping
LANG_TO_ID = {
    "en": 0, "zh": 1, "de": 2, "es": 3, "ru": 4, "ko": 5, "fr": 6, "ja": 7,
    "pt": 8, "tr": 9, "pl": 10, "ca": 11, "nl": 12, "ar": 13, "sv": 14, 
    "it": 15, "id": 16, "hi": 17, "fi": 18, "vi": 19, "he": 20, "uk": 21,
    "el": 22, "ms": 23, "cs": 24, "ro": 25, "da": 26, "hu": 27, "ta": 28,
    "no": 29, "th": 30, "ur": 31, "hr": 32, "bg": 33, "lt": 34, "la": 35,
    "mi": 36, "ml": 37, "cy": 38, "sk": 39, "te": 40, "fa": 41, "lv": 42,
    "bn": 43, "sr": 44, "az": 45, "sl": 46, "kn": 47, "et": 48, "mk": 49,
    "br": 50, "eu": 51, "is": 52, "hy": 53, "ne": 54, "mn": 55, "bs": 56,
    "kk": 57, "sq": 58, "sw": 59, "gl": 60, "mr": 61, "pa": 62, "si": 63,
    "km": 64, "sn": 65, "yo": 66, "so": 67, "af": 68, "oc": 69, "ka": 70,
    "be": 71, "tg": 72, "sd": 73, "gu": 74, "am": 75, "yi": 76, "lo": 77,
    "uz": 78, "fo": 79, "ht": 80, "ps": 81, "tk": 82, "nn": 83, "mt": 84,
    "sa": 85, "lb": 86, "my": 87, "bo": 88, "tl": 89, "mg": 90, "as": 91,
    "tt": 92, "haw": 93, "ln": 94, "ha": 95, "ba": 96, "jw": 97, "su": 98
}

@spaces.GPU
def transcribe(
        inputs, 
        model, 
        language, 
        batch_size, 
        chunk_length_s, 
        stride_length_s
    ):
    
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        device=DEVICE,
    )

    
    forced_decoder_ids = None
    if model.endswith(".en") == False and language in LANG_TO_ID:
        forced_decoder_ids = [[2, LANG_TO_ID[language]]]  # Setting forced decoder for language
    
    generate_kwargs = {}
    if forced_decoder_ids:
        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
    
    output = pipe(inputs, batch_size=batch_size, padding=True, truncation=True, **generate_kwargs)

    transcription_text = output['text']

    transcription_file_path = "transcription.txt"
    with open(transcription_file_path, "w") as f:
        f.write(transcription_text)
    
    return transcription_text, transcription_file_path