import torch
import spaces
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else "cpu"


@spaces.GPU
def transcribe(inputs, model, language, batch_size, chunk_length_s, stride_length_s):
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
    
    output = pipe(inputs, batch_size=batch_size, **generate_kwargs)

    transcription_text = output['text']

    transcription_file_path = "transcription.txt"
    with open(transcription_file_path, "w") as f:
        f.write(transcription_text)
    
    return transcription_text, transcription_file_path