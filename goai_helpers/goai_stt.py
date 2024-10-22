import torch
import librosa
import time
from transformers import set_seed, Wav2Vec2ForCTC, AutoProcessor
import numpy as np
import spaces
import os


@spaces.GPU
def goai_stt(fichier):
    """
    transcrire un fichier audio donné.
    
    paramètres
    ---------- 
    fichier: str | tuple[int, np.ndarray]
        le chemin d'accès au fichier audio ou le tuple contenant le taux d'échantillonnage et les données audio.
        
    return
    ---------- 
    transcript: str
        le texte transcrit.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("fichier entré en entrée ---------> ", fichier)
    
    if fichier is None:
        raise ValueError("le fichier audio est manquant.")

    ### assurer reproductibilité
    set_seed(2024)
    
    start_time = time.time()
    
    ### charger le modèle de transcription 
    model_id = "ArissBandoss/wav2vec2-large-mms-1b-mos-V2"

    processor = AutoProcessor.from_pretrained(model_id, token=auth_token)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, token=auth_token, target_lang="mos", ignore_mismatched_sizes=True).to(device)

    if isinstance(fichier, str):
        ### preprocessing de l'audio à partir d'un fichier
        signal, sampling_rate = librosa.load(fichier, sr=16000)
    else:
        ### preprocessing de l'audio à partir d'un tableau numpy
        sampling_rate, signal = fichier

    # convert the signal to float32
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    
    if sampling_rate != 16000:
        signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
        
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
        
    if len(signal) < 160:
        raise ValueError("Le fichier audio est trop court pour être traité.")

    inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    ### faire l'inférence
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    pred_ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(pred_ids)

    print("temps écoulé: ", int(time.time() - start_time), " secondes")
    return transcription
