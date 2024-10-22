import time
import torch
import spaces
import os
import numpy as np
from transformers import pipeline, set_seed
from huggingface_hub import login


auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)


@spaces.GPU
def goai_tts(texte):
    """
    Pour un texte donné, donner le speech en Mooré correspondant
    
    Paramètres
    ---------- 
    texte: str
        Le texte écrit.
        
    Return
    ------
        Un tuple contenant le taux d'échantillonnage et les données audio sous forme de tableau numpy.
    """
    
    # Assurer la reproductibilité
    set_seed(2024) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Charger le modèle TTS avec le token d'authentification
    model_id = "ArissBandoss/mms-tts-mos-V2"
    synthesiser = pipeline("text-to-speech", model_id, device=device)

    # Inférence
    speech = synthesiser(texte)
    
    sample_rate = speech["sampling_rate"]
    audio_data = np.array(speech["audio"][0], dtype=float)

    print("Temps écoulé: ", int(time.time() - start_time), " secondes")
    
    return sample_rate, audio_data
