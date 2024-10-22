import os
import re
import time
import torch
import spaces
import requests
import tempfile
import concurrent
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download, hf_hub_url, login

from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from resemble_enhance.enhancer.inference import denoise, enhance


def download_file(url: str, destination: str, token: str = None):
    """
    Télécharge un fichier à partir d'une URL avec une barre de progression. Prend en charge les tokens API Hugging Face pour les modèles protégés.
    :param url: L'URL à partir de laquelle télécharger.
    :param destination: Le chemin de destination pour enregistrer le fichier téléchargé.
    :param token: Le jeton API Hugging Face (optionnel). Si non fourni, la variable d'environnement HF_API_TOKEN sera utilisée.
    """


    # utiliser le jeton passé ou récupérer depuis la variable d'environnement
    if token is None:
        token = os.getenv("HF_SPACE_TOKEN")

    # en-têtes pour la requête
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # requête GET en streaming avec en-têtes
    response = requests.get(url, stream=True, headers=headers)

    # taille totale en octets, définie à zéro si manquante
    total_size = int(response.headers.get('content-length', 0))

    # afficher la progression
    with open(destination, 'wb') as file, tqdm(desc=destination, total=total_size, unit='B', unit_scale=True,
                                               unit_divisor=1024) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def diviser_phrases_moore(texte: str) -> list:
    """
    Divise un texte en phrases en fonction des signes de ponctuation de fin de phrase.

    Cette fonction prend un texte en entrée et le divise en phrases en se basant sur les
    signes de ponctuation (tels que le point (.) ...). 
    Elle nettoie également les espaces superflus et filtre les chaînes vides.

    Args:
        texte (str): Le texte à diviser en phrases.

    Returns:
        list: Une liste de phrases nettoyées et divisées à partir du texte.
    """
    # définir les motifs de ponctuation de fin de phrase
    fin_de_phrase = re.compile(r'(?<=[.!?])\s+')

    # diviser le texte en phrases
    phrases = fin_de_phrase.split(texte)
    
    # nettoyer les espaces superflus et filtrer les chaînes vides
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    
    return phrases


# function to enhance speech
@spaces.GPU
def enhance_speech(audio_array, sampling_rate, solver, nfe, tau, denoise_before_enhancement):
    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoise_before_enhancement else 0.1

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def denoise_audio():
        try:
            return denoise(audio_array, sampling_rate, device)
        except Exception as e:
            print("> Error while denoising : ", str(e))
            return audio_array, sampling_rate

    def enhance_audio():
        try:
            return enhance(audio_array, sampling_rate, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        except Exception as e:
            print("> Error while enhancement : ", str(e))
            return audio_array, sampling_rate

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_denoise = executor.submit(denoise_audio)
        future_enhance = executor.submit(enhance_audio)

        denoised_audio, new_sr1 = future_denoise.result()
        enhanced_audio, new_sr2 = future_enhance.result()

        # convert to numpy and return
        return (new_sr1, denoised_audio.cpu().numpy()), (new_sr2, enhanced_audio.cpu().numpy())
