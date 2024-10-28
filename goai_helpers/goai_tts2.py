import os
import time
import torch
import torchaudio
import spaces
import tempfile
from tqdm import tqdm
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download, hf_hub_url, login

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from goai_helpers.utils import download_file, diviser_phrases_moore, enhance_speech
from goai_helpers.goai_traduction import goai_traduction

# authentification
auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MooreTTS:
    """
    Classe Mooré Text-to-Speech (TTS) qui initialise et utilise un modèle TTS.
    Attributs :
        language_code (str) : code ISO de la langue pour le mooré.
        checkpoint_repo_or_dir (str) : URL ou chemin local vers le répertoire du point de contrôle du modèle.
        local_dir (str) : Le répertoire pour stocker les points de contrôle téléchargés.
        paths (dict) : Un dictionnaire des chemins vers les composants du modèle.
        config (XttsConfig) : Objet de configuration pour le modèle TTS.
        model (Xtts) : L'instance du modèle TTS.
    """

    def __init__(self, checkpoint_repo_or_dir: str, local_dir: Optional[str] = None):
        """
        Initialise l'instance MooreTTS.
        Args :
            checkpoint_repo_or_dir : Une chaîne représentant soit un dépôt Hugging Face,
                                     soit un répertoire local où le point de contrôle du modèle TTS est situé.
            local_dir : Une chaîne optionnelle représentant un chemin de répertoire local où les points de contrôle du modèle
                        seront téléchargés. Si non spécifié, un répertoire local par défaut est utilisé
                        basé sur `checkpoint_repo_or_dir`.
        Le processus d'initialisation implique la configuration de répertoires locaux pour les composants du modèle,
        l'assurance que le point de contrôle du modèle est disponible, et le chargement de la configuration et du tokenizer du modèle.
        """

        # code de langue
        self.language_code = 'mos'

        # emplacement du point de contrôle et le chemin du répertoire local
        self.checkpoint_repo_or_dir = checkpoint_repo_or_dir
        
        # si aucun répertoire local n'est fourni, utiliser le répertoire par défaut basé sur le point de contrôle
        self.local_dir = local_dir if local_dir else self.default_local_dir(checkpoint_repo_or_dir)

        # initialiser les chemins pour les composants du modèle
        self.paths = self.init_paths(self.local_dir)

        # s'assurer que le point de contrôle du modèle est disponible localement
        self.ensure_checkpoint_is_downloaded()

        # charger la configuration du modèle à partir d'un fichier JSON
        self.config = XttsConfig()
        self.config.load_json(self.paths['config.json'])

        # initialiser le modèle TTS avec la configuration chargée
        self.model = Xtts.init_from_config(self.config)

        #print(f"\n\n============ DEBUGGING   =========== {self.local_dir}\n\n")
        # charger le point de contrôle du modèle dans le modèle initialisé
        self.model.load_checkpoint(
            self.config,
            checkpoint_path=self.local_dir+ "/model_compressed.pth" ,
            vocab_path=self.paths['vocab.json'],
            use_deepspeed=False 
        )

        if torch.cuda.is_available():
            self.model.cuda()
            
        print("Model loaded successfully!")

    
    def ensure_checkpoint_is_downloaded(self):
        """
        S'assure que le point de contrôle du modèle est téléchargé et disponible localement.
        """
        if os.path.exists(self.checkpoint_repo_or_dir):
            return

        os.makedirs(self.local_dir, exist_ok=True)
        print("Téléchargement du point de contrôle depuis le hub...")

        for filename, filepath in self.paths.items():
            if os.path.exists(filepath):
                print(f"Fichier {filepath} déjà existant. Passé...")
                continue

            file_url = hf_hub_url(repo_id=self.checkpoint_repo_or_dir, filename=filename)
            print(f"Téléchargement de {filename} depuis {file_url}")
            try:
                download_file(file_url, filepath)
            except Exception as e:
                print(f"Téléchargement de {filename} échoué: {e}")


        print("Point de contrôle téléchargé avec succès !")

        
    def default_local_dir(self, checkpoint_repo_or_dir: str) -> str:
        """
        Génère un chemin de répertoire local par défaut pour stocker le point de contrôle du modèle.
        Args :
            checkpoint_repo_or_dir : Le dépôt ou chemin de répertoire original du point de contrôle.
        Returns :
            Le chemin de répertoire local par défaut.
        """
        if os.path.exists(checkpoint_repo_or_dir):
            return checkpoint_repo_or_dir

        model_path = f"models--{checkpoint_repo_or_dir.replace('/', '--')}"
        local_dir = os.path.join(os.path.expanduser('~'), "mooreTTS", model_path)
        return local_dir.lower()

    @staticmethod
    def init_paths(local_dir: str) -> dict:
        """
        Initialise les chemins vers les divers composants du modèle basés sur le répertoire local.
        Args :
            local_dir : Le répertoire local où les composants du modèle sont stockés.
        Returns :
            Un dictionnaire avec des clés comme noms des composants et des valeurs comme chemins des fichiers.
        """
        components = ['model_compressed.pth', 'config.json', 'vocab.json', 'dvae.pth', 'mel_stats.pth']
        return {name: os.path.join(local_dir, name) for name in components}

    def text_to_speech(
            self,
            tts_text: str,
            speaker_reference_wav_path: Optional[str] = None,
            temperature: Optional[float] = 0.1
    ) -> Tuple[int, torch.Tensor]:
        """
        Convertit un texte en audio de synthèse vocale.
        Args :
            text : Le texte d'entrée à convertir en audio.
            speaker_reference_wav_path : Un chemin vers un fichier WAV de référence pour l'orateur.
            temperature : Le paramètre de température pour l'échantillonnage.
            enable_text_splitting : Indicateur pour activer ou désactiver la découpe du texte.
        Returns :
            Un tuple contenant le taux d'échantillonnage et le tenseur audio généré.
        """
        if speaker_reference_wav_path is None:
            speaker_reference_wav_path = "./audios/ref1_male_17.wav"
            print("Utilisation du fichier de référence par défaut ./audios/ref1_male_17.wav")

        print("Calcul des latents de conditionnement de l'orateur...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[speaker_reference_wav_path],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

        tts_texts = diviser_phrases_moore(tts_text)
        
        print("Début de l'inférence...")
        start_time = time.time()

        wav_chunks = []
        for text in tqdm(tts_texts):
            wav_chunk = self.model.inference(
                text=text,
                language=self.language_code,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.1,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=10,
                top_p=0.3,
            )
            wav_chunks.append(torch.tensor(wav_chunk["wav"]))
        
        end_time = time.time()

        audio = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
        sampling_rate = torch.tensor(self.config.model_args.output_sample_rate).cpu().item()

        print(f"Voix générée en {end_time - start_time:.2f} secondes.")

        return sampling_rate, audio


# function to convert text to speech
@spaces.GPU
def text_to_speech(tts, text, reference_speaker: str, reference_audio: Optional[Tuple] = None):
    if reference_audio is not None:
        ref_sr, ref_audio = reference_audio
        ref_audio = torch.from_numpy(ref_audio)

        # Add a channel dimension if the audio is 1D
        if ref_audio.ndim == 1:
            ref_audio = ref_audio.unsqueeze(0)

        # Save the reference audio to a temporary file if it's not None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            torchaudio.save(tmp.name, ref_audio, ref_sr)
            tmp_path = tmp.name

        # Use the temporary file as the speaker reference
        sr, audio = tts.text_to_speech(text, speaker_reference_wav_path=tmp_path)

        # Clean up the temporary file
        os.unlink(tmp_path)
    else:
        # If no reference audio provided, proceed with the reference_speaker
        sr, audio = tts.text_to_speech(text, speaker_reference_wav_path=reference_speaker)

    audio = audio.mean(dim=0)
    return audio, sr


# gradio interface text to speech function
@spaces.GPU
def goai_tts2(
        text,
        reference_speaker,
        reference_audio=None,
        solver="Midpoint",
        nfe=128,
        prior_temp=0.01,
        denoise_before_enhancement=False
):
    # TTS pipeline
    tts_model = "ArissBandoss/coqui-tts-moore-V1"
    tts = MooreTTS(tts_model)
    
    reference_speaker = os.path.join("./exples_voix", reference_speaker)


    # convert translated text to speech with reference audio
    if reference_audio is not None:
        audio_array, sampling_rate = text_to_speech(tts, text, reference_speaker, reference_audio)
    else:
        audio_array, sampling_rate = text_to_speech(tts, text, reference_speaker=reference_speaker)

    yield text, (sampling_rate, audio_array.numpy()), None, None

    # enhance audio
    denoised_audio, enhanced_audio = enhance_speech(
        audio_array,
        sampling_rate,
        solver,
        nfe,
        prior_temp,
        denoise_before_enhancement
    )

    yield (sampling_rate, audio_array.numpy()), denoised_audio, enhanced_audio



# gradio interface translation and text to speech function
@spaces.GPU(duration=120)
def goai_ttt_tts(
        text,
        reference_speaker,
        reference_audio=None,
        solver="Midpoint",
        nfe=128,
        prior_temp=0.01,
        denoise_before_enhancement=False
):

    # translation    
    mos_text = goai_traduction(
                        text, 
                        src_lang="fra_Latn", 
                        tgt_lang="mos_Latn"
                    )
    yield mos_text, None, None, None
    
    # TTS pipeline
    reference_speaker = os.path.join("./exples_voix", reference_speaker)
    tts_model = "ArissBandoss/coqui-tts-moore-V1"
    tts = MooreTTS(tts_model)
    
    # convert translated text to speech with reference audio
    if reference_audio is not None:
        audio_array, sampling_rate = text_to_speech(tts, mos_text, reference_speaker, reference_audio)
    else:
        audio_array, sampling_rate = text_to_speech(tts, mos_text, reference_speaker=reference_speaker)

    yield mos_text, (sampling_rate, audio_array.numpy()), None, None

    # enhance audio
    denoised_audio, enhanced_audio = enhance_speech(
        audio_array,
        sampling_rate,
        solver,
        nfe,
        prior_temp,
        denoise_before_enhancement
    )

    yield mos_text, (sampling_rate, audio_array.numpy()), denoised_audio, enhanced_audio