import os
import spaces
from huggingface_hub import login

from goai_helpers.goai_traduction import goai_traduction
from goai_helpers.goai_tts2 import MooreTTS, text_to_speech
from goai_helpers.goai_tts import goai_tts


# authentification
auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)


# gradio interface text to speech function
@spaces.GPU
def goai_many_tts(
        text,
        tts_model,
        reference_speaker,
        reference_audio=None,
    ):

    if tts_model == "ArissBandoss/coqui-tts-moore-V1":
        # TTS pipeline
        tts = MooreTTS(tts_model)
        reference_speaker = os.path.join("./exples_voix", reference_speaker)

        # convert translated text to speech with reference audio
        if reference_audio is not None:
            audio_array, sampling_rate = text_to_speech(tts, text, reference_speaker, reference_audio)
        else:
            audio_array, sampling_rate = text_to_speech(tts, text, reference_speaker=reference_speaker)

        return sampling_rate, audio_array.numpy()
    
    elif tts_model == "ArissBandoss/mms-tts-mos-male-17-V5":
        sample_rate, audio_data = goai_tts(text)
        return sample_rate, audio_data

    else:
        print("Erreur de modèle!!! Veuillez vérifier le modèle sélectionné.")


# gradio interface translation and text to speech function
@spaces.GPU(duration=120)
def goai_ttt_tts(
        text,
        tts_model,
        reference_speaker,
        reference_audio=None,
    ):

    # 1. TTT: Translation fr ==> mos    
    mos_text = goai_traduction(
        text, 
        src_lang="fra_Latn", 
        tgt_lang="mos_Latn"
    )
    yield mos_text, None
    
    # 2. TTS: Text to Speech
    sample_rate, audio_data = goai_many_tts(
        text,
        tts_model,
        reference_speaker,
        reference_audio=reference_audio,
    )
    yield mos_text, (sample_rate, audio_data)