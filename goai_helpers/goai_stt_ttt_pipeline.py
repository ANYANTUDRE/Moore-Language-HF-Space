import os
import spaces
from huggingface_hub import login

from goai_helpers.goai_traduction import goai_traduction
from goai_helpers.goai_stt2 import transcribe

# authentification
auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)

MODEL_ASR = "ArissBandoss/whisper-small-mos"
LANGUAGE  = "Automatic Detection"


# gradio interface translation and text to speech function
@spaces.GPU(duration=120)
def goai_stt_ttt(
        inputs,  
        batch_size, 
        chunk_length_s, 
        stride_length_s,
        model=MODEL_ASR, 
        language=LANGUAGE,
    ):

    # 1. STT: Speech To Text
    mos_text = transcribe(
        inputs,  
        batch_size, 
        chunk_length_s, 
        stride_length_s,
        model=model, 
        language=language,
    )
    yield mos_text, None

    # 2. TTT: Translation mos ==> fr    
    fr_text = goai_traduction.goai_traduction(
        mos_text, 
        src_lang="fra_Latn", 
        tgt_lang="mos_Latn"
    )
    yield mos_text, fr_text