import os
import spaces
from huggingface_hub import login

from goai_helpers.goai_traduction import goai_traduction
from goai_helpers.goai_stt2 import transcribe

# authentification
auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)


# gradio interface translation and text to speech function
@spaces.GPU(duration=120)
def goai_stt_ttt(
        inputs, 
        model, 
        language, 
        batch_size, 
        chunk_length_s, 
        stride_length_s
    ):

    # 1. STT: Speech To Text
    mos_text = transcribe(
        inputs, 
        model, 
        language, 
        batch_size, 
        chunk_length_s, 
        stride_length_s
    )[0]
    yield mos_text, None

    # 2. TTT: Translation mos ==> fr    
    fr_text = goai_traduction(
        mos_text, 
        src_lang="mos_Latn", 
        tgt_lang="fra_Latn"
    )
    yield mos_text, fr_text