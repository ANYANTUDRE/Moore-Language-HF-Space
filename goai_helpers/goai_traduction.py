import torch
import spaces
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
from huggingface_hub import login

max_length = 512
auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)


@spaces.GPU
def goai_traduction(text, src_lang, tgt_lang):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if src_lang == "fra_Latn" and tgt_lang == "mos_Latn":
        model_id = "ArissBandoss/nllb-200-distilled-600M-finetuned-fr-to-mos-V4"
        
    elif src_lang == "mos_Latn" and tgt_lang == "fra_Latn":
        model_id = "ArissBandoss/nllb-200-distilled-600M-finetuned-mos-to-fr-V5"

    else:
        model_id = "ArissBandoss/nllb-200-distilled-600M-finetuned-fr-to-mos-V4"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=auth_token)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=auth_token)
        
    trans_pipe = pipeline("translation", 
                          model=model, tokenizer=tokenizer, 
                          src_lang=src_lang, tgt_lang=tgt_lang, 
                          max_length=max_length,
                          device=device
                         )
    
    return trans_pipe(text)[0]["translation_text"]
