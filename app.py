import spaces
import torch
import scipy
import torchaudio

import gradio as gr
from transformers import pipeline, set_seed
from huggingface_hub import login

import os
from languages import get_language_names
from goai_helpers import goai_traduction, goai_stt, goai_stt2, goai_tts,  goai_tts2


auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)


# list all files in the ./audios directory for the dropdown
AUDIO_FILES = [f for f in os.listdir('./exples_voix') if os.path.isfile(os.path.join('./exples_voix', f))]

DESCRIPTION = """<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 250px;">
                        Ce modèle de traduction vers la <b>langue Mooré</b> a été développé from scratch par <b>GO AI CORP</b> et la version disponible en test est celle à 700 millions de paramètres.
                        <br><br>
                        Pour les détails techniques sur l'architecture du modèle, prendre attache avec nous via WhatsApp au <b>+226 66 62 83 03</b>.
                    </div>
                    <div style="flex-shrink: 0; min-width: 150px; text-align: center;">
                        <img src="https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/blob/main/img/goaicorp-logo2.jpg?raw=true" width="300px" style="max-width: 100%; height: auto;">
                    </div>
                </div>
                """
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



demo = gr.Blocks(theme=gr.themes.Ocean())

goai_stt = gr.Interface(
    fn=goai_stt2.transcribe,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath"),
        gr.Dropdown(
            choices=[
                "ArissBandoss/whisper-small-mos",
                "openai/whisper-large-v3-turbo", 
            ],
            value="ArissBandoss/whisper-small-mos", 
            label="Model Name"
        ),
        gr.Dropdown(choices=["Automatic Detection"] + sorted(get_language_names()), value="Automatic Detection", label="Language", interactive = True,),
        gr.Slider(label="Batch Size", minimum=1, maximum=32, value=8, step=1),
        gr.Slider(label="Chunk Length (s)", minimum=1, maximum=60, value=17.5, step=0.1),
        gr.Slider(label="Stride Length (s)", minimum=1, maximum=30, value=1, step=0.1),
    ],
    outputs=[gr.Textbox(label="Output"), gr.File(label="Download Files")],
    examples=[["./audios/example1.mp3", "a ye ligdi"], 
              ["./audios/example2.mp3", "zoe nimbãanega"],
              ["./audios/example3.mp3", "zãng-zãnga"],
              ["./audios/example4.mp3", "yõk foto"]
             ],
    cache_examples=False,
    title="Mooré ASR: Transcribe Audio",
    description=DESCRIPTION,
    flagging_mode="auto",
)

goai_tts = gr.Interface(
    fn=goai_tts2.goai_ttt_tts,
    inputs=[
        gr.Text(label="Texte à traduire", lines=2, value="Par cette ouverture, le centre se veut contribuer à la formation professionnelle des jeunes et des femmes, renforcer les capacités des acteurs du monde agricole, et contribuer à la lutte contre le chômage au Burkina Faso."),
        gr.Dropdown(label="Voix", choices=audio_files, value="exple_voix_masculine.wav"),
        gr.Audio(label="Cloner votre voix (optionel)", type="numpy", format="wav"),
    ],
    outputs=[
        gr.Text(label="Texte traduit"),
        gr.Audio(label="Audio original généré", format="wav"),
        gr.Audio(label="Denoised Audio", format='wav'),
        gr.Audio(label="Enhanced Audio", format='wav')
    ],
    examples=[["Ils vont bien, merci. Mon père travaille dur dans les champs et ma mère est toujours occupée à la maison.", "exple_voix_masculine.wav", None], 
              ["La finale s’est jouée en présence du Président du Faso, Ibrahim Traoré.", "exple_voix_feminine.wav", None],
              ["Les enfants apprennent les danses traditionnelles de leurs ancêtres, jouent à des jeux traditionnels dans les rues et aident leurs parents dans les tâches quotidiennes.", "exple_voix_masculine.wav", None],
              ["Ils achetèrent des troupeaux, firent construire des cases, parcoururent tout le pays pour offrir à leur mère et à leurs femmes les plus beaux bijoux, les plus belles étoffes.", "exple_voix_feminine.wav", None]
             ],
    cache_examples=False,
    title="Démo des Modèles pour le Mooré: Traduction (Text-to-Text) et Synthèse Vocale (Text-to-Speech)",
    description=DESCRIPTION,
)

goai_traduction = gr.Interface(
    fn=goai_traduction.goai_traduction,
    inputs=[
        gr.Textbox(label="Texte", placeholder="Yaa sõama"),
        gr.Dropdown(label="Langue source", choices=["fra_Latn", "mos_Latn"], value='fra_Latn'),
        gr.Dropdown(label="Langue cible", choices=["fra_Latn", "mos_Latn"], value='mos_Latn')
    ],
    outputs=["text"],
    examples=[["Yʋʋm a wãn la b kẽesd biig lekolle?", "mos_Latn", "fra_Latn"],
              ["Zak-soab la kasma.", "mos_Latn", "fra_Latn"],
              ["Le gouvernement avait pris des mesures louables par rapport à l’augmentation des prix de certaines denrées alimentaires.", "fra_Latn", "mos_Latn"],
              ["Comme lors du match face à la Côte d’Ivoire, c’est sur un coup de pied arrêté que les Etalons encaissent leur but.", "fra_Latn", "mos_Latn"],
    ],
    cache_examples=False,
    title="Traduction du Mooré: texte vers texte",
    description=DESCRIPTION
)


with demo:
    gr.TabbedInterface(
        interface_list=[goai_stt, goai_tts, goai_traduction],
        tab_names=["Microphone & Audio file"]
    )

demo.queue().launch(ssr_mode=False)
