import gradio as gr
from huggingface_hub import login

import os
#from languages import get_language_names
from goai_helpers import goai_traduction, goai_stt, goai_stt2, goai_tts,  goai_tts2, goai_ttt_tts_pipeline, goai_stt_ttt_pipeline

auth_token = os.getenv('HF_SPACE_TOKEN')
login(token=auth_token)


# list all files in the ./audios directory for the dropdown
AUDIO_FILES = [f for f in os.listdir('./exples_voix') if os.path.isfile(os.path.join('./exples_voix', f))]
MODELES_TTS = ["ArissBandoss/coqui-tts-moore-V1", "ArissBandoss/mms-tts-mos-V18"]
MODELES_ASR = ["ArissBandoss/whisper-small-mos", "openai/whisper-large-v3-turbo"]
LANGUAGES   = ["Automatic Detection"]

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


demo = gr.Blocks(theme=gr.themes.Soft())

goai_traduction_if = gr.Interface(
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
    title="Traduction Mooré-Francais",
    description=DESCRIPTION
)

goai_stt_if = gr.Interface(
    fn=goai_stt2.transcribe,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath"),
        gr.Dropdown(
            choices=MODELES_ASR,
            value="ArissBandoss/whisper-small-mos", 
            label="Model Name"
        ),
        gr.Dropdown(
            choices=LANGUAGES, 
            value="Automatic Detection",  # + sorted(get_language_names())
            label="Language", 
            interactive = True,
        ), 
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
    title="Mooré ASR",
    description=DESCRIPTION,
    flagging_mode="auto",
)

goai_ttt_tts_pipeline_if = gr.Interface(
    fn=goai_ttt_tts_pipeline.goai_ttt_tts,
    inputs=[
        gr.Text(
            label="Texte à traduire", 
            lines=3, 
            value="Par cette ouverture, le centre se veut contribuer à la formation professionnelle des jeunes et des femmes, renforcer les capacités des acteurs du monde agricole, et contribuer à la lutte contre le chômage au Burkina Faso."
        ),
        gr.Dropdown(
            label="Modèles de TTS", 
            choices=MODELES_TTS, 
            value="ArissBandoss/coqui-tts-moore-V1"
        ),
        gr.Dropdown(
            label="Voix", 
            choices=AUDIO_FILES, 
            value="exple_voix_masculine.wav"
        ),
        gr.Audio(
            label="Cloner votre voix (optionel)", 
            type="numpy", 
            format="wav"
        ),
    ],
    outputs=[
        gr.Text(label="Texte traduit"),
        gr.Audio(label="Audio généré", format="wav"),
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


goai_stt_ttt_pipeline_if = gr.Interface(
    fn=goai_stt_ttt_pipeline.goai_stt_ttt,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath"),
        gr.Dropdown(
            label="Modèles d'ASR", 
            choices=MODELES_ASR, 
            value="ArissBandoss/whisper-small-mos",
        ),
        gr.Dropdown(
            choices=LANGUAGES, 
            value="Automatic Detection",  # + sorted(get_language_names())
            label="Language", 
            interactive = True,
        ), 
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
    title="Mooré ASR",
    description=DESCRIPTION,
    flagging_mode="auto",
)

with demo:
    gr.TabbedInterface(
        interface_list=[goai_traduction_if, goai_stt_if, goai_ttt_tts_pipeline_if, goai_stt_ttt_pipeline_if],
        tab_names=["Traduction Mooré-Francais", "Mooré ASR", "Mooré TTS & Traduction", "Mooré ASR & Traduction"]
    )

demo.queue().launch(ssr_mode=False)