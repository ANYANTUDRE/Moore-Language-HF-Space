import os
import torch
import torchaudio
import scipy

import gradio as gr
from transformers import set_seed, pipeline
from datasets import load_dataset, Audio

import goai_traduction, goai_stt, goai_stt2, goai_tts,  goai_tts2

#language_list = ['mos', 'fra', 'eng']

# list all files in the ./audios directory for the dropdown
audio_files = [f for f in os.listdir('./exples_voix') if os.path.isfile(os.path.join('./exples_voix', f))]

# device
device = 0 if torch.cuda.is_available() else "cpu"

# texte décrivant chaque tab
description = """<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
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

# gradio app
demo = gr.Blocks()

goai_stt = gr.Interface(
    fn = goai_stt.goai_stt,
    inputs=[
        #gr.Audio(sources=["microphone", "upload"], type="numpy")
        gr.Audio(sources=["upload"], type="numpy")
    ],
    outputs="text",
    examples=[["./audios/example1.mp3", "a ye ligdi"], 
              ["./audios/example2.mp3", "zoe nimbãanega"],
              ["./audios/example3.mp3", "zãng-zãnga"],
              ["./audios/example4.mp3", "yõk foto"]
             ],
     cache_examples=False,
    title="Traduction du Mooré: texte vers texte",
    description=description
)


goai_stt2 = gr.Interface(
    fn = goai_stt2.goai_stt2,
    inputs=[
        #gr.Audio(sources=["microphone", "upload"], type="numpy")
        gr.Audio(sources=["upload"], type="numpy")
    ],
    outputs="text",
    examples=[["./audios/example1.mp3", "a ye ligdi"], 
              ["./audios/example2.mp3", "zoe nimbãanega"],
              ["./audios/example3.mp3", "zãng-zãnga"],
              ["./audios/example4.mp3", "yõk foto"]
             ],
     cache_examples=False,
    title="Traduction du Mooré: texte vers texte",
    description=description
)


goai_tts = gr.Interface(
    fn=goai_tts.goai_tts,
    inputs=[
        gr.Text(label="Texte", placeholder="a ye ligdi")
    ],
    outputs=[
        gr.Audio(label="Audio généré", type="numpy")
    ],
    examples=[["a ye ligdi"], 
              ["zoe nimbãanega "],
              ["zãng-zãnga"],
              ["yõk foto"]
             ],
    cache_examples=False,
    title="Traduction du Mooré: texte vers texte",
    description=description
)

goai_tts2 = gr.Interface(
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
    description=description,
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
    description=description
)


with demo:
    gr.TabbedInterface(
        #[goai_traduction, goai_tts, goai_stt, goai_tts2],
        #["Traduction", "Text-2-speech", "Speech-2-text", "Text-2-speech-2"],
        [goai_tts2, goai_stt, goai_stt2],
        ["Traduction et Synthèse vocale du Mooré", "Speech-2-text", "Speech-2-text-Whisper"],
    )

demo.launch()