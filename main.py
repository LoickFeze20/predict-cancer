import streamlit as st
st.set_page_config(page_title="Predict-Cancer", page_icon="🧬", layout="centered")

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np

@st.cache_resource
def load_cancer_model():
    model_path = "modeles.keras"

model = load_cancer_model()
classes = {
    0: "🧬 Adenocarcinome",
    1: "🫁 Bénin",
    2: "🔬 Carcinome épidermoïde"
}

# ------------------------ CSS pour design complet -------------------------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    :root {
        --clr-light-bg: #f5f7fa;
        --clr-dark-bg: #0f1118;
        --clr-light-card: rgba(255,255,255,0.75);
        --clr-dark-card: rgba(40,44,60,0.6);
        --clr-primary: #6A82FB;
        --clr-secondary: #FC5C7D;
    }
    @media (prefers-color-scheme: dark) {
        html {
            --bg: var(--clr-dark-bg);
            --card: var(--clr-dark-card);
            --text: #f0f0f0;
        }
    }
    @media (prefers-color-scheme: light) {
        html {
            --bg: var(--clr-light-bg);
            --card: var(--clr-light-card);
            --text: #111;
        }
    }
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }
    .hero {
        background: linear-gradient(135deg, #6A82FB, #FC5C7D);
        padding: 4rem 1rem 3rem;
        text-align: center;
        border-radius: 0 0 2rem 2rem;
        color: white;
    }
    .card {
        background: var(--card);
        padding: 2rem;
        border-radius: 1rem;
        backdrop-filter: blur(16px) saturate(180%);
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .sidebar-title {
        font-weight: 600;
        font-size: 1.3rem;
        padding: 1rem 0 0.5rem;
        border-bottom: 2px solid var(--clr-primary);
    }
    .prediction-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .image-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .image-row img {
        width: 180px;
        border-radius: 0.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- SIDEBAR PERSONNALISÉ ---------------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧬 Predict-Cancer</div>', unsafe_allow_html=True)
    menu = st.radio("Navigation", ["🏠 Accueil", "🔬 Prédire"])
    st.markdown("---")
    st.markdown("👨‍⚕️ *Projet Deep Learning - M1 IA*\n\n🧠 *Modèle CNN entraîné*", unsafe_allow_html=True)

# ----------------------------- PAGE ACCUEIL -------------------------------- #
if menu == "🏠 Accueil":
    st.markdown("""
    <div class="hero">
        <h1>Bienvenue sur <b>Predict-Cancer</b></h1>
        <p>Un outil intelligent pour la détection des <b>cancers pulmonaires</b> à partir d’images médicales par <b>Deep Learning</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    accueil_html = """
    <div class="card" style="margin-top:2rem;">
        <h3>🩺 À propos de l'application</h3>
        <p>
            <b>Predict-Cancer</b> est une application de diagnostic assisté par IA capable de classifier les types suivants :
        </p>
        <ul>
            <li><b>Adenocarcinome</b> : type courant de cancer du poumon chez les non-fumeurs.</li>
            <li><b>Carcinome épidermoïde</b> : souvent associé au tabagisme.</li>
            <li><b>Lésion bénigne</b> : non cancéreuse.</li>
        </ul>
        <p>
            Le modèle utilise un réseau de neurones convolutif (CNN) entraîné sur des images médicales pour fournir une prédiction précise.
        </p>
    </div>

    <div class="card" style="margin-top:2rem;">
        <h3>🚀 Comment ça fonctionne ?</h3>
        <ol>
            <li>Téléchargez une image radiologique dans l’onglet <b>🔬 Prédire</b>.</li>
            <li>L’IA analyse l’image et prédit le type de cancer.</li>
            <li>Vous recevez un résultat avec un pourcentage de confiance.</li>
        </ol>
        <p style="opacity:0.8">
            Cette application est destinée à des fins académiques. Toujours consulter un professionnel de santé pour tout diagnostic.
        </p>
    </div>
    <div class="card" style="margin-top:2rem;">
        <h3> Quel maladie l'application peut prédire ?</h3>
        <ol>
            <li>benign</li>
            <li>adenocarcinoma</li>
            <li>squamous_cell_carcinoma</li>
        </ol>
    </div>
    """

    st.markdown(accueil_html, unsafe_allow_html=True)



# ----------------------------- PAGE PREDICTION ----------------------------- #
else:
    st.markdown("## 🔬 Prédiction d’un cancer pulmonaire")
    uploaded = st.file_uploader("📥 Chargez une image médicale", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Image téléversée",width=80, use_container_width=True)
        if st.button("🧠 Lancer la prédiction"):
            img = keras_image.load_img(uploaded, target_size=(64, 64))
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)

            with st.spinner("Analyse en cours..."):
                preds = model.predict(arr)
                pred_index = int(np.argmax(preds[0]))
                confidence = float(np.max(preds[0])) * 100

            # Affichage résultat
            st.markdown(f"""
                <div class="card" style="margin-top:2rem; text-align:center;">
                    <div class="prediction-icon">{classes[pred_index].split()[0]}</div>
                    <h3 style="margin-bottom:.2rem;">{classes[pred_index]}</h3>
                    <p style="opacity:0.7">Confiance du modèle : <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

import streamlit as st
st.set_page_config(page_title="Predict-Cancer", page_icon="🧬", layout="centered")

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np

import os
st.write("Fichiers dans le dossier courant :", os.listdir("."))

@st.cache_resource
def load_cancer_model():
    return tf.keras.models.load_model("/modeles.keras")

model = load_cancer_model()
classes = {
    0: "🧬 Adenocarcinome",
    1: "🫁 Bénin",
    2: "🔬 Carcinome épidermoïde"
}

# ------------------------ CSS pour design complet -------------------------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');

    :root {
        --clr-light-bg: #f5f7fa;
        --clr-dark-bg: #0f1118;
        --clr-light-card: rgba(255,255,255,0.75);
        --clr-dark-card: rgba(40,44,60,0.6);
        --clr-primary: #6A82FB;
        --clr-secondary: #FC5C7D;
    }
    @media (prefers-color-scheme: dark) {
        html {
            --bg: var(--clr-dark-bg);
            --card: var(--clr-dark-card);
            --text: #f0f0f0;
        }
    }
    @media (prefers-color-scheme: light) {
        html {
            --bg: var(--clr-light-bg);
            --card: var(--clr-light-card);
            --text: #111;
        }
    }
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }
    .hero {
        background: linear-gradient(135deg, #6A82FB, #FC5C7D);
        padding: 4rem 1rem 3rem;
        text-align: center;
        border-radius: 0 0 2rem 2rem;
        color: white;
    }
    .card {
        background: var(--card);
        padding: 2rem;
        border-radius: 1rem;
        backdrop-filter: blur(16px) saturate(180%);
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .sidebar-title {
        font-weight: 600;
        font-size: 1.3rem;
        padding: 1rem 0 0.5rem;
        border-bottom: 2px solid var(--clr-primary);
    }
    .prediction-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .image-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .image-row img {
        width: 180px;
        border-radius: 0.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- SIDEBAR PERSONNALISÉ ---------------------------- #
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧬 Predict-Cancer</div>', unsafe_allow_html=True)
    menu = st.radio("Navigation", ["🏠 Accueil", "🔬 Prédire"])
    st.markdown("---")
    st.markdown("👨‍⚕️ *Projet Deep Learning - M1 IA*\n\n🧠 *Modèle CNN entraîné*", unsafe_allow_html=True)

# ----------------------------- PAGE ACCUEIL -------------------------------- #
if menu == "🏠 Accueil":
    st.markdown("""
    <div class="hero">
        <h1>Bienvenue sur <b>Predict-Cancer</b></h1>
        <p>Un outil intelligent pour la détection des <b>cancers pulmonaires</b> à partir d’images médicales par <b>Deep Learning</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    accueil_html = """
    <div class="card" style="margin-top:2rem;">
        <h3>🩺 À propos de l'application</h3>
        <p>
            <b>Predict-Cancer</b> est une application de diagnostic assisté par IA capable de classifier les types suivants :
        </p>
        <ul>
            <li><b>Adenocarcinome</b> : type courant de cancer du poumon chez les non-fumeurs.</li>
            <li><b>Carcinome épidermoïde</b> : souvent associé au tabagisme.</li>
            <li><b>Lésion bénigne</b> : non cancéreuse.</li>
        </ul>
        <p>
            Le modèle utilise un réseau de neurones convolutif (CNN) entraîné sur des images médicales pour fournir une prédiction précise.
        </p>
    </div>

    <div class="card" style="margin-top:2rem;">
        <h3>🚀 Comment ça fonctionne ?</h3>
        <ol>
            <li>Téléchargez une image radiologique dans l’onglet <b>🔬 Prédire</b>.</li>
            <li>L’IA analyse l’image et prédit le type de cancer.</li>
            <li>Vous recevez un résultat avec un pourcentage de confiance.</li>
        </ol>
        <p style="opacity:0.8">
            Cette application est destinée à des fins académiques. Toujours consulter un professionnel de santé pour tout diagnostic.
        </p>
    </div>
    """

    st.markdown(accueil_html, unsafe_allow_html=True)



# ----------------------------- PAGE PREDICTION ----------------------------- #
else:
    st.markdown("## 🔬 Prédiction d’un cancer pulmonaire")
    uploaded = st.file_uploader("Chargez une image médicale", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Image téléversée",width="500", use_container_width=True)
        if st.button("🧠 Lancer la prédiction"):
            img = keras_image.load_img(uploaded, target_size=(64, 64))
            arr = img_to_array(img)
            arr = np.expand_dims(arr, axis=0)

            with st.spinner("Analyse en cours..."):
                preds = model.predict(arr)
                pred_index = int(np.argmax(preds[0]))
                confidence = float(np.max(preds[0])) * 100

            # Affichage résultat
            st.markdown(f"""
                <div class="card" style="margin-top:2rem; text-align:center;">
                    <div class="prediction-icon">{classes[pred_index].split()[0]}</div>
                    <h3 style="margin-bottom:.2rem;">{classes[pred_index]}</h3>
                    <p style="opacity:0.7">Confiance du modèle : <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
