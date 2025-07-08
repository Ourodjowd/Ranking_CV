import streamlit as st

from langchain.embeddings import AzureOpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import pandas as pd
import PyPDF2 
from PyPDF2 import PdfReader
import io
import numpy as np

# Configuration de la page Streamlit
st.set_page_config(page_title="Ranking CVs", layout="wide")
st.title("Ranking CVs based on the Job Description")

# Chargement du modèle d'embedding

load_dotenv()

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint = os.getenv("EMBEDDING_API_BASE"),
    openai_api_key = os.getenv("EMBEDDING_API_KEY"),
    deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    chunk_size=10,
)

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(file, password=''):
    pdf_reader = PdfReader(file)
    
    if pdf_reader.is_encrypted:
        try:
            pdf_reader.decrypt(password)
        except Exception as e:
            raise ValueError("Failed to decrypt PDF. A password may be required or incorrect.") from e

    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''

    return text

# Interface utilisateur
st.header("1. Enter the job description")
job_description = st.text_area("Job Description", height=200)

st.header("2. Upload CVs (PDF)")
uploaded_files = st.file_uploader("Select up to 10 CVs", type=['pdf'], accept_multiple_files=True)

if len(uploaded_files) > 10:
    st.error("Please upload a maximum of 10 CVs")
    uploaded_files = uploaded_files[:10]

if job_description and uploaded_files:
    if st.button("Analyze CVs"):
        # Extraction du texte des CV
        resumes = []
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            resumes.append(resume_text)

        # Création des embeddings
        job_vector = embedding_model.embed_query(job_description)
        resume_vectors = embedding_model.embed_documents(resumes)

        job_vector = np.array(job_vector).reshape(1, -1)


        # Calcul des similarités
        similarities = cosine_similarity(job_vector, resume_vectors)[0]

        # Création du DataFrame
        df = pd.DataFrame({
            "CV": [file.name for file in uploaded_files],
            "Score": similarities
        })

        # Tri par score
        df = df.sort_values(by="Score", ascending=False)

        # Affichage des résultats
        st.header("3. Ranking Results")
        st.dataframe(df)

        # Visualisation graphique
        st.bar_chart(df.set_index("CV")["Score"])


print("Application is running. Please open the Streamlit app to view the results.")