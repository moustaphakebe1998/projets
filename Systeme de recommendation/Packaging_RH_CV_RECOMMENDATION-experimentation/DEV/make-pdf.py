from pathlib import Path
from fpdf import FPDF

# Création des CVs en PDF
cv_data = {
    "candidat_1.pdf": """Développeur Data Science avec 6 ans d'expérience en Python et Machine Learning.
- Excellente maîtrise de pandas, NumPy et scikit-learn.
- Expérience en Deep Learning avec TensorFlow et PyTorch.
- Connaissance en NLP avec spaCy et transformers.
- Bonnes compétences en bases de données SQL et NoSQL.""",
    
    "candidat_2.pdf": """Développeur Backend avec 5 ans d'expérience en Python.
- Bonne connaissance de Django et Flask.
- Expérience en bases de données SQL (PostgreSQL, MySQL).
- Familiarité avec pandas mais peu d'expérience en Machine Learning.""",
    
    "candidat_3.pdf": """Ingénieur réseau avec 7 ans d'expérience en administration système.
- Spécialiste en configuration de serveurs et cybersécurité.
- Expérience en administration Linux et Windows.
- Aucun projet en Python ou Machine Learning.""",
"candidat_4.pdf": """Développeur Python junior avec 2 ans d'expérience.
- Langages de Programmation
- Python, R FactoMineR, Shiny), SQL,NoSQL SAS, C, Java
- Machine Learning , Deep Learning et GenAI
- Modèles avancés  LSTM, CNN, RNN, GAN, Transformers, VAE, Apprentissage (supervisé, non supervisé et renforcement).
- NLP & IA Générative  Traitement automatique de langage avec NLTK, spaCy, FastText, Camembert, RAG, LangChain. Ragas.
- Outils et Frameworks : Azure ML, MLflow, Docker, FastAPI, Gradio, Streamlit.
- ClOUD & MLOps :
- Microsoft Certified: Azure Data Scientist Associate,
- Base de données
- SQL PostgreSQL, Trino), NoSQL MongoDB, Elasticsearch)
- Pipelines de données
- ETL Apache NiFi, Apache Spark, Apache Beam), ELT DBT
"""
}

# Dossier de sortie
dossier_cvs = Path("cvs")
dossier_cvs.mkdir(exist_ok=True)

# Génération des fichiers PDF
for filename, content in cv_data.items():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    
    pdf_path = dossier_cvs / filename
    pdf.output(str(pdf_path))
    print(f"✅ Fichier créé : {pdf_path}")
