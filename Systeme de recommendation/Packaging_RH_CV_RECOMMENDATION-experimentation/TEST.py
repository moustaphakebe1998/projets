from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

class CVAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            stop_words=['le', 'la', 'les', 'de', 'des', 'du', 'et', 'en', 'un', 'une']
        )
        
    def preprocess_text(self, text):
        """Nettoie et prépare le texte pour l'analyse."""
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux
        text = re.sub(r'[^\w\s]', ' ', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def calculate_match_score(self, cv_text, job_description):
        """Calcule le score de correspondance entre un CV et une description de poste."""
        # Prétraitement des textes
        cv_text = self.preprocess_text(cv_text)
        job_description = self.preprocess_text(job_description)
        
        # Vectorisation des textes
        tfidf_matrix = self.vectorizer.fit_transform([job_description, cv_text])
        
        # Calcul de la similarité cosinus
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity * 100  # Convertir en pourcentage
    
    def analyze_cv(self, cv_text, job_description, threshold=60):
        """Analyse un CV et décide s'il correspond au poste."""
        match_score = self.calculate_match_score(cv_text, job_description)
        
        analysis = {
            'score': round(match_score, 2),
            'recommendation': 'Accepté' if match_score >= threshold else 'Rejeté',
            'raison': f"Score de correspondance de {round(match_score, 2)}% " +
                     f"({'supérieur' if match_score >= threshold else 'inférieur'} au seuil de {threshold}%)"
        }
        
        return analysis

    def analyze_multiple_cvs(self, cvs_dict, job_description, threshold=60):
        """Analyse plusieurs CV et les classe par ordre de pertinence."""
        results = []
        
        for name, cv_text in cvs_dict.items():
            analysis = self.analyze_cv(cv_text, job_description, threshold)
            results.append({
                'nom_candidat': name,
                'score': analysis['score'],
                'recommendation': analysis['recommendation'],
                'raison': analysis['raison']
            })
        
        # Trier les résultats par score décroissant
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        return results_df

# Exemple d'utilisation
if __name__ == "__main__":
    analyzer = CVAnalyzer()
    
    # Description du poste exemple
    job_description = """
    Nous recherchons un développeur Python expérimenté avec une expertise en
    analyse de données et machine learning. La personne devra avoir une bonne
    maîtrise de scikit-learn et pandas. Une expérience en NLP est un plus.
    """
    
    # CV exemple
    cv_text = """
    Développeur Python senior avec 5 ans d'expérience en analyse de données.
    maîtrise de scikit-learn et pandas. Une expérience en NLP , tensorflow pytorch
    Nous recherchons un développeur Python expérimenté avec une expertise en
    analyse de données et machine learning. La personne devra avoir une bonne
    maîtrise de scikit-learn et pandas. Une expérience en NLP est un plus.
    """
    
    # Analyser un seul CV
    result = analyzer.analyze_cv(cv_text, job_description)
    print(f"Résultat de l'analyse :\n{result}")
    
    # Analyser plusieurs CV
    cvs = {
        "Candidat 1": cv_text,
        "Candidat 2": "Autre CV avec moins d'expérience en Python...",
    }
    
    results_df = analyzer.analyze_multiple_cvs(cvs, job_description)
    print("\nRésultats pour plusieurs candidats :\n", results_df)





    #! /usr/bin/env python3
from pathlib import Path
import shutil
import fitz  # PyMuPDF
from cv_rh_kebe.analyse import CVAnalyzer

# Définition des chemins de répertoires
DOSSIER_CVS = Path("cvs")
DOSSIER_ACCEPTES = Path("cvs_acceptes")
DOSSIER_REFUSES = Path("cvs_refuses")
FICHIER_OFFRE = Path("job_description.txt")

# Fonctions utilitaires
def lire_fichier(chemin: Path) -> str:
    """Lit le contenu d'un fichier texte."""
    return chemin.read_text(encoding="utf-8").strip()

def extraire_texte_pdf(chemin_pdf: Path) -> str:
    """Extrait le texte d'un fichier PDF."""
    texte = ""
    try:
        with fitz.open(chemin_pdf) as doc:
            for page in doc:
                texte += page.get_text("text") + "\n"
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de {chemin_pdf.name} : {e}")
    return texte.strip()

def deplacer_fichier(chemin_fichier: Path, dossier_destination: Path):
    """Déplace un fichier vers un autre dossier en le créant si nécessaire."""
    dossier_destination.mkdir(parents=True, exist_ok=True)  # Créer le dossier si nécessaire
    shutil.move(str(chemin_fichier), str(dossier_destination / chemin_fichier.name))

def analyser_cvs(dossier_cvs: Path, job_description: str):
    """Analyse tous les CVs en PDF dans un dossier et les classe en Accepté ou Refusé."""
    analyzer = CVAnalyzer()
    cvs = {}

    # Lire et extraire le texte de chaque PDF
    for fichier in dossier_cvs.glob("*.pdf"):
        texte_cv = extraire_texte_pdf(fichier)
        if texte_cv:
            cvs[fichier.name] = texte_cv

    # Analyser les CVs
    results_df = analyzer.analyze_multiple_cvs(cvs, job_description)
    print("\n📊 Résultats de l'analyse :\n", results_df)

    # Trier les CVs en fonction du résultat
    for fichier, score, recommendation, raison in results_df.itertuples(index=False):
        chemin_fichier = dossier_cvs / fichier
        if "Accepté" in recommendation:  # Vérifie si le CV est accepté
            deplacer_fichier(chemin_fichier, DOSSIER_ACCEPTES)
        else:
            deplacer_fichier(chemin_fichier, DOSSIER_REFUSES)

    print("\n✅ Les CVs acceptés ont été déplacés vers 'cvs_acceptes/'.")
    print("❌ Les CVs refusés ont été déplacés vers 'cvs_refuses/'.")


if __name__ == "__main__":
    # Vérifier que le fichier de description existe
    if not FICHIER_OFFRE.exists():
        print(f"❌ Erreur : Le fichier {FICHIER_OFFRE} est introuvable !")
        exit(1)

    # Charger la description de l'offre depuis le fichier
    job_description = lire_fichier(FICHIER_OFFRE)

    # Vérifier que le dossier contenant les CVs existe
    if not DOSSIER_CVS.exists():
        print(f"❌ Erreur : Le dossier {DOSSIER_CVS} est introuvable !")
        exit(1)

    # Analyser et trier les CVs
    analyser_cvs(DOSSIER_CVS, job_description)
