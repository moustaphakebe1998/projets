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
    
         # Nettoie et prépare le texte pour l'analyse.
        # Convertir en minuscules
        text = text.lower()
        # Supprimer les caractères spéciaux
        text = re.sub(r'[^\w\s]', ' ', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def calculate_match_score(self, cv_text, job_description):
        #Calcule le score de correspondance entre un CV et une description de poste.
        # Prétraitement des textes
        cv_text = self.preprocess_text(cv_text)
        job_description = self.preprocess_text(job_description)
        
        # Vectorisation des textes
        tfidf_matrix = self.vectorizer.fit_transform([job_description, cv_text])
        
        # Calcul de la similarité cosinus
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity * 100  # Convertir en pourcentage
    
    def analyze_cv(self, cv_text, job_description, threshold=40):
        #Analyse un CV et décide s'il correspond au poste.
        match_score = self.calculate_match_score(cv_text, job_description)
        
        analysis = {
            'score': round(match_score, 2),
            'recommendation': 'Accepté' if match_score >= threshold else 'Rejeté',
            'raison': f"Score de correspondance de {round(match_score, 2)}% " +
                     f"({'supérieur' if match_score >= threshold else 'inférieur'} au seuil de {threshold}%)"
        }
        
        return analysis

    def analyze_multiple_cvs(self, cvs_dict, job_description, threshold=30):
        #Analyse plusieurs CV et les classe par ordre de pertinence.
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
