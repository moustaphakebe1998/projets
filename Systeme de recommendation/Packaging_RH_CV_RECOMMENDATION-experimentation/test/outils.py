# outils.py
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('fr_core_news_md')
        
    def nettoyer_texte(self, texte):
        """Nettoie et normalise le texte"""
        doc = self.nlp(texte.lower())
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)
    
    def extraire_competences(self, texte):
        """Extrait les compétences du texte"""
        doc = self.nlp(texte)
        competences = set([token.text.lower() for token in doc 
                          if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2])
        return competences

class ComparateurCV:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='french')
        self.preprocessor = TextPreprocessor()
        
    def calculer_similarite(self, texte1, texte2):
        """Calcule la similarité entre deux textes"""
        textes_nettoyes = [
            self.preprocessor.nettoyer_texte(texte1),
            self.preprocessor.nettoyer_texte(texte2)
        ]
        tfidf_matrix = self.vectorizer.fit_transform(textes_nettoyes)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]