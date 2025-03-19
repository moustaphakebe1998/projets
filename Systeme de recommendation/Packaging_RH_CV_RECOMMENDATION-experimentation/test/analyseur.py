# analyseur.py
from outils import ComparateurCV, TextPreprocessor

class AnalyseurCV:
    def __init__(self, seuil_acceptation=0.3):
        self.comparateur = ComparateurCV()
        self.preprocessor = TextPreprocessor()
        self.seuil_acceptation = seuil_acceptation
        
    def analyser(self, cv_texte, description_poste):
        """Analyse un CV par rapport à une description de poste"""
        # Calcul de la similarité
        similarite = self.comparateur.calculer_similarite(cv_texte, description_poste)
        
        # Extraction des compétences
        competences_poste = self.preprocessor.extraire_competences(description_poste)
        competences_cv = self.preprocessor.extraire_competences(cv_texte)
        
        # Analyse des correspondances
        competences_trouvees = competences_poste.intersection(competences_cv)
        competences_manquantes = competences_poste - competences_cv
        
        return {
            'score_similarite': round(similarite, 2),
            'competences_trouvees': list(competences_trouvees),
            'competences_manquantes': list(competences_manquantes),
            'recommandation': 'Accepté' if similarite >= self.seuil_acceptation else 'Rejeté',
            'confiance': f"{round(similarite * 100)}%"
        }
