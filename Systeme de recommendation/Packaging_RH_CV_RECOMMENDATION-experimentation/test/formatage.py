# formatage.py
class FormateurRapport:
    @staticmethod
    def generer_rapport(resultats):
        """Génère un rapport formaté des résultats"""
        rapport = f"""
=== Rapport d'évaluation du CV ===
Score de correspondance: {resultats['confiance']}
Recommandation: {resultats['recommandation']}

Compétences trouvées:
- {", ".join(resultats['competences_trouvees']) if resultats['competences_trouvees'] else "Aucune"}

Compétences manquantes:
- {", ".join(resultats['competences_manquantes']) if resultats['competences_manquantes'] else "Aucune"}
"""
        return rapport