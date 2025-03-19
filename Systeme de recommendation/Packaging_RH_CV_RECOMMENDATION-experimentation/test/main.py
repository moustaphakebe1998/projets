# main.py
from RH_CV_RECOMMENDATION.test.analyseur import AnalyseurCV
from formatage import FormateurRapport

def main():
    # Exemple d'utilisation
    analyseur = AnalyseurCV(seuil_acceptation=0.3)
    formateur = FormateurRapport()
    
    description_poste = """
    Nous recherchons un développeur Python expérimenté avec des connaissances en 
    machine learning et développement web. Maîtrise de Flask et scikit-learn requise.
    """
    
    cv = """
    Développeur Python senior avec 5 ans d'expérience.
    Expertise en développement web avec Flask et Django.
    Projets en machine learning avec scikit-learn et TensorFlow.
    """
    
    resultats = analyseur.analyser(cv, description_poste)
    rapport = formateur.generer_rapport(resultats)
    print(rapport)

if __name__ == "__main__":
    main()