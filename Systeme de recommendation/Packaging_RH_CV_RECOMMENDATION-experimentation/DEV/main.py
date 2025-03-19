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
