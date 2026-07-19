#!/usr/bin/env python3
"""
Script CLI pour anonymisation compatible NiFi ExecuteStreamCommand.
Supporte désormais les modes 'mask' et 'pseudo'.
"""

import argparse
import json
import sys
import pandas as pd
from anonymizer.anonymer import anonymer


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Anonymisation et Pseudonymisation de données pour NiFi'
    )
    parser.add_argument(
        '--columns', '-c',
        type=str,
        required=True,
        help='Colonnes à traiter (séparées par des virgules)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['csv', 'json', 'parquet'],
        default='csv',
        help='Format du fichier'
    )
    parser.add_argument(
        '--suffix', '-s',
        type=str,
        default='_anonyme',
        help='Suffixe pour les colonnes traitées'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['mask', 'pseudo'],
        default='mask',
        help='Mode : mask (anonymisation) ou pseudo (identifiant ID_...)'
    )
    parser.add_argument(
        '--salt',
        type=str,
        default='nifi_default_salt',
        help='Sel pour la pseudonymisation (cohérence des ID)'
    )
    parser.add_argument(
        '--use-ner',
        action='store_true',
        help='Activer le modèle NER (CamemBERT)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Activer le traitement parallèle'
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='Garder les colonnes originales'
    )
    parser.add_argument(
        '--separator',
        type=str,
        default=',',
        help='Séparateur CSV'
    )
    return parser.parse_args()


def main():
    """Fonction principale."""
    try:
        args = parse_arguments()
        colonnes = [col.strip() for col in args.columns.split(',')]

        # Lecture des données
        if args.format == 'csv':
            df = pd.read_csv(
                sys.stdin,
                sep=args.separator,
                engine='python'
            )
        elif args.format == 'json':
            data = json.load(sys.stdin)
            df = pd.DataFrame(data)
        else:
            df = pd.read_parquet(sys.stdin.buffer)

        if df.empty:
            return 0

        # Vérification des colonnes
        colonnes_manquantes = set(colonnes) - set(df.columns)
        if colonnes_manquantes:
            print(
                f"ERREUR: Colonnes non trouvées : {colonnes_manquantes}",
                file=sys.stderr
            )
            return 1

        # Traitement
        df_traite = anonymer(
            df,
            colonnes=colonnes,
            suffixe=args.suffix,
            use_ner=args.use_ner,
            parallel=args.parallel,
            nettoyer=True,
            mode=args.mode,
            salt=args.salt
        )

        # Suppression des colonnes originales
        if not args.keep_original:
            df_traite = df_traite.drop(columns=colonnes)

        # Écriture du résultat
        if args.format == 'csv':
            df_traite.to_csv(
                sys.stdout,
                index=False,
                sep=args.separator
            )
        elif args.format == 'json':
            df_traite.to_json(
                sys.stdout,
                orient='records',
                lines=False
            )
        else:
            df_traite.to_parquet(sys.stdout.buffer, index=False)

        return 0

    except Exception as e:
        print(f"ERREUR CLI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
