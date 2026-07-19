#!/usr/bin/env python3
import hashlib
import os
import re
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

_nlp = None
_model_loaded = False

PATTERNS = {
    'iban': r"[A-Z]{2}\d{2}\s?(\d{4}\s?){4,}",
    'phone': r"(?:(\+33|0033|0)\s?[1-9](?:[\s.-]?\d{2}){4})",
    'email': r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    'siret': r"\b\d{14}\b",
    'plate': r"[A-Z]{2}-\d{3}-[A-Z]{2}",
}

NAME_PATTERNS = [
    r'\bM\. [A-Z][a-zàâäçéèêëïîôùûüÿ]+\b',
    r'\bMme [A-Z][a-zàâäçéèêëïîôùûüÿ]+\b',
    r'\bMr [A-Z][a-zàâäçéèêëïîôùûüÿ]+\b',
    r'(?<!^)(?<![\.\!\?]\s)\b[A-Z][a-zàâäçéèêëïîôùûüÿ]+\s+'
    r'[A-Z][a-zàâäçéèêëïîôùûüÿ]+\b',
    r'\b[A-Z][a-zàâäçéèêëïîôùûüÿ]+ [A-Z][A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ]+\b',
]

NER_CONFIDENCE_THRESHOLD = 0.4


def _generer_pseudo_hash(valeur: str, type_donnee: str, salt: str) -> str:
    raw_str = f"{valeur.strip().lower()}:{salt}"
    hash_hex = hashlib.sha256(raw_str.encode('utf-8')).hexdigest()
    hash_digits = ''.join([str(int(c, 16) % 10) for c in hash_hex])

    if type_donnee == 'person':
        return f"PERS_{hash_hex[:8].upper()}"
    elif type_donnee == 'phone':
        return f"06{hash_digits[:8]}"
    elif type_donnee == 'email':
        return f"user_{hash_hex[:8]}@anonymous.com"
    elif type_donnee == 'iban':
        return f"FR76{hash_digits[:20]}"
    else:
        return f"ID_{hash_hex[:8].upper()}"


def _configure_ssl():
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.set_ciphers('DEFAULT@SECLEVEL=1')
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        return ssl_context
    except Exception as e:
        print(f"ERREUR SSL: {e}", file=sys.stderr)
        return ssl.create_default_context()


def _load_ner_model():
    global _nlp, _model_loaded
    if _model_loaded:
        return _nlp
    try:
        from transformers import (AutoTokenizer,
                                  AutoModelForTokenClassification,
                                  pipeline)
        model_name = "Jean-Baptiste/camembert-ner-with-dates"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, local_files_only=True
            )
        _nlp = pipeline('ner', model=model, tokenizer=tokenizer,
                        aggregation_strategy="simple")
        _model_loaded = True
        return _nlp
    except Exception as e:
        print(f"ERREUR NER: {e}", file=sys.stderr)
        _model_loaded = True
        _nlp = None
        return None


def _anonymise_donnees_structurees(texte: str, mode: str = 'mask',
                                   salt: str = "salt") -> str:
    anonymized = texte
    for label, pattern in PATTERNS.items():
        if mode == 'mask':
            anonymized = re.sub(pattern, f"{label.upper()}_ANONYMIZED",
                                anonymized)
        else:
            def repl(match):
                return _generer_pseudo_hash(match.group(0), label, salt)
            anonymized = re.sub(pattern, repl, anonymized)
    return anonymized


def _anonymise_noms_ner(texte: str, mode: str = 'mask',
                        salt: str = "salt") -> str:
    if _nlp is None:
        return texte
    try:
        results = _nlp(texte)
        entities = sorted(
            [e for e in results if (e['score'] > NER_CONFIDENCE_THRESHOLD
                                    and e['entity_group'] == 'PER')],
            key=lambda x: x['start'],
            reverse=True
        )
        anonymized = texte
        for entity in entities:
            val = entity["word"]
            rep = ("[PERSONNE]" if mode == 'mask'
                   else _generer_pseudo_hash(val, 'person', salt))
            anonymized = re.sub(re.escape(val), rep, anonymized)
        return anonymized
    except Exception as e:
        print(f"ERREUR NER: {e}", file=sys.stderr)
        return texte


def _anonymise_noms_regex(texte: str, mode: str = 'mask',
                          salt: str = "salt") -> str:
    anonymized = texte
    for pattern in NAME_PATTERNS:
        if mode == 'mask':
            anonymized = re.sub(pattern, '[PERSONNE]', anonymized)
        else:
            def repl(match):
                return _generer_pseudo_hash(match.group(0), 'person', salt)
            anonymized = re.sub(pattern, repl, anonymized)

    patterns_gen = [
        r'(?<!^)(?<!\. )\b(?:[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\']*\.?\s+)*'
        r'[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\']*(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\']*)*\b',
        r'(?<!^)(?<!\. )\b[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\']*\b'
    ]
    for p in patterns_gen:
        if mode == 'mask':
            anonymized = re.sub(p, '[PERSONNE]', anonymized)
        else:
            def repl_gen(m):
                return _generer_pseudo_hash(m.group(0), 'person', salt)
            anonymized = re.sub(p, repl_gen, anonymized)

    anonymized = re.sub(r'(\[PERSONNE\]\s*)+', '[PERSONNE] ', anonymized)
    return anonymized.strip()


def _masquer_chiffres(texte: str) -> str:
    return "".join("*" if c.isdigit() else c for c in texte)


def _anonymiser_texte(texte: str, use_ner: bool = True,
                      mode: str = 'mask', salt: str = "salt") -> str:
    if pd.isna(texte) or not isinstance(texte, str) or texte.strip() == "":
        return texte if isinstance(texte, str) else str(texte)
    anonymized = _anonymise_donnees_structurees(texte, mode, salt)
    if use_ner and _nlp is not None:
        anonymized = _anonymise_noms_ner(anonymized, mode, salt)
    has_person = "[PERSONNE]" in anonymized or "PERS_" in anonymized
    if not use_ner or _nlp is None or not has_person:
        anonymized = _anonymise_noms_regex(anonymized, mode, salt)
    if mode == 'mask':
        anonymized = _masquer_chiffres(anonymized)
    return anonymized


def anonymer(df: pd.DataFrame, colonnes: Optional[List[str]] = None,
             suffixe: str = "_anonyme", use_ner: bool = False,
             parallel: bool = False, nettoyer: bool = True,
             mode: str = 'mask', salt: str = "static_salt") -> pd.DataFrame:
    if df.empty:
        raise ValueError("Le DataFrame est vide")
    if use_ner:
        ssl_context = _configure_ssl()
        ssl._create_default_https_context = lambda: ssl_context
        _load_ner_model()
    df_res = df.copy()
    if colonnes is None:
        colonnes = df_res.select_dtypes(include=['object']).columns.tolist()
    for col in colonnes:
        df_res[col] = df_res[col].astype(str)
        if nettoyer:
            df_res[col] = df_res[col].str.replace(
                "_x000D_\n", " ", regex=False)
        if parallel and len(df_res) > 1000:
            with ThreadPoolExecutor() as executor:
                res = list(executor.map(
                    lambda x: _anonymiser_texte(x, use_ner, mode, salt),
                    df_res[col]
                ))
            df_res[f"{col}{suffixe}"] = res
        else:
            df_res[f"{col}{suffixe}"] = df_res[col].apply(
                lambda x: _anonymiser_texte(x, use_ner, mode, salt)
            )
    return df_res


def main():
    df_test = pd.DataFrame({
        'id': [1, 2],
        'message': ['Moustapha Kebe. Tel: 0612345678', 'M. Dupont.']
    })
    df_res = anonymer(df_test, colonnes=['message'], mode='pseudo')
    print(df_res)


if __name__ == "__main__":
    main()
