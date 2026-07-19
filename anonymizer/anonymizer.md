# Fiche descriptive - Solution d'Anonymisation

## Acteurs

| Référent     | Nom              |
| ------------ | ---------------- |
| SIAD   | Moustapha Kebe   |

---

## Présentation fonctionnelle de la solution

### Vue d'ensemble

Cette solution permet l'**anonymisation et la pseudonymisation** de données sensibles (téléphones, emails, noms, IBAN, etc.) via un script Python utilisable en ligne de commande ou intégré dans Apache NiFi.

### Fonctionnalités principales

-  Anonymisation irréversible (RGPD compliant)
-  Pseudonymisation réversible (avec clé secrète)
-  Support multi-formats : CSV, JSON, Parquet
-  Détection intelligente des entités (regex + NER optionnel)
-  Mode parallèle pour gros volumes
-  Conservation optionnelle des colonnes originales

---

## Guide d'utilisation - anonymer_cli.py

### Syntaxe de base

```bash
cat fichier.csv | python anonymer_cli.py --columns COLONNES [OPTIONS]
```

### Arguments disponibles

| Argument          | Court | Obligatoire | Description                              | Valeur par défaut |
|-------------------|-------|-------------|------------------------------------------|-------------------|
| `--columns`       | `-c`  |  Oui      | Colonnes à traiter (séparées par `,`)    | -                 |
| `--format`        | `-f`  |  Non      | Format du fichier                        | `csv`             |
| `--suffix`        | `-s`  |  Non      | Suffixe des colonnes anonymisées         | `_anonyme`        |
| `--mode`          | `-m`  |  Non      | Mode de traitement                       | `mask`            |
| `--separator`     |       |  Non      | Séparateur CSV                           | `,`               |
| `--salt`          |       |  Non      | Clé secrète (mode `pseudo`)              | Auto-généré       |
| `--keep-original` |       |  Non      | Conserver les colonnes originales (flag) | `false`           |
| `--use-ner`       |       |  Non      | Activer la détection NER (flag)          | `false`           |
| `--parallel`      |       |  Non      | Mode parallèle (flag)                    | `false`           |

#### Détail des valeurs

- **`--format`** : `csv`, `json`, `parquet`
- **`--mode`** : 
  - `mask` : Anonymisation irréversible (ex: `[PERSONNE]`, `PHONE_ANONYMIZED`)
  - `pseudo` : Pseudonymisation réversible (ex: `Jean Martin`, `06a1b2c3d4`)
- **`--separator`** : `,` ou `;` (pour CSV français)

---

## Exemples d'utilisation

### Version 1 : Minimal (colonne unique)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone
```
**Résultat :** Colonne `telephone_anonyme` ajoutée

---

### Version 2 : Garder les colonnes originales
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone \
  --keep-original
```
**Résultat :** Colonnes `telephone` ET `telephone_anonyme`

---

### Version 3 : Plusieurs colonnes
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,email,nom
```
**Résultat :** 3 colonnes anonymisées ajoutées

---

### Version 4 : Suffixe personnalisé
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone \
  --suffix _masque
```
**Résultat :** Colonne `telephone_masque`

---

### Version 5 : Avec NER (détection avancée des noms)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,nom \
  --use-ner
```
 Plus lent mais détecte mieux les noms de personnes

---

### Version 6 : Mode parallèle (gros volumes)
```bash
cat big_file.csv | python anonymer_cli.py \
  --columns telephone \
  --parallel
```
 Recommandé pour fichiers > 1000 lignes

---

### Version 7 : CSV avec point-virgule (format français)
```bash
cat data_fr.csv | python anonymer_cli.py \
  --columns telephone \
  --separator ";"
```

---

### Version 8 : Mode pseudonymisation (réversible)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,email \
  --mode pseudo \
  --salt "ma_cle_secrete_2024"
```
**Résultat :** Pseudonymes cohérents (même valeur → même pseudonyme)

---

### Version 9 : COMPLÈTE (production)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,email,nom \
  --format csv \
  --suffix _secure \
  --separator "," \
  --keep-original \
  --parallel \
  --mode mask \
  > output_secure.csv
```

---

## Cas d'usage pratiques

### Cas 1 : Anonymiser téléphone + email (simple)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,email \
  > output.csv
```

---

### Cas 2 : Debug - Garder toutes les colonnes
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,email \
  --keep-original \
  > output.csv
```
**Usage :** Comparer avant/après

---

### Cas 3 : Conformité RGPD
```bash
cat clients.csv | python anonymer_cli.py \
  --columns nom,prenom,telephone,email,adresse \
  --suffix _rgpd \
  --mode mask \
  > clients_rgpd.csv
```

---

### Cas 4 : Environnement de développement
```bash
cat prod_data.csv | python anonymer_cli.py \
  --columns telephone,email,nom \
  --mode pseudo \
  --salt "dev_env_2024" \
  > dev_data.csv
```
**Avantage :** Données cohérentes et réversibles pour les tests

---

### Cas 5 : Gros fichier (>10 000 lignes)
```bash
cat big_data.csv | python anonymer_cli.py \
  --columns telephone,email \
  --parallel \
  --mode mask \
  > big_data_anonym.csv
```

---

### Cas 6 : Tout anonymiser sauf ID et date
```bash
cat data.csv | python anonymer_cli.py \
  --columns nom,prenom,telephone,email,adresse,message,commentaire \
  > data_anonym.csv
```

---

## Configuration Apache NiFi

### ExecuteStreamCommand - Paramètres

| Propriété                  | Valeur                                    |
|----------------------------|-------------------------------------------|
| **Working Directory**      | `/py_scripts/anonymizer`                  |
| **Command Path**           | `/py_scripts/anonymizer/venv/bin/python3` |
| **Command Arguments Strategy** | Command Arguments Property            |
| **Argument Delimiter**     | `\|`                                      |
| **Ignore STDIN**           | `false`                                   |

### Command Arguments (Anonymisation)

```
anonymer_cli.py|--columns|nom,telephone,email,message|--keep-original|--parallel|--mode|mask|--format|csv|--suffix|_anonyme|--separator|;
anonymer_cli.py|--columns|nom,telephone,email,message|--keep-original|--parallel|--mode|pseudo|--format|csv|--suffix|_anonyme|--separator|;
```

 **IMPORTANT** : 
- Pas de guillemets autour du séparateur : utiliser `;` et non `";"`
- Le délimiteur `|` gère la séparation des arguments

### Command Arguments (Pseudonymisation)

```
anonymer_cli.py|--columns|nom,telephone,email,message|--mode|pseudo|--salt|nifi_prod_key_2024|--format|csv|--separator|;
```

---

## Tests progressifs

### Test 1 : Minimal
```bash
cat test_multi.csv | python anonymer_cli.py -c telephone
```

### Test 2 : + Garder original
```bash
cat test_multi.csv | python anonymer_cli.py -c telephone --keep-original
```

### Test 3 : + Plusieurs colonnes
```bash
cat test_multi.csv | python anonymer_cli.py -c telephone,email --keep-original
```

### Test 4 : + Suffixe personnalisé
```bash
cat test_multi.csv | python anonymer_cli.py -c telephone,email -s _masque
```

### Test 5 : Complet
```bash
cat test_multi.csv | python anonymer_cli.py \
  -c telephone,email,nom \
  -s _secure \
  --keep-original \
  > final.csv
```

---

## Recommandations selon le contexte

### Pour NiFi (production) - Anonymisation
```bash
cat data.csv | python anonymer_cli.py \
  --columns telephone,email,nom \
  --mode mask \
  --parallel
```
 **Rapide et sécurisé**

---

### Pour debug/développement
```bash
cat data.csv | python anonymer_cli.py \
  --columns telephone,email \
  --keep-original \
  > debug.csv
```
 **Permet de comparer avant/après**

---

### Pour conformité RGPD (publication)
```bash
cat data.csv | python anonymer_cli.py \
  --columns nom,prenom,telephone,email,adresse \
  --suffix _rgpd \
  --mode mask \
  > data_rgpd.csv
```
 **Anonymisation irréversible**

---

### Pour environnement de test
```bash
cat prod_data.csv | python anonymer_cli.py \
  --columns nom,prenom,telephone,email \
  --mode pseudo \
  --salt "test_env_fixtures" \
  > test_data.csv
```
 **Pseudonymes cohérents et reproductibles**

---

### Pour gros volumes (>10K lignes)
```bash
cat big_data.csv | python anonymer_cli.py \
  --columns telephone,email \
  --parallel \
  --mode mask \
  > big_data_anonym.csv
```
 **Traitement parallèle optimisé**

---

## Exemple de résultat

### Données originales (test_multi.csv)
```csv
id,nom,telephone,email
1,M. Dupont,0612345678,jean.dupont@example.com
2,Mme Martin,0698765432,marie.martin@example.com
```

### Après anonymisation (mode mask)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns nom,telephone,email \
  --keep-original
```

**Résultat :**
```csv
id,nom,telephone,email,nom_anonyme,telephone_anonyme,email_anonyme
1,M. Dupont,0612345678,jean.dupont@example.com,[PERSONNE],PHONE_ANONYMIZED,EMAIL_ANONYMIZED
2,Mme Martin,0698765432,marie.martin@example.com,[PERSONNE],PHONE_ANONYMIZED,EMAIL_ANONYMIZED
```

### Après pseudonymisation (mode pseudo)
```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns nom,telephone,email \
  --mode pseudo \
  --salt "demo_key_123" \
  --keep-original
```

**Résultat :**
```csv
id,nom,telephone,email,nom_pseudo,telephone_pseudo,email_pseudo
1,M. Dupont,0612345678,jean.dupont@example.com,Jean Martin,06a1b2c3d4,usera1b2c3d4@example.com
2,Mme Martin,0698765432,marie.martin@example.com,Sophie Bernard,06e5f6a7b8,usere5f6a7b8@example.com
```

---

## Commande recommandée pour votre projet

Basé sur `test_multi.csv` et l'utilisation dans NiFi :

```bash
cat test_multi.csv | python anonymer_cli.py \
  --columns telephone,email,nom \
  --mode mask \
  --suffix _anonyme \
  --separator "," \
  --parallel \
  > test_multi_anonyme.csv
```

**Configuration NiFi correspondante :**
```
anonymer_cli.py|--columns|telephone,email,nom|--mode|mask|--suffix|_anonyme|--separator|,|--parallel
```



## Dépannage

### Erreur : `unrecognized arguments`
**Cause :** Guillemets autour du séparateur dans NiFi

**Solution :** Utiliser `;` au lieu de `";"`

---

### Erreur : `No columns specified`
**Cause :** L'argument `--columns` est manquant

**Solution :** Toujours spécifier au moins une colonne

---

### Performance lente
**Cause :** Fichier volumineux sans `--parallel`

**Solution :** Ajouter le flag `--parallel`

---

### Colonnes vides dans le résultat
**Cause :** Séparateur incorrect (`,` vs `;`)

---

## Support et contact

| Rôle       | Contact          |
|------------|------------------|
| Référent   | Moustapha Kebe   |
| Équipe     | SIAD       |

---

**Version du document :** 2.0  
**Dernière mise à jour :** 12 février 2026