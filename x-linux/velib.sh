#!/bin/bash
time_shell=$(date +"%Y-%m-%dT%H-%M-%S+00:00")

wget -O "velib_shell_${time_shell}.json" "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin"

wget -O "velib_data_${time_shell}.csv" "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B"

