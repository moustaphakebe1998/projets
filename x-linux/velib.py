#! /usr/bin/env python3
import requests
from datetime import datetime
time_py=datetime.now().strftime("%Y-%m-%dT%H-%M-%S+00:00")
reponse=requests.get("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin")

reponse2=requests.get("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")

with open(f"velib_py_{time_py}.json","wb") as file:
	file.write(reponse.content)

with open(f"velib_py_{time_py}.csv","wb") as file:
        file.write(reponse2.content)
