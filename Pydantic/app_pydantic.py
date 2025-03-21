import sys
import csv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


""" class User(BaseModel):
    id: int = Field(description="Identifiant de l'utilisateur", le=1235)
    name: str = Field(description="Nom de l'utilisateur", max_length=5)
    last_name: str = Field(description="Prénom de l'utilisateur", max_length=300)
    email: EmailStr = Field(description="Email de l'utilisateur")
    date_validate: Optional[datetime] = Field(default_factory=datetime.now) """


class Parc(BaseModel):
    num_du_parc: str
    parc: str
    arrdt: int
    insertion_date: datetime
    date_validate: Optional[datetime] = Field(default_factory=datetime.now)


def load_csv(filepath, delimiter=','):
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        data = [row for row in reader]
    return data


def load_csv_from_stdin(delimiter=';'):
    input_stream = sys.stdin.read()
    lines = input_stream.splitlines()
    reader = csv.DictReader(lines, delimiter=delimiter)
    data = [row for row in reader]
    return data


def validate_data(data, Model):
    validated_data = []
    for index, item in enumerate(data):
        try:
            validated_data.append(Model(**item))
        except ValidationError as e:
            print(f"\nErreur sur les données de la ligne: {index+2}:\n{e}\n\n")
            colonne_error = [error['loc'] for error in e.errors()]
            print(f"Ce qu'il faut changer à la :\n>>> ligne : {index+2}\n>>> colonne: {colonne_error}\n>>> Données: {item}")
    return validated_data


def save_data_validate_to_csv(data):
    if not data:
        print("Aucune donnée validée à sauvegarder.")
    fieldnames = data[0].dict().keys()
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    for item in data:
        writer.writerow(item.dict())


if __name__ == "__main__":
    data = load_csv_from_stdin()
    validated_data = validate_data(data, Parc)
    # Convert validated data to DataFrame
    df = pd.DataFrame([item.dict() for item in validated_data])
    # Save validated data to CSV
    # save_data_validate_to_csv(validated_data)
    # Save validated data to Parquet
    if not df.empty:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, sys.stdout.buffer)
