import os
import pandas as pd
import sys
import wandb

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_loader import NHLDataLoader

loader = NHLDataLoader()

# Initialisation d'une exécution de Wandb
run = wandb.init(project="IFT6758.2024-A09", entity="toma-allary-universit-de-montr-al")  

# Chargement du DataFrame du match "2017021065"
game_id = "2017021065"
df = loader.load_csv_file(game_id)

# Création d'un artefact
artifact = wandb.Artifact("wpg_v_wsh_2017021065", type="dataset")

# Ajout du DataFrame filtré comme Table
my_table = wandb.Table(dataframe=df)
artifact.add(my_table, "wpg_v_wsh_2017021065")

# Téléchargement de l'artefact
run.log_artifact(artifact)

# Fin de l'exécution
run.finish()
