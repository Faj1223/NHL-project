import wandb
import pandas as pd

# Initialisation d'une exécution de Wandb
run = wandb.init(project="IFT6758.2024-A09", entity="toma-allary-universit-de-montr-al")  

# Chargement du DataFrame filtré
df = pd.read_csv("/ift6758/data/play_by_play/csv/2017")  
df_filtered = df[df["game_id"] == "2017021065"]  

# Création d'un artefact
artifact = wandb.Artifact("wpg_v_wsh_2017021065", type="dataset")

# Ajout du DataFrame filtré comme Table
my_table = wandb.Table(dataframe=df_filtered)
artifact.add(my_table, "wpg_v_wsh_2017021065")

# Téléchargement de l'artefact
run.log_artifact(artifact)

artifact_link = artifact.url
print(f"Lien direct vers l'artefact : {artifact_link}")
# fin de l'exécution
run.finish()
