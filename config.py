from pathlib import Path
import os

def get_config():
    return {"src_lang" : 'en',
            "tgt_lang" : 'it',
            "tokenizer" : "tokenizer_{0}.json",
            "seq_len" : 512,
            'clip' : 1,
            "datasource" : "opus_books",
            "d_model" : 512,
            "num_blocks" : 6,
            "num_heads": 8,
            "batchsize" : 8,
            "epochs" : 30,
            "lr" : 3e-4,
            "model_folder": "weights",
            "model_basename": "transformerModel_",
            "preload": "latest",
            "experiment_name": "runs/pytorch-transformers"
            } 

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    # Ensure the folder exists
    model_folder_path = Path('.') / model_folder
    if not model_folder_path.exists():
        os.makedirs(model_folder_path)
    return str(model_folder_path / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    model_folder_path = Path(model_folder)
    
    # Ensure the folder exists
    if not model_folder_path.exists():
        os.makedirs(model_folder_path)  # Create the folder if it doesn't exist
        return None  # No weights available if folder doesn't exist

    weights_files = list(model_folder_path.glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])