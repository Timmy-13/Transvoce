import os
import torch 

BATCH_SIZE = 32
EPOCHS = 500
HIDDEN_SIZE = 512
DATA_PATH = "data.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "g2p_model_ur.pt"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)