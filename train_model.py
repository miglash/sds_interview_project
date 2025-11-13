import yaml

from src.data_loader import load_as_sales_data
from src.data_features import build_features
from src.model import save_model, train_model

config_path = "./config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

#TODO: validate config

train_pl = load_as_sales_data(config["dataset_path"])

X, Y = build_features(train_pl, config)

model = train_model(X, Y, config=config)

save_model(model)