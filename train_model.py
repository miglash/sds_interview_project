import yaml
import logging
import argparse

from src.data_loader import load_as_sales_data
from src.data_features import build_features
from src.model_utils import save_model, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("Training.log"), logging.StreamHandler()],
)

parser = argparse.ArgumentParser("Model Training")
parser.add_argument(
    "-c",
    "--config",
    help="Path of a config file to use instead of default",
    type=str,
)
args = parser.parse_args()

if args.config is None:
    config_path = "./config.yaml"
else:
    config_path = args.config

logging.info(f"Config path set to: {config_path}")
# TODO: validate config path

with open(config_path) as f:
    config = yaml.safe_load(f)

# TODO: validate config

train_pl = load_as_sales_data(config["dataset_path"])
logging.info("Data Loaded successfully")

X, Y = build_features(train_pl, config)
logging.info("Training features built successfully")

model = train_model(X, Y, config=config)
logging.info("Model training complete")

model_path = save_model(model, path="./models/forecast_model_latest.pkl")
logging.info(f"Model saved: {model_path}")
