import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:

    PROJECT_NAME = os.getenv(
        "PROJECT_NAME",
        "QuantEdge"
    )

    ENVIRONMENT = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    DEFAULT_TICKER = os.getenv(
        "DEFAULT_TICKER",
        "RELIANCE.NS"
    )

    DEFAULT_PERIOD = os.getenv(
        "DEFAULT_PERIOD",
        "5y"
    )

    DEFAULT_INTERVAL = os.getenv(
        "DEFAULT_INTERVAL",
        "1d"
    )

    DATA_DIR = PROJECT_ROOT / os.getenv(
        "DATA_DIR",
        "data"
    )

    RAW_DATA_DIR = PROJECT_ROOT / os.getenv(
        "RAW_DATA_DIR",
        "data/raw"
    )

    PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv(
        "PROCESSED_DATA_DIR",
        "data/processed"
    )

    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO"
    )


settings = Settings()


def create_directories():

    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "artifacts"
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )