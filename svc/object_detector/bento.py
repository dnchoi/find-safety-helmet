import base64
import os

import bentoml
import cv2
import numpy as np
import torch
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.artifact import PytorchModelArtifact
from models import common, torch_utils, yolo
from fz_logger import fz_logger
from svc.object_detector.detector import model_predict

log = fz_logger.Logger(os.path.basename(__file__).split(".")[0])
logger = log.initLogger("INFO", os.path.basename(__file__).split(".")[0])


@bentoml.env(
    infer_pip_packages=True,
    pip_packages=["torch", "torchvision", "funzin-vc-utils", "mlflow", "boto3"],
    requirements_txt_file="./requirements.txt",
    docker_base_image="bentoml/model-server:0.13.1-slim-py38",
)
@bentoml.artifacts(
    [
        PytorchModelArtifact("model"),
    ]
)
class api_service(bentoml.BentoService):
    @bentoml.api(
        input=JsonInput(
            http_input_example=[
                {
                    "uuid": "401c739d-d94c-4bd2-a795-9a8740e655b4",
                    "picture": "/9j/4AAQSkZJRgABAQAAAQ...",
                    "conf_th": 0.5,
                    "nms_th": 0.45,
                }
            ],
        ),
        output=JsonOutput(),
        batch=False,
    )
    def predict(self, parsed_json):
        try:
            # device = "cpu"
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            response = model_predict.detect(self.artifacts.model, parsed_json, device)
            return response
        except Exception as e:
            logger.error("API service error : {}".format(e))
