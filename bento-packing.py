import argparse
import os
import sys
import torch
from libs.logger import Logger

from libs.mlflow_utils import mlflow_utils
from svc import api_service

log = Logger(os.path.basename(__file__).split(".")[0])
logger = log.initLogger("INFO", os.path.basename(__file__).split(".")[0])


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        logger.info("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            logger.info("{}{}".format(subindent, f))


def mlflow_load_model_weights(artifact_name, artifact_path):
    try:
        ml_utils = mlflow_utils()

        return ml_utils.load(model_name=artifact_name, artiface_path=artifact_path, dst="./")
    except Exception as e:
        logger.error("mlflow utils -> mlflow load model & weights : {}".format(e))


def get_instance(argv):
    try:
        color = "\033[32m"
        end = "\033[0m"
        # device = "cpu"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        artifact_name = "[kt-moa]yolov5"
        artifact_path = "artifact"
        model_name = "model.pt"
        if argv == "True":
            logger.info("local model load")
            from models import common
            import gdown

            uri = "https://drive.google.com/drive/u/0/folders/"
            ids = "10zE2YMON9-gH-z1YZCenNf-2RYHvxqVL"
            gdown.download_folder(
                url=uri + ids,
                id=None,
                output=artifact_path,
                quiet=False,
                use_cookies=False,
            )

            model = common.attempt_load(os.path.join(artifact_path, model_name), device)
            uuid, version = "local_uuid", "local_version"
            logger.info(
                "   \n{color}MODEL PATH :{end} {}\
                    \n{color}MLFLOW ID :{end} {}\
                    \n{color}MLFLOW VERSION :{end} {}\
                    \n{color}MLFLOW ARTIFACTS PATH :{end} {}\
                    \n{color}NETWORK INFO :{end} {}\
                    \n{color}DEVICE :{end} {}\
                    \n".format(
                    os.path.join(artifact_path, model_name),
                    uuid,
                    version,
                    artifact_path,
                    "\n{color}anchors : {end}{}\n{color}stride : {end}{}".format(
                        model.yaml["anchors"], model.stride.tolist(), color="\033[33m", end=end
                    ),
                    device,
                    color=color,
                    end=end,
                )
            )
        else:
            logger.info("using mlflow artifact")
            model, uuid, version = mlflow_load_model_weights(
                artifact_name=artifact_name,
                artifact_path=artifact_path,
            )
            logger.info(
                "   \n{color}MODEL PATH :{end} {}\
                    \n{color}MLFLOW NAME :{end} {}\
                    \n{color}MLFLOW ID :{end} {}\
                    \n{color}MLFLOW VERSION :{end} {}\
                    \n{color}MLFLOW ARTIFACTS PATH :{end} {}\
                    \n{color}NETWORK INFO :{end} {}\
                    \n{color}DEVICE :{end} {}\
                    \n".format(
                    os.path.join(artifact_path, model_name),
                    model.name,
                    uuid,
                    version,
                    artifact_path,
                    "\n{color}anchors : {end}{}\n{color}stride : {end}{}".format(
                        model.yaml["anchors"], model.stride.tolist(), color="\033[33m", end=end
                    ),
                    device,
                    color=color,
                    end=end,
                )
            )
        checkpoint = torch.load(os.path.join(artifact_path, model_name), map_location=device)["model"]

        model.load_state_dict(checkpoint.state_dict())
        return model.eval()
    except Exception as e:
        logger.error("Bento packing -> load model : {}".format(e))


def main(argv):

    try:
        instance = get_instance(argv)
        service = api_service()
        service.pack("model", instance)
        service.save()
        logger.info("🚀BENTOML🚀 API Server service initialized")
    except Exception as e:
        logger.error(
            "Bento packing -> bentoml_packing : error please checked bento-packing.py script\n**** Error ****\n{}".format(
                e
            )
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        argv = sys.argv[1]
    else:
        argv = "False"
    print(argv)
    main(argv)
