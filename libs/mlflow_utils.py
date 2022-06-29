import os

import mlflow.pyfunc
import mlflow.pytorch
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
import torch


class mlflow_utils:
    def __init__(self) -> None:
        self.client = MlflowClient()

    def load(self, model_name: str, artiface_path: str, dst: str):
        model_url = f"models:/{model_name}/Production"

        try:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            model = mlflow.pytorch.load_model(model_uri=model_url, map_location=device)
        except RestException as e:
            print(e.message)
            return -1
        model.name = model_name

        uuid, version = self.download_ckpt(model_name, artiface_path, dst)
        return model, uuid, version

    def check_ckpt(self, ckpt_path):
        return os.path.isfile(ckpt_path)

    def resume_download_ckpt(self, runid):
        download_path = f"resume/{runid}"
        ckpt_path = f"{download_path}/ckpt/last.ckpt"
        if self.check_ckpt(ckpt_path):
            return ckpt_path
        else:
            os.makedirs(download_path, exist_ok=True)
            mlflow_run_id_artifacts_name = "ckpt"
            self.client.download_artifacts(runid, mlflow_run_id_artifacts_name, dst_path=download_path)
            return ckpt_path

    def download_ckpt(self, model_name, artiface_name, dst):
        mlflow_run_id, mlflow_run_version = self._get_model_runid(
            self._get_model_info(model_name), "Production"
        )
        download_path = dst
        mlflow_run_id_artifacts_name = artiface_name
        self.client.download_artifacts(mlflow_run_id, mlflow_run_id_artifacts_name, dst_path=download_path)
        return mlflow_run_id, mlflow_run_version

    def _get_model_info(self, model_name):
        filter_string = f"name='{model_name}'"
        results = self.client.search_model_versions(filter_string)
        return results

    def _get_model_runid(self, results, stage):
        for res in results:
            if res.current_stage == stage:
                # if res.current_stage == "Production":
                deploy_version = res.version
                deploy_run_id = res.run_id
        return deploy_run_id, deploy_version
