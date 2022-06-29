<!-- # [kt-moa] 안전모 검출 서비스

## Object detector

* 안전모 검출을 위한 Deep learning model 구현

* 사용 모델 : YoloV5

```bash
from  n    params  module                                  arguments
  0                -1  1      8800  models.common.Conv                      [3, 80, 6, 2, 2]
  1                -1  1    115520  models.common.Conv                      [80, 160, 3, 2]
  2                -1  4    309120  models.common.C3                        [160, 160, 4]
  3                -1  1    461440  models.common.Conv                      [160, 320, 3, 2]
  4                -1  8   2259200  models.common.C3                        [320, 320, 8]
  5                -1  1   1844480  models.common.Conv                      [320, 640, 3, 2]
  6                -1 12  13125120  models.common.C3                        [640, 640, 12]
  7                -1  1   7375360  models.common.Conv                      [640, 1280, 3, 2]
  8                -1  4  19676160  models.common.C3                        [1280, 1280, 4]
  9                -1  1   4099840  models.common.SPPF                      [1280, 1280, 5]
 10                -1  1    820480  models.common.Conv                      [1280, 640, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  4   5332480  models.common.C3                        [1280, 640, 4, False]
 14                -1  1    205440  models.common.Conv                      [640, 320, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  4   1335040  models.common.C3                        [640, 320, 4, False]
 18                -1  1    922240  models.common.Conv                      [320, 320, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  4   4922880  models.common.C3                        [640, 640, 4, False]
 21                -1  1   3687680  models.common.Conv                      [640, 640, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  4  19676160  models.common.C3                        [1280, 1280, 4, False]
 24      [17, 20, 23]  1     47103  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [320, 640, 1280]]
```

- input dimension → [1, 3, 640, 640] #[batch size, channels, width, height]
- outputs → [[1, 3, 80, 80, 85], [1, 3, 40, 40, 85], [1, 3, 20, 20, 85]] #[anchor 0, anchor 1, anchor 2]
- Training dataset
    - AIhub data
        - [https://aihub.or.kr/aidata/33921](https://aihub.or.kr/aidata/33921)
- Model → XL model

## Bento API Server

- Requirements
    - Python ≥ 3.8
        - PyPI reauirements

            ```
            aiohttp==3.8.1
            aiohttp-cors==0.7.0
            aiosignal==1.2.0
            alembic==1.7.7
            astroid==2.11.5
            asttokens==2.0.5
            async-timeout==4.0.2
            backcall==0.2.0
            bentoml==0.13.1
            black==22.3.0
            boto3==1.23.2
            botocore==1.26.2
            cerberus==1.3.4
            chardet==4.0.0
            charset-normalizer==2.0.12
            click==8.1.3
            cloudpickle==2.1.0
            commonmark==0.9.1
            configparser==5.2.0
            contextlib2==21.6.0
            cycler==0.11.0
            databricks-cli==0.16.6
            decorator==5.1.1
            deepmerge==1.0.1
            dill==0.3.5.1
            docker==5.0.3
            entrypoints==0.4
            executing==0.8.3
            flake8==4.0.1
            flask==2.1.2
            flatbuffers==2.0
            fonttools==4.33.3
            frozenlist==1.3.0
            funzin-vc-utils==0.1
            gitdb==4.0.9
            gitpython==3.1.27
            grpcio==1.46.1
            gunicorn==20.1.0
            humanfriendly==10.0
            idna==3.3
            importlib-metadata==4.11.3
            importlib-resources==5.7.1
            ipython==8.4.0
            isort==5.10.1
            itsdangerous==2.1.2
            jedi==0.18.1
            jinja2==3.1.2
            jmespath==1.0.0
            kiwisolver==1.4.2
            lazy-object-proxy==1.7.1
            mako==1.2.0
            markupsafe==2.1.1
            matplotlib==3.5.2
            matplotlib-inline==0.1.3
            mccabe==0.6.1
            mlflow==1.26.0
            multidict==6.0.2
            numpy==1.22.3
            oauthlib==3.2.0
            pandas==1.4.2
            parso==0.8.3
            pexpect==4.8.0
            pickleshare==0.7.5
            pillow==9.1.1
            platformdirs==2.5.2
            prometheus-client==0.14.1
            prometheus-flask-exporter==0.20.1
            prompt-toolkit==3.0.29
            protobuf==3.20.1
            psutil==5.9.0
            ptyprocess==0.7.0
            pure-eval==0.2.2
            pycodestyle==2.8.0
            pyflakes==2.4.0
            pygments==2.12.0
            pyjwt==2.4.0
            pylint==2.13.9
            pyparsing==3.0.9
            python-dateutil==2.8.2
            python-json-logger==2.0.2
            pytz==2022.1
            pyyaml==6.0
            querystring-parser==1.2.4
            requests==2.27.1
            rich==12.4.4
            ruamel-yaml==0.17.21
            ruamel-yaml-clib==0.2.6
            s3transfer==0.5.2
            schema==0.7.5
            scipy==1.8.0
            seaborn==0.11.2
            simple-di==0.1.5
            six==1.16.0
            smmap==5.0.0
            sqlalchemy==1.3.24
            sqlalchemy-utils==0.36.5
            sqlparse==0.4.2
            stack-data==0.3.0
            tabulate==0.8.9
            thop==0.0.31-2005241907
            tomli==2.0.1
            torch==1.10.1
            torchsummary==1.5.1
            torchvision==0.11.2
            tqdm==4.64.0
            traitlets==5.3.0
            typing-extensions==4.2.0
            urllib3==1.25.11
            wcwidth==0.2.5
            websocket-client==1.3.2
            werkzeug==2.1.2
            wrapt==1.14.1
            yarl==1.7.2
            zipp==3.8.0
            ```

- API name
    - http://192.168.1.194:5000:/predict
- API Json input

    | name | type |
    | --- | --- |
    | uuid | string |
    | picture | base64 encoding string |
    | conf threashold | float |
    | nms threashold | float |

    ```json
    {
    	"uuid": "401c739d-d94c-4bd2-a795-9a8740e655b4",
    	"picture": "eAF8Q28FrBrC3gifSJp8vNGM5bAHygZHPeuak....",
    	"conf_th": 0.5,
    	"nms_th": 0.45,
    }
    ```

- API Json output


    | classes | name | type |
    | --- | --- | --- |
    | persons | number | int |
    |  | bboxes | list(vector 2d array) |
    |  | confs | float array |
    | caps | number | int |
    |  | bboxes | list(vector 2d array) |
    |  | confs | float array |

    ```json
    {
        "persons": [
            2,
            [
                [
                    223.0,
                    216.0,
                    267.0,
                    324.0
                ],
                [
                    282.0,
                    113.0,
                    457.0,
                    476.0
                ]
            ],
            [
                78.14946174621582,
                91.7441725730896
            ]
        ],
        "caps": [
            1,
            [
                [
                    325.0,
                    114.0,
                    377.0,
                    199.0
                ]
            ],
            [
                90.39649367332458
            ]
        ]
    }
    ```


## How to

### Setting python env

```bash
python3 --version # upper 3.8

sudo apt -y install python3-pip

cd kt-moa-hardcap-api-server

pip3 install -r requirements.txt
```

### Setting .bashrc

```bash
vim ~/.bashrc

.
.
.

# MLFLOW Initialize parameters
export MLFLOW_TRACKING_URI=http://192.168.1.145:31442
export MLFLOW_S3_ENDPOINT_URL=http://192.168.1.145:30575
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

### Create API server

```bash
cd kt-moa-hardcap-api-server

python3 bento-packing.py

[2022-06-24 16:48:00,985] WARNING - Importing from "bentoml.artifact.*" has been deprecated. Instead, use`bentoml.frameworks.*` and `bentoml.service.*`. e.g.:, `from bentoml.frameworks.sklearn import SklearnModelArtifact`, `from bentoml.service.artifacts import BentoServiceArtifact`, `from bentoml.service.artifacts.common import PickleArtifact`
[2022-06-24 16:48:01,091] WARNING - Ignoring pip_packages as requirements_txt_file is set.
[2022-06-24 16:48:01,116] INFO - Using user specified docker base image: `bentoml/model-server:0.13.1-slim-py38`, usermust make sure that the base image either has Python 3.8 or conda installed.
[INFO    ] <bento-packing>: bento-packing:47:
MODEL PATH : artifact/model.pt
MLFLOW NAME : [kt-moa]yolov5
MLFLOW ID : 6370f1444ae040a1a1414aaa1910b6ba
MLFLOW VERSION : 2
MLFLOW ARTIFACTS PATH : artifact
NETWORK INFO :
anchors : [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
stride : [8.0, 16.0, 32.0]
DEVICE : cuda

[2022-06-24 16:48:33,739] WARNING - BentoML by default does not include spacy and torchvision package when using PytorchModelArtifact. To make sure BentoML bundle those packages if they are required for your model, either import those packages in BentoService definition file or manually add them via `@env(pip_packages=['torchvision'])` when defining a BentoService
[2022-06-24 16:48:35,253] INFO - BentoService bundle 'api_service:20220624164834_D7B429' saved to: /home/luke/bentoml/repository/api_service/20220624164834_D7B429
[INFO    ] <bento-packing>: bento-packing:84:  🚀BENTOML🚀 API Server service initialized
```

### Run API server

```bash
bentoml serve api_service:latest

[2022-06-24 16:49:39,004] INFO - Getting latest version api_service:20220624164834_D7B429
[2022-06-24 16:49:39,015] INFO - Starting BentoML API proxy in development mode..
[2022-06-24 16:49:39,016] INFO - Starting BentoML API server in development mode..
[2022-06-24 16:49:39,075] INFO - Your system nofile limit is 1048576, which means each instance of microbatch service is able to hold this number of connections at same time. You can increase the number of file descriptors for the server process, or launch more microbatch instances to accept more concurrent connection.
======== Running on http://0.0.0.0:5000 ========
(Press CTRL+C to quit)
[2022-06-24 16:49:39,367] WARNING - Importing from "bentoml.artifact.*" has been deprecated. Instead, use`bentoml.frameworks.*` and `bentoml.service.*`. e.g.:, `from bentoml.frameworks.sklearn import SklearnModelArtifact`, `from bentoml.service.artifacts import BentoServiceArtifact`, `from bentoml.service.artifacts.common import PickleArtifact`
[2022-06-24 16:49:40,019] WARNING - Ignoring pip_packages as requirements_txt_file is set.
[2022-06-24 16:49:40,030] INFO - Using user specified docker base image: `bentoml/model-server:0.13.1-slim-py38`, usermust make sure that the base image either has Python 3.8 or conda installed.
[2022-06-24 16:49:40,030] WARNING - Ignoring pip_packages as requirements_txt_file is set.
[2022-06-24 16:49:40,032] INFO - Using user specified docker base image: `bentoml/model-server:0.13.1-slim-py38`, usermust make sure that the base image either has Python 3.8 or conda installed.
[2022-06-24 16:49:45,120] WARNING - BentoML by default does not include spacy and torchvision package when using PytorchModelArtifact. To make sure BentoML bundle those packages if they are required for your model, either import those packages in BentoService definition file or manually add them via `@env(pip_packages=['torchvision'])` when defining a BentoService
 * Serving Flask app 'api_service' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:59065 (Press CTRL+C to quit)
```

### Request API

```bash
"http://127.0.0.1:5000/predict"

#or

"http://{insert your ip address}:5000/predict"
```

### Test API

```bash
cd kt-moa-hardcap-api-server

python3 api_test.py {video file.avi(mp4) or camera number(0)}
``` -->

# This project is REST API, safety helmet detection based YoloV5 with Bentoml and Mlflow