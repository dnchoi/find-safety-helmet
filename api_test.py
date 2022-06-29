import base64
import os
import random
import sys
import time

import cv2
import requests
from fz_logger import fz_logger

log = fz_logger.Logger(os.path.basename(__file__).split(".")[0])
logger = log.initLogger("DEBUG", os.path.basename(__file__).split(".")[0])
# numpy image to base64 image, base64 to json string, API server request
def get_predict(img):
    try:
        url = "http://localhost:5000/predict"
        image_base64 = image_to_base64(img)  # input : numpy image / output : base64 string
        data = json_maker(
            "401c739d-d94c-4bd2-a795-9a8740e655b4", image_base64, 0.5, 0.45
        )  # input : (UUID, base64 image, conf th, nms th) / output : json string
        response = requests.post(
            url, json=data
        )  # input : (API server address, json string) / output : API server return value
        return response.json()
    except Exception as e:
        print("predict error : {}".format(e))


# image id and base64 image to json dict
def json_maker(uuid, picture, conf, nms):
    json_obj = {
        "uuid": "",
        "picture": None,
        "conf_th": None,
        "nms_th": None,
    }

    json_obj["uuid"] = uuid
    json_obj["picture"] = picture
    json_obj["conf_th"] = conf
    json_obj["nms_th"] = nms

    return json_obj


# image to base64 enconding type
def image_to_base64(img_path):
    image = cv2.imencode(".jpg", img_path)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


# create color list
def gen_colors(classes):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]

    return colors


# draw bbox from predict results
def plot_one_box(xyxy, img, color=None, label=None):
    for x in xyxy:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, 5)
        cv2.putText(
            img,
            label,
            (int(x[0]), int(x[1]) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            thickness=5,
            lineType=cv2.LINE_AA,
        )
    return img


def main(opt):
    t1 = time.time()
    colors = gen_colors(["hardcap", "person"])
    f = cv2.imread(opt[0])

    if f is not None:
        res = get_predict(f)
        if res is not None:
            persons = res["persons"]
            caps = res["caps"]
            if len(opt) > 1:
                if opt[1] == "True":
                    if persons[0] > 0:
                        print(persons[-1])
                        f = plot_one_box(persons[1], f, colors[1], "person")

                    if caps[0] > 0:
                        f = plot_one_box(caps[1], f, colors[0], "hardcap")
                    cv2.imwrite("image_result.jpg", f)
            t2 = time.time()
            ms = t2 - t1
            fps = 1 / ms
            print("fps:{}\tms:{}".format(fps, ms))
            print(persons)
            print(caps)


if __name__ == "__main__":
    argv = sys.argv[1:]
    print(argv)
    main(argv)
