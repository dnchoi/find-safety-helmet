import base64
import json
import os
import time

from fz_logger import fz_logger
from libs.utils import *

log = fz_logger.Logger(os.path.basename(__file__).split(".")[0])
logger = log.initLogger("INFO", os.path.basename(__file__).split(".")[0])


class model_predict:
    @classmethod
    def pre_processing(cls, parsed_json, device):
        try:
            TEST = False
            if not TEST:
                img_data = base64.b64decode(parsed_json["picture"])
                img_array = np.fromstring(img_data, np.uint8)
                img2 = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
                img_shape = img2.shape[:2]
                img = letterbox(img2, cls.input_dim[2], stride=cls.stride[2], auto=False)[0]

                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float()
                img /= 255.0
                img = img.unsqueeze(0)
                return img, img_shape
            else:
                img2 = parsed_json.copy()
                img_shape = img2.shape[:2]
                img = letterbox(img2, cls.input_dim[2], stride=cls.stride[2], auto=False)[0]

                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float()
                img /= 255.0
                img = img.unsqueeze(0)
                return img, img_shape
        except Exception as e:
            logger.warning("preprocessing : {}".format(e))

    @classmethod
    def post_processing(cls, predict, conf_th, nms_th, ori_img_shape):
        try:
            if predict is not None:
                pred = non_max_suppression(predict, conf_th, nms_th, None, False, max_det=1000)
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(
                            [cls.input_dim[2], cls.input_dim[2]], det[:, :4], ori_img_shape
                        ).round()
                    pred = det.cpu().detach().numpy()
                    return pred
                else:
                    return None
            else:
                return None
        except Exception as e:
            logger.warning("postprocessing : {}".format(e))

    @classmethod
    @torch.no_grad()
    def detect(cls, model, json, device):
        try:
            classes = ["cap", "person"]
            conf_th = json["conf_th"]
            nms_th = json["nms_th"]
            cls.input_dim = [1, 3, 640, 640]
            cls.stride = model.stride.tolist()
            pre_img, img_shape = cls.pre_processing(json, device)
            pred = model(pre_img)[0]
            result = cls.post_processing(predict=pred, conf_th=conf_th, nms_th=nms_th, ori_img_shape=img_shape)
            predicted = {"persons": [], "caps": []}
            if result is not None:
                caps_bboxes = []
                caps_confs = []
                caps_count = 0
                person_bboxes = []
                person_confs = []
                person_count = 0
                for *xyxy, conf, clss in reversed(result):
                    if classes[int(clss)] == "person":
                        person_count += 1
                        bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                        person_bboxes.append((bbox))
                        conf = "{}".format(conf * 100)
                        person_confs.append(float(conf))
                    if classes[int(clss)] == "cap":
                        caps_count += 1
                        bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                        caps_bboxes.append((bbox))
                        conf = "{}".format(conf * 100)
                        caps_confs.append(float(conf))

                predicted = {
                    "persons": [person_count, person_bboxes, person_confs],
                    "caps": [caps_count, caps_bboxes, caps_confs],
                }
                # predicted = [person_count, caps_count, caps_bboxes, caps_confs]
                return predicted
            else:
                return predicted
        except Exception as e:
            logger.error("detector : {}".format(e))
