import copy
import cv2
import argparse
import onnxruntime
import numpy as np


class YolopONNX(object):
    def __init__(
        self,
        model_path='yolop-640-640.onnx',
        input_shape=(640, 640),
        class_score_th=0.3,
        nms_th=0.45,
        providers=['CPUExecutionProvider'],
    ):
        # 入力サイズ
        self.input_shape = input_shape

        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

    def inference(self, image):
        height, width, _ = image.shape

        # convert to RGB
        img_rgb = image[:, :, ::-1].copy()

        # resize & normalize
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = self._resize_unscale(
            img_rgb, self.input_shape)

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)

        # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
        det_out, da_seg_out, ll_seg_out = self.onnx_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": img})

        # det_out = torch.from_numpy(det_out).float()
        boxes = self._nms(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
        # boxes = boxes.cpu().numpy().astype(np.float32)

        # scale coords to original size.
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r

        img_det = img_rgb[:, :, ::-1].copy()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(
                label)
            img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0),
                                    2, 2)

        # select da & ll segment area.
        da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

        color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
        color_area[da_seg_mask == 1] = [0, 255, 0]
        color_area[ll_seg_mask == 1] = [255, 0, 0]
        color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        color_mask = np.mean(color_seg, 2)
        img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        # merge: resize to original size
        img_merge[color_mask != 0] = \
            img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img_merge = img_merge.astype(np.uint8)
        img_merge = cv2.resize(img_merge, (width, height),
                               interpolation=cv2.INTER_LINEAR)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(
                label)
            img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2),
                                      (0, 255, 0), 2, 2)

        # da: resize to original size
        da_seg_mask = da_seg_mask * 255
        da_seg_mask = da_seg_mask.astype(np.uint8)
        da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                                 interpolation=cv2.INTER_LINEAR)

        # ll: resize to original size
        ll_seg_mask = ll_seg_mask * 255
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                                 interpolation=cv2.INTER_LINEAR)

        return img_merge

    def _xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _box_iou(self, box1, box2):
        def box_area(box):
            # (x2 - x1) * (y2 - y1)
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (np.min(box1[:, None, 2:], box2[:, 2:]) -
                 np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)

    def _nms(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
    ):
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        max_wh = 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        redundant = True  # require redundant detections
        merge = False  # use merge-NMS

        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self._xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            # conf, j = x[:, 5:].max(1, keepdim=True)
            conf = copy.deepcopy(x[:, 5:])
            j = np.zeros((x[:, 5:].shape[0], 1))
            # x = np.concatenate((box, conf, j), 1)[conf.view(-1) > conf_thres]
            x = np.concatenate((box, conf, j), 1)

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :
                              4] + c, x[:,
                                        4]  # boxes (offset by class), scores
            i = cv2.dnn.NMSBoxes(
                bboxes=boxes.tolist(),
                scores=scores.tolist(),
                score_threshold=0.3,
                nms_threshold=0.45,
                top_k=5000,
            )
            if len(i) > 0:
                i = i.flatten()
                if i.shape[0] > max_det:  # limit detections
                    i = i[:max_det]
                if merge and (
                        1 < n <
                        3E3):  # Merge NMS (boxes merged using weighted mean)
                    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = self._box_iou(boxes[i],
                                        boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = np.dot(weights, x[:, :4]).float() / weights.sum(
                        1, keepdim=True)  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy

                output[xi] = x[i]

        return output

    def _resize_unscale(self, img, new_shape=(640, 640), color=114):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        canvas = np.zeros((new_shape[0], new_shape[1], 3))
        canvas.fill(color)
        # Scale ratio (new / old) new_shape(h,w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
        new_unpad_w = new_unpad[0]
        new_unpad_h = new_unpad[1]
        pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[
            0] - new_unpad_h  # wh padding

        dw = pad_w // 2  # divide padding into 2 sides
        dh = pad_h // 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

        canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

        return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop-640-640.onnx")
    args = parser.parse_args()

    onnx_path = f"./weights/{args.weight}"
    yolop = YolopONNX(
        onnx_path,
        input_shape=(640, 640),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    print(f"Load {onnx_path} done!")

    cap = cv2.VideoCapture("video/sample.mp4")
    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        result_image = yolop.inference(frame)

        result_image = cv2.resize(result_image, dsize=None, fx=0.5, fy=0.5)

        cv2.imshow("YOLOP", result_image)
        cv2.waitKey(1)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
