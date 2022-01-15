import copy

import cv2
import numpy as np
import onnxruntime


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
        # 前処理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_rgb_image, r, dw, dh, new_unpad_w, new_unpad_h = self._resize_unscale(
            rgb_image,
            self.input_shape,
        )

        input_image = resize_rgb_image.copy().astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_image = (input_image / 255 - mean) / std
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')

        # 推論
        det_out, drive_area_seg, lane_line_seg = self.onnx_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": input_image},
        )

        # NMS
        bboxes = self._nms(
            det_out,
            conf_thres=self.class_score_th,
            iou_thres=self.nms_th,
            agnostic=False,
        )[0]

        # バウンディングボックスの座標を元画像のスケールに変換
        bboxes[:, 0] -= dw
        bboxes[:, 1] -= dh
        bboxes[:, 2] -= dw
        bboxes[:, 3] -= dh
        bboxes[:, :4] /= r

        # 各セグメンテーション領域を選択する
        drive_area_seg = drive_area_seg[:, :, dh:dh + new_unpad_h,
                                        dw:dw + new_unpad_w]
        lane_line_seg = lane_line_seg[:, :, dh:dh + new_unpad_h,
                                      dw:dw + new_unpad_w]

        drive_area_seg_mask = np.argmax(drive_area_seg, axis=1)[0]
        lane_line_seg_mask = np.argmax(lane_line_seg, axis=1)[0]

        return bboxes, drive_area_seg_mask, lane_line_seg_mask

    def draw(
        self,
        image,
        bboxes,
        drive_area_seg_mask,
        lane_line_seg_mask,
    ):
        image_height, image_width, _ = image.shape
        debug_image = copy.deepcopy(image)

        # 道路セグメンテーション
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = [0, 255, 0]

        mask = np.stack((drive_area_seg_mask, ) * 3, axis=-1).astype('uint8')
        mask = cv2.resize(
            mask,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = np.where(mask > 0.5, 0, 1)

        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

        # レーンセグメンテーション
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = [255, 0, 0]

        mask = np.stack((lane_line_seg_mask, ) * 3, axis=-1).astype('uint8')
        mask = cv2.resize(
            mask,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = np.where(mask > 0.5, 0, 1)

        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

        # 車バウンディングボックス
        for bbox in bboxes:
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            # score = bbox[4]
            # class_id = int(bbox[5])
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2, 2)

        return debug_image

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

        resize_rgb_image = np.zeros((new_shape[0], new_shape[1], 3))
        resize_rgb_image.fill(color)
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

        resize_rgb_image[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

        return resize_rgb_image, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


if __name__ == "__main__":
    yolop = YolopONNX(
        model_path='weights/yolop-640-640.onnx',
        input_shape=(640, 640),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    video_capture = cv2.VideoCapture("video/sample.mp4")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        bboxes, da_seg_mask, ll_seg_mask = yolop.inference(frame)
        result_image = yolop.draw(
            frame,
            bboxes,
            da_seg_mask,
            ll_seg_mask,
        )

        result_image = cv2.resize(result_image, dsize=None, fx=0.5, fy=0.5)

        cv2.imshow("YOLOP", result_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    video_capture.release()
    cv2.destroyAllWindows()
