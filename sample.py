import argparse

import cv2

from yolop_onnx import YolopONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video',
        type=str,
        default='video/sample.mp4',
    )

    parser.add_argument(
        '--model',
        type=str,
        default='weights/yolop-640-640.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='640,640',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    video_path = args.video

    model_path = args.model
    input_size = args.input_size
    score_th = args.score_th
    nms_th = args.nms_th

    input_shape = [int(i) for i in input_size.split(',')]

    # モデルロード
    yolop = YolopONNX(
        model_path=model_path,
        input_shape=input_shape,
        score_th=score_th,
        nms_th=nms_th,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # ビデオ読み込み
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # 推論
        bboxes, da_seg_mask, ll_seg_mask = yolop.inference(frame)

        # デバッグ描画
        result_image = yolop.draw(
            frame,
            bboxes,
            da_seg_mask,
            ll_seg_mask,
        )

        cv2.imshow("YOLOP", result_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    video_capture.release()
    cv2.destroyAllWindows()
