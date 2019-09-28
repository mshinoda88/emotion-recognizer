# ライブラリのロード
import numpy as np
import os,sys
from ImageUtil import *


# 画像から顔の位置情報を検知する機能を提供するクラス
class FaceDetector:
    # model構成、重みをdarknetが取得
    #net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # コンストラクタ
    #   特定画像に依存する情報はここで処理しない
    #   モデルの情報のみロードする
    #
    # input
    #   conf_dir   設定ファイルのディレクトリ
    def __init__(self, conf_dir):
        print("conf_dir",conf_dir)
        self.model_cfg = conf_dir + '/yolov3-face.cfg'
        self.model_weights = conf_dir + '/yolov3-wider_16000.weights'
        # ファイルの存在チェック
        if os.path.isfile(self.model_cfg) == False:
            print("File Not Found:",self.model_cfg)
            sys.exit()
        if os.path.isfile(self.model_weights) == False:
            print("File Not Found:",self.model_weights)
            sys.exit()
        #print("model_cfg:",self.model_cfg)
        #print("model_weights:",self.model_weights)
        self.net = cv2.dnn.readNetFromDarknet(self.model_cfg, self.model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # ディレクトリの存在チェック
        if os.path.isdir(conf_dir) == False:
            print("Directory Not Found:",conf_dir)
            sys.exit()


    # 画像から顔の位置情報を返します。
    #
    # input
    #   image_path 入力画像のパス
    # output
    #   画像中の顔の位置情報。
    #   バウンディングボックス [left,top,width,height] の配列        
    #
    # Create a 4D blob from a frame.
    #     IMG_WIDTH:416 utils.pyに定義
    #     IMG_HEIGHT :416　utils.pyに定義
    #     dnnのモジュールとして使えるように型変換している
    #     blob.shape:(1, 3, 416, 416)
    def detect_face(self, image_path):
        # ファイルの存在チェック
        if os.path.isfile(image_path) == False:
            print("File Not Found:",image_path)
            return []

        cap = cv2.VideoCapture(image_path)
        # has_frame:frameの有無
        # frame:読み込んだ画像 array形式
        self.has_frame, self.frame = cap.read()
        self.image_width = self.frame.shape[0]
        self.image_height = self.frame.shape[1]
        self.image_channel = self.frame.shape[2]

        blob = cv2.dnn.blobFromImage(self.frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        # get_outputs_names:utils.pyに定義
        # 　input: ['yolo_82', 'yolo_94', 'yolo_106']
        # outs: バウンディングボックス全候補と判定結果
        outs = self.net.forward(self._get_outputs_names(self.net))
        
        # Remove the bounding boxes with low confidence
        faces_bbox_ary = self._filter_bbox(self.frame, outs)

        # バウンディングボックス [left,top,width,height] の配列        
        return faces_bbox_ary

    # 検出位置にキャプションを付与する
    #
    # input
    #   image_path 入力画像のパス
    #   faces_bbox_ary 顔のバウンディングボックス配列
    #   output_path    出力先のファイルパス
    # output
    #   なし
    def put_caption(self, image_path, faces_bbox_ary, output_path):
        # ファイルの存在チェック
        if os.path.isfile(image_path) == False:
            print("File Not Found:",image_path)
            return

        # ディレクトリの存在チェック
        dir_output = os.path.dirname(output_path)
        if os.path.isdir(dir_output) == False:
            print("Directory Not Found:",output_path)
            return

        frame = self.frame
        info = [('number of faces detected','{}'.format(len(faces_bbox_ary)))]
        for (i,(txt,val)) in enumerate(info):
            text = '{}:{}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 100),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            cv2.imwrite(output_path, frame.astype(np.uint8))

    # 入力画像と出力ディレクトリから出力ファイルのパスを生成します。
    #
    # input
    #   image_path 入力画像のパス
    #   output_dir 出力ディレクト
    # output
    #   出力ファイルのパスを生成して返します
    @classmethod
    def generate_outpath(self, image_path, output_dir):
        # ファイルの存在チェック
        if os.path.isfile(image_path) == False:
            print("File Not Found:",image_path)
            return

        # ディレクトリの存在チェック
        if os.path.isdir(output_dir) == False:
            print("Directory Not Found:",output_dir)

        output_file = image_path[:-4].rsplit('/')[-1] + '_yoloface.jpg'
        output_path = os.path.join(output_dir, output_file)
        return output_path

    # Get the names of the output layers
    def _get_outputs_names(self, net):
        # Get the names of all the layers in the network
        # conv,bn,reluとかの流れを配列で所持
        layers_names = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected
        # outputs yoloの層のみ抽出：yolo_82,yolo_94, yolo_106
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 候補となるバウンディングボックスからしきい値以上となるものを抽出する
    #
    # input
    #   frame : 読み込んだ画像(682, 1024, 3)
    #   outs  : バウンディングボックスとその判定結果
    #   threshold_conf : コンフィデンスのしきい値、初期値：CONF_THRESHOLD = 0.5
    #   threshold_nms  : non maximum suppression のしきい値、初期値：NMS_THRESHOLD = 0.4
    # output
    #    バウンディングボックス [left,top,width,height] の配列
    def _filter_bbox(self, frame, outs,
                       threshold_conf=CONF_THRESHOLD, threshold_nms=NMS_THRESHOLD):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs: # len(outs) = 3　   "3"はなんの数値だ？
            for detection in out:
            # detectionは3042個ある。yolo１層目で検出されたboundingboxの結果？
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold_conf:
                    # ボックス中心のX座標(割合で表記)
                    center_x = int(detection[0] * frame_width)
                    # ボックス中心のY座標(割合で表記)
                    center_y = int(detection[1] * frame_height)
                    # ボックスの幅(割合で表記)
                    width = int(detection[2] * frame_width)
                    # ボックスの高さ(割合で表記)
                    height = int(detection[3] * frame_height)
                    # ボックスの左端X座標
                    left = int(center_x - width / 2)
                    # ボックスの上端Y座標
                    top = int(center_y - height / 2)
                    # 判定scoreを格納
                    confidences.append(float(confidence))
                    # ボックス位置を格納
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        # 重なってるバウンディングボックスの高いコンフィデンスのみ残す
        indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_conf,
                                   threshold_nms)
        # 重なっていたバウンディングボックスの高い値のみ検出した配列indices
        # を使って、boxesの中から必要な者のみに限定
        for i in indices:
            i = i[0]
            box = boxes[i]
            left,top,width,height = box[0],box[1],box[2],box[3]
            #final_boxes.append(box)
            left, top, right, bottom = ImageUtil.refined_box(left, top, width, height)
            box_obj = BBox(left, right, top, bottom)
            final_boxes.append(box_obj)
            # draw_predict(frame, confidences[i], left, top, left + width,
            #              top + height)
            ImageUtil.draw_predict(frame, confidences[i], left, top, right, bottom)
        return final_boxes

if __name__ == "__main__":
    output_dir="../data/outputs"
    conf_dir="../conf/face-detect"
    image_path="../data/inputs/meeting_11_304.jpg"
    face_detector = FaceDetector(conf_dir)
    faces = face_detector.detect_face(image_path)
    output_path = FaceDetector.generate_outpath(image_path, output_dir)
    print("output path:",output_path)
    face_detector.put_caption(image_path, faces, output_path)

