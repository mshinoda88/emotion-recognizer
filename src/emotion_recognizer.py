from keras.models import load_model
import numpy as np
import os,sys
import cv2
from face_detector import FaceDetector
from ImageUtil import BBox
import time

# 画像から表情を検知する機能を提供するクラス
#
#
class EmotionRecognizer:

    # コンストラクタ
    #
    # input
    #   conf_dir   設定ファイル置き場のディレクトリ
    #   img_size   画像サイズ
    #   output_dir 出力フォルダ
    #
    def __init__(self, conf_dir, img_size, output_dir):
        self.output_dir = output_dir
        self.recognizer_path = conf_dir + "/_mini_XCEPTION.102-0.66.hdf5"
        self.recognizer = load_model(self.recognizer_path)
        self.emotions = ['angry','disgust','scared','happy','sad','surprised',
                         'neutral']

    # 画像から表情を数値で返します。
    #
    # input
    #   image_path 入力画像のパス
    #   boxes      画像中の顔の位置情報。
    #              バウンディングボックスBBox [left,top,width,height] の配列 
    # output
    #   ret_emotions  表情を表す数値
    def classify_emotion(self, img_path, boxes):
        # ファイルの存在チェック
        if os.path.isfile(img_path) == False:
            print("File Not Found:",img_path)
            return []

        if len(boxes)==0:
            return []
        img = cv2.imread(img_path)

        # グレースケールに変換した画像データ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 顔のバウンディングボックスのサイズ調整
        boxes = self._adjust_boxes(boxes)

        # グレースケール画像とバウンディングボックスから入力テンソル作成
        input_tensor = self._generate_input_tensor(gray, boxes)

        # 入力テンソルから感情推定ベクトルを取得
        ret_emotions = self._tensor2emotion(input_tensor)
        #print("face box size:",len(boxes),len(ret_emotions))
        return ret_emotions


    # 顔を検知した画像に対してキャプション設定を行います
    #
    # input
    #   image_path 入力画像のパス
    #   boxes      画像中の顔の位置情報。
    #              バウンディングボックスBBox [left,top,width,height] の配列 
    #   emotions   感情推定値のベクトル配列
    # output
    #   なし
    def put_caption(self, img_path, boxes, emotions):
        img = cv2.imread(img_path)
        if len(boxes) == 0:
            try:
                cv2.putText(img,'NO FACE',(640,360),
                  cv2.FONT_HERSHEY_SIMPLEX,3.0,(0,255,0),lineType=cv2.LINE_AA)    
            except Exception as e:
                print("putText got exception:")
                print(e)

        for emotion,box in zip(emotions, boxes):
            top,left,bottom,right,width,height = box.top,box.left,box.bottom,box.right,box.width,box.height
            cv2.rectangle(img,(left,top),(right,bottom),(255,0,0),3)

            # 感情ベクトルから文字列へ変換
            e_status = self.get_emotion_label(emotion)
            #print("emotion",e_status,"left=",left,"top=",top)
            try:
                # 画像に感情キャプションを追加
                cv2.putText(img,e_status,(left,top),
                  cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),lineType=cv2.LINE_AA)
            except Exception as e:
                print("putText got exception:")
                print(e)

        img_dir,img_name = os.path.split(img_path)
        save_path = os.path.join(self.output_dir, img_name)
        cv2.imwrite(save_path, img)


    # 感情ベクトルから文字列へ変換
    #
    # input
    #   e_vectors 感情ベクトル
    # output
    #   感情を表す文字列
    @classmethod
    def get_emotion_label(self, e_vectors):
        # TODO
        #e_total,e_max = sum(e_vectors),max(e_vectors)
        #emotion= int(np.floor(e_max * 100.0 / e_total))
        #e_status = "happy"
        #return e_status
        predicted_class = np.argmax(e_vectors)
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}
        return labels[predicted_class]

    # バウンディングボックスのサイズ微調整
    #
    # input
    #   boxes 画像中の顔の位置情報。BBoxの配列
    # output
    #   微調整後のバウンディングボックス配列
    def _adjust_boxes(self, boxes):
        ret_boxes = []
        for box in boxes:
            box_new = self._adjust_rect(box, 1280, 720)
            ret_boxes.append(box_new)

        return ret_boxes

    # 感情推定のための入力テンソルの生成 
    #
    # input
    #   gray  グレースケールに変換した画像データ
    #   boxes 画像中の顔の位置情報。
    #         バウンディングボックス [left,top,width,height] の配列 
    def _generate_input_tensor(self, gray, boxes):
        preprocessed_input = []
        for box in boxes:
            roi = gray[box.top:box.bottom,box.left:box.right]
            #print("roi",roi)
            try:
                roi = cv2.resize(roi,(64,64))
            except Exception as e:
                print("Image resize got exception:")
                print(e)
                continue
            roi = roi/255.0
            preprocessed_input.append(roi)

        #print("size preprocessed_input",len(preprocessed_input))
        input_tensor = np.array(preprocessed_input)
        return np.expand_dims(input_tensor,axis=-1)

    # 顔のバウンディングボックスのサイズ調整
    #
    # input
    #   box 画像中の顔の位置情報。BBoxオブジェクト
    #   img_width  画像のサイズ幅
    #   img_height 画像のサイズ高さ
    # output
    #   微調整後のバウンディングボックス
    # 
    def _adjust_rect(self, box, img_width, img_height):
        #img_width = 1280
        #img_height = 720 -> detected 704
        top,left,width,height = box.top, box.left, box.width, box.height
        top = max(0.0,top)
        left = max(0.0,left)
        width = box.width
        height = box.height
        new_width = max(width,height)
    
        #left = max(0,left -8)
        bottom = min(top + new_width,img_height)
        right = min(left+new_width,img_width)
        return BBox(int(left),int(right),int(top),int(bottom))

    # self.emotions で指定した7次元のベクトルを推定値として返す
    #
    # input
    #   input_tensor 入力テンソル
    # output
    #   推定値ベクトル
    def _tensor2emotion(self, input_tensor):
        ret_emotions = []
        #print("_tensor2emotion input_tensor size",len(input_tensor))
        preds = self.recognizer.predict(input_tensor)
        #print("_tensor2emotion preds size",len(preds))
        return preds

if __name__ == "__main__":
    output_dir="../data/outputs"
    face_detect_conf_dir="../conf/face-detect"
    erecognizer_conf_dir="../conf/emotion-recognizer"
    image_path="../data/inputs/sand01.jpg"
    print("input image path",image_path)
    face_detector = FaceDetector(face_detect_conf_dir)
    box_faces = face_detector.detect_face(image_path)
    print(box_faces)
    if len(box_faces)>0:
        emotion_recognizer = EmotionRecognizer(erecognizer_conf_dir, 416, output_dir)
        ret_emotions = emotion_recognizer.classify_emotion(image_path, box_faces)
        print("ret_emotions",ret_emotions)
        emotion_recognizer.put_caption(image_path, box_faces, ret_emotions)


