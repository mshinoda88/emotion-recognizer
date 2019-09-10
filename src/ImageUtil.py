from PIL import Image
import cv2

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

# しきい値 コンフィデンス
CONF_THRESHOLD = 0.5
# しきい値 Non Maximum Suppression
NMS_THRESHOLD = 0.4

IMG_WIDTH = 416
IMG_HEIGHT = 416

# BGR(OpenCV形式) で表した色
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# バウンディングボックスを表現するクラス
class BBox:
    def __init__(self, left, right, top, bottom):
        self.left   = left
        self.right  = right
        self.top    = top
        self.bottom = bottom
        self.width  = self.right - self.left
        self.height = self.bottom - self.top 

    def setSize(self, left, top, width, height):
        self.width  = width
        self.height = height
        self.left   = left
        self.top    = top
        self.right  = self.left + self.width
        self.bottom = self.top  + self.height

# 画像処理ユーティリティクラス
class ImageUtil:
    # OpenCV format からPIL format にイメージデータ変換
    #
    # input
    #   image 変換対象のイメージデータ
    # output
    #   変換後のイメージデータ
    @classmethod
    def opencv2PIL(self, image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    # イメージデータの白色化
    #
    # input
    #   image 変換対象のイメージデータ
    # output
    #   変換後のイメージデータ
    @classmethod
    def bgr2gray(self, image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    @classmethod
    def adjust_rect(self, out_box, img_width, img_height):
        #img_width = 1280
        #img_height = 720 -> detected 704
        top,left,bottom,right = out_box
        top = max(0.0,top)
        left = max(0.0,left)
        width = right - left
        height = bottom - top
        new_width = max(width,height)

        #left = max(0,left -8)
        bottom = min(top + new_width,img_height)
        right = min(left+new_width,img_width)
        return (int(top),int(left),int(bottom),int(right))

    # Draw the predicted bounding box
    @classmethod
    def draw_predict(self, frame, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

        text = '{:.2f}'.format(conf)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(top, label_size[1])
        cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)

    @classmethod
    def refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

        right = right + margin
        return left, top, right, bottom


