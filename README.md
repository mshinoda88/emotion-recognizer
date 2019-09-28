# emotion-recognizer
Predict the emotional state of people in an image from their facial expressions

## Requirements

```bash
pip install -r requirements.txt
```

## Installation
```bash
cd conf/emotion-recognizer/ 
bash download.sh

cd ../face-detect
bash download.sh
```


## Demo
- Face Detection

```bash
cd src
python face_detector.py
```

元画像<br>
<img src="https://d-dtc.backlog.com/git/CAMP_TASK/emotion-detector/blob/master/data/inputs/pict01.png" size=50%>

処理後の画像<br>
<img src="https://d-dtc.backlog.com/git/CAMP_TASK/emotion-detector/blob/master/data/outputs/pict01_yoloface.jpg" size=50%>

- Face Emotion Recognizer

```bash
cd src
python emotion_recognizer.py
```


