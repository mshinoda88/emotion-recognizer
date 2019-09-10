# emotion-recognizer
Predict the emotional state of people in an image from their facial expressions


## Requirements

```bash
pip install -r requirements.txt
```

## Installation
```bash
cd conf/emotion-recognizer/ 
wget https://storage.cloud.google.com/public_teamml/cnn/emotion-recognizer/_mini_XCEPTION.102-0.66.hdf5

cd conf/face-detect
wget https://storage.cloud.google.com/public_teamml/cnn/face-detect/yolov3-wider_16000.weights
wget https://storage.cloud.google.com/public_teamml/cnn/face-detect/yolov3-face.cfg
```


## Demo
- Face Detection

```bash
cd src
python face_detector.py
```

- Face Emotion Recognizer

```bash
cd src
python emotion_recognizer.py
```


