#!/bin/bash

python emotion_recognizer.py 2>&1 |egrep -v "WARNING|FutureWarning|UserWarning|dtype"\
 |egrep -v "warnings.warn|DeprecationWarning"
