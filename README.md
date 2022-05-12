# ProjectAim-ApexLegends
An efficient Apex Legends Aimbot based on YoloV5 real-time target detection

## Credit to
YOLO V5 https://github.com/ultralytics/yolov5

Apex Dataset https://github.com/goldjee/AL-YOLO-dataset

## What you need

A CUDA Compatible GPU, Graphics cards with lower performance than RTX3060 may not work perfect

## How to use
Install all on requirements.txt by

    pip install -r requirements.txt
    
    # CUDA 11.3
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge


And then have a happy test

    # run this in NON-CHINESE folder and using a python 3.7+ (Recommend Python 3.8 anaconda)
    # May need Admin Powershell
    python main.py 

## To-Do List
  Test to determine if it can be used in normal games
  
  Increase accuracy by improving screenshots and mouse efficiency

  Replacing with a smaller model to reduce hardware requirements

  A more user-friendly interface

  Cut out the extra code

  


## Current Situation
Functional but NOT tested in real game.

Performance accepable on using video

<video id="video" controls="" preload="none" poster="game.jpg">
      <source id="mp4" src="testvideo.mp4" type="video/mp4">
      </video>
