# YOLOv5를 Fine-tuning하여 Custom Model 만들기
Glenn Jocher가 만든 사전 학습된 YOLOv5를 **Roboflow에서 가져온 데이터셋**을 통해 **Fine-tuning**하여 몇 가지 **Custom model**을 제작했다.
1. 포트홀 감지 모델 (Pothole)
2. 산불 감지 모델 (Wildfire)
3. 공사장 헬멧 감지 모델 (HardHatWorkers)
4. 주차장 빈 자리 감지 모델 (PKlot)
5. 손동작 인식 모델 (Auto_cam)
***

### 사용 언어 및 기술
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
***

## Custom Model 제작 방법
**0. 환경설정**
  ```
  import torch
  import yaml
  import glob
  from glob import glob
  import random
  from IPython.display import Image, display
  from IPython.core.magic import register_line_cell_magic
  ```   
**1. YOLOv5 git clone**
  ```
  !git clone https://github.com/ultralytics/yolov5
  ```
**2. Dataset 다운로드 받기**
  ```
  %mkdir /content/yolov5/데이터셋을 담을 폴더
  %cd /content/yolov5/데이터셋을 담은 폴더
  !curl -L "roboflow에서 가져온 데이터셋 다운로드 링크" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
  ```
**3. glob 메서드를 이용해서 데이터 전처리(학습/테스트/검증 데이터로 나누기)**
  ```
  train_img_list = glob('/content/yolov5/데이터셋이 담긴 폴더/train/images/*.jpg')
  test_img_list = glob('/content/yolov5/데이터셋이 담긴 폴더/test/images/*.jpg')
  valid_img_list = glob('/content/yolov5/데이터셋이 담긴 폴더/valid/images/*.jpg')
  ```
**4. YOLOv5 Fine-tuning에 사용될 data.yaml으로 지정**
  ```
  %%writetemplate /content/yolov5/wildfire/data.yaml
  
  train: ./wildfire/train/images
  test: ./wildfire/test/images
  val: ./wildfire/valid/images
  
  nc: 레이블 개수
  names: ['레이블명', '레이블명']
  ```
**5. 기존의 YOLOv5 모델 중 하나를 선택하여 Custom model로 변경(레이블 개수 맞춰주기)**
  ```
  %%writetemplate /content/yolov5/models/custom_yolov5s.yaml #YOLOv5 small 모델을 선택하여 커스터마이징 했다. 
  
  # Parameters
  nc: {num_classes}  # number of classes
  depth_multiple: 0.33  # model depth multiple
  width_multiple: 0.50  # layer channel multiple
  anchors:
    - [10,13, 16,30, 33,23]  # P3/8
    - [30,61, 62,45, 59,119]  # P4/16
    - [116,90, 156,198, 373,326]  # P5/32
  
  # YOLOv5 v6.0 backbone
  backbone:
    # [from, number, module, args]
    [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
     [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
     [-1, 3, C3, [128]],
     [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
     [-1, 6, C3, [256]],
     [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
     [-1, 9, C3, [512]],
     [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
     [-1, 3, C3, [1024]],
     [-1, 1, SPPF, [1024, 5]],  # 9
    ]
  
  # YOLOv5 v6.0 head
  head:
    [[-1, 1, Conv, [512, 1, 1]],
     [-1, 1, nn.Upsample, [None, 2, 'nearest']],
     [[-1, 6], 1, Concat, [1]],  # cat backbone P4
     [-1, 3, C3, [512, False]],  # 13
  
     [-1, 1, Conv, [256, 1, 1]],
     [-1, 1, nn.Upsample, [None, 2, 'nearest']],
     [[-1, 4], 1, Concat, [1]],  # cat backbone P3
     [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
  
     [-1, 1, Conv, [256, 3, 2]],
     [[-1, 14], 1, Concat, [1]],  # cat head P4
     [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
  
     [-1, 1, Conv, [512, 3, 2]],
     [[-1, 10], 1, Concat, [1]],  # cat head P5
     [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
  
     [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
    ]
  ```
**6. Fine-tuning(batch size, epochs 조정, 조금 더 빠른 학습을 위해 disk 사용)**
  ```
  !python /content/yolov5/train.py --img 640 --batch 16 --epochs 50 --data /content/yolov5/hardhat/data.yaml --cfg /content/yolov5/models/custom_yolov5s.yaml --weights '' --name hardhat_result --cache disk
  ```
**7. 학습 완료, 결과물 도출**
  ```
  Image(filename='/content/yolov5/runs/train/결과물 파일 경로/train_batch0.jpg', width=1500)
  ```
***
## Fine-tuning 시 문제 해결
YOLOv5를 Fine-tuning하던 과정 중 몇 가지 에러가 발생했다. <br>
**1. train할 때, label을 찾지 못하는 문제** <br>
다음과 같이, label을 찾지 못하는 문제가 발생한다. <br>
  ```
  train: No labels found in /content/yolov5/wildfire/train/labels.cache, can not start training.
  ```
학습 코드를 처음 돌릴 때, 학습을 위한 라이브러리를 다운로드 및 업데이트하느라 label cache가 생긴다. <br>
다운로드 및 업데이트를 완료한 후에 학습을 재시도 해야하는데, 이때 처음에 생성된 label cache를 삭제하고 재시도해야 정상적으로 학습이 시작된다. <br>
  ```
  %rm /content/yolov5/dataset dir/train/labels.cache
  ```

**2. Label 개수 및 순서 지정 문제** <br>
  ```
  %%writetemplate /content/yolov5/hardhat/data.yaml
  
  train: ./hardhat/train/images
  test: ./hardhat/test/images
  val : ./hardhat/val/images
  
  nc: 3
  names: ['head', 'helmet', 'person']
  ```
여기서 레이블과 레이블명을 지정해주는데, <br>
1) 데이터셋에 Validation 데이터가 없을 수도 있고 <br>
2) 레이블의 순서가 뒤바꿀 수도 있다(학습까지 다 시켰는데, 결과물이 뒤바뀌어 나오는 어이없는 경험을 할 수 있음) <br>
데이터셋 설명을 잘 읽어서 데이터가 없다면 " " 공백으로 두고, label의 순서가 뒤바뀌지 않도록 주의해야 한다!
