# 수어 인식 및 번역 모델



## 📍 프로젝트 설명
Mediapipe와 TensorFlow를 활용하여 수어 데이터를 인식하고 학습하는 모델을 개발합니다.

---


## 🙌 데이터셋
- 한국 수어 데이터셋 부족 => 수어 사전 및 유튜브 영상을 통해 수어 따라하며 직접 촬영
- opencv 활용하여 실시간 영상 캡처 방식으로 데이터 수집
- 1초당 약 10장의 프레임 저장하여 각각 다른 배경과 조명에서 40초씩 세 번 진행하여 제스처별 720장 이상 확보
- 총 32개의 단어/문장 (p_signlanguage_labels.json 참고)

---

## 🛠 기술 스택
![image](https://github.com/user-attachments/assets/96be0767-f99a-4bff-904c-9e7a5c005267)
![image](https://github.com/user-attachments/assets/9424cd4b-8837-427e-beff-0e3dddbe0559)
![image](https://github.com/user-attachments/assets/5e811f26-7ce3-42c0-8ee6-eae44e741ffd)
![image](https://github.com/user-attachments/assets/1f19acb6-ad67-4d4c-8893-c8c34659d0e4)
![image](https://github.com/user-attachments/assets/808f8e03-7969-4da7-a84b-4f44ddce4d1d)

---

## 💡 핵심 기능 

### 1. 데이터 수집 (data_collection.py)
- **기능**:
  - OpenCV를 사용하여 웹캠에서 데이터를 수집하고 제스처별로 이미지를 저장합니다.

- **핵심 내용**:
  - 카메라를 통해 데이터를 수집하며, 초당 약 10개의 이미지를 저장합니다.
  - 40초 동안 데이터를 수집하거나 사용자가 수동으로 종료할 수 있습니다.
  - 저장된 데이터는 제스처별로 폴더에 구분됩니다.




### 2. 랜드마크 추출 (landmark_extraction.py)
- **기능**:
  - Mediapipe를 활용하여 손의 21개 랜드마크를 추출합니다.

- **핵심 내용**:
  - 이미지를 Mediapipe 모델에 입력하여 랜드마크를 추출합니다.
  - 추출된 랜드마크는 (x, y, z) 좌표 형태로 평탄화(flatten)되어 반환됩니다.
  - 손이 감지되지 않는 경우 None을 반환합니다.




### 3. 모델 학습 (train_model.py)
- **기능**:
  - LSTM을 사용하여 수어 인식을 위한 모델을 학습합니다.

- **핵심 내용**:
  - 데이터는 시퀀스 형태로 변환되어 모델에 입력됩니다.
  - 모델은 2개의 LSTM 레이어와 BatchNormalization, Dropout을 사용합니다.
  - EarlyStopping과 ReduceLROnPlateau 콜백을 사용하여 효율적인 학습을 지원합니다.
  - 학습 완료 후 모델은 .h5 파일로 저장니다.
 



### 4. 필요 패키지 (requirements.txt)
주요 패키지:
- numpy==1.23.0
- opencv-python==4.10.0.84
- mediapipe==0.10.14
- matplotlib==3.7.1
- tensorflow==2.18.0
- scikit-learn==1.2.2
- pandas==2.2.2




## 📂 프로젝트 구조
/ai-model 

├── README.md # 프로젝트 설명 파일 

├── requirements.txt # Python 패키지 목록 파일

├── data_collection.py # 데이터 수집을 위한 코드 

├── landmark_extraction.py # Mediapipe를 활용한 랜드마크 추출 코드 

├── train_model.py # LSTM 모델 학습 코드 

└── model/ # 학습된 모델 또는 기타 관련 파일 저장 디렉토리

---
