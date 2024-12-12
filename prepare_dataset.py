# 필요한 라이브러리 import
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split  # train_test_split 함수 추가
from tensorflow.keras.utils import to_categorical  # 원-핫 인코딩용
import numpy as np

def prepare_dataset(base_folder):
    """
    최종 폴더 내 모든 하위 폴더에서 이미지를 읽어 학습 데이터를 준비합니다.
    """
    X = []  # 입력 데이터 (랜드마크)
    y = []  # 라벨
    gestures = os.listdir(base_folder)  # "최종" 폴더의 하위 폴더 이름 (동작 이름)
    
    for label, gesture_name in enumerate(gestures):  # 각 제스처별로 라벨 할당
        folder_path = os.path.join(base_folder, gesture_name)
        if os.path.isdir(folder_path):  # 폴더인지 확인
            print(f"Processing gesture: {gesture_name}, Label: {label}")
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                landmarks = extract_landmarks(image_path)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label)  # 각 동작에 대해 라벨 추가
    
    return np.array(X), np.array(y), gestures

base_folder = os.getenv('FINAL_DATASET_FOLDER', r'D:\default\path')
X, y, gestures = prepare_dataset(base_folder)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 원-핫 인코딩
num_classes = len(gestures)  # 동작 클래스 수
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"입력 데이터 크기: {X_train.shape}")
print(f"라벨 데이터 크기: {y_train.shape}")
print(f"제스처 클래스: {gestures}")
