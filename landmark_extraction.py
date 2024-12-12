import mediapipe as mp
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random  # 랜덤 선택을 위한 모듈

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_and_draw(image_path):
    """
    이미지에서 Mediapipe 손 랜드마크를 추출하고, 시각화한 결과를 반환합니다.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return annotated_image, np.array(landmarks).flatten()
    return image, None  # 손이 감지되지 않으면 원본 이미지 반환

# 최종 폴더 경로 설정
base_folder = os.getenv('FINAL_DATASET_FOLDER', r'D:\default\path')

# 각 폴더(동작)에서 랜덤 2장씩 선택하여 시각화
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path):  # 폴더인지 확인
        print(f"폴더: {folder_name}")
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]  # 이미지 파일만 선택
        random_images = random.sample(image_files, min(2, len(image_files)))  # 랜덤으로 최대 2장 선택
        
        for image_name in random_images:
            image_path = os.path.join(folder_path, image_name)
            annotated_image, landmarks = extract_landmarks_and_draw(image_path)
            
            # 결과 출력
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            if landmarks is not None:
                plt.title(f"{folder_name}: {landmarks[:5]}...")  # 폴더 이름과 랜드마크 일부 출력
            else:
                plt.title(f"{folder_name}: No hand detected")
            plt.show()
