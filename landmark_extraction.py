import mediapipe as mp
import cv2
import numpy as np

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image_path):
    """
    이미지에서 Mediapipe 손 랜드마크를 추출합니다.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # x, y, z 좌표
            return np.array(landmarks).flatten()  # 평탄화하여 반환
    return None  # 손이 감지되지 않으면 None 반환
