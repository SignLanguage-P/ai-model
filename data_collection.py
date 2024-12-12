import cv2
import os
import time  # 타이머를 위한 모듈 추가

# 데이터 저장 경로 설정
base_dir = os.path.join(os.getcwd(), 'dataset', 'final')  # 저장할 기본 경로
gesture_name = '회사'  # 저장할 제스처 이름
save_dir = os.path.join(base_dir, gesture_name)  # 제스처 이름에 따른 폴더 경로 생성
os.makedirs(save_dir, exist_ok=True)  # 폴더 생성

cap = cv2.VideoCapture(0)  # 카메라 시작

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 기존 저장된 이미지 개수 확인하여 고유 카운터 설정
existing_images = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
if existing_images:
    max_index = max(int(os.path.splitext(f)[0]) for f in existing_images)
    image_count = max_index + 1  # 기존 파일의 가장 높은 번호에서 시작
else:
    image_count = 0  # 기존 파일이 없으면 0부터 시작

print("Press 'q' to stop capturing images.")  # 종료 안내 메시지

start_time = time.time()  # 시작 시간 기록

while True:
    ret, frame = cap.read()  # 카메라에서 한 프레임 읽기
    if not ret:
        break  # 카메라 오류 시 종료
    
    # 현재 시간 계산
    elapsed_time = time.time() - start_time
    
    # 40초가 지나면 자동 종료
    if elapsed_time > 40:
        print("40초가 경과하여 데이터 수집을 종료합니다.")
        break

    # 화면에 카메라 영상 표시
    cv2.imshow('frame', frame)
    
    # 1초에 약 10장의 이미지를 저장 (프레임 간격 조절)
    if image_count % 3 == 0:  # 3프레임마다 1번 저장
        file_path = os.path.join(save_dir, f'{image_count}.jpg')  # 이미지 파일 경로
        cv2.imwrite(file_path, frame)  # 이미지 저장
        print(f"Saved: {file_path}")  # 저장된 파일 경로 출력
    
    image_count += 1  # 이미지 개수 증가
    
    # 'q'를 누르면 조기 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("사용자 요청으로 종료합니다.")
        break

cap.release()  # 카메라 릴리즈
cv2.destroyAllWindows()  # 모든 창 닫기
