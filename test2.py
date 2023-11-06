import cv2
import numpy as np

# 이미지를 읽어옵니다.
image = cv2.imread('coins.png')

# 이미지를 그레이스케일로 변환합니다.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러 처리를 통해 이미지를 노이즈를 줄입니다.
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Canny 에지 검출을 수행합니다.
edged = cv2.Canny(blurred, 30, 150)

# 컨투어(윤곽선)를 찾습니다.
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 동전의 갯수를 초기화합니다.
coin_count = 0

# 각 컨투어를 순회하며 동전을 판별합니다.
for contour in contours:
    # 컨투어를 근사화합니다.
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 근사화된 컨투어의 꼭짓점 갯수를 확인하여 동전인지 판별합니다.
    if len(approx) >= 8:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        coin_count += 1
        # 동전 위에 숫자를 작게 출력합니다.
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(image, str(coin_count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 결과를 출력합니다.
cv2.putText(image, f'Coins: {coin_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Coins', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
