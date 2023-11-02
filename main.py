import cv2
import mediapipe as mp
 
#mp.solutions.drawing_utils用于绘制
mp_drawing = mp.solutions.drawing_utils
 
#参数：1、颜色，2、线条粗细，3、点的半径
DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 3 , 3)
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 3, 3)
 
#mp.solutions.hands，是人的手
mp_hands = mp.solutions.hands
 
#参数：1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
hands_mode = mp_hands.Hands(max_num_hands=2)
 
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while cap.isOpened():
    success, image = cap.read()
    #image=cv2.imread("input.png")
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

 
    # 处理RGB图像
    results = hands_mode.process(image1)

    # 绘制
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
 
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("output.png",image)
        break
 
hands_mode.close()
cv2.destroyAllWindows()
cap.release()
