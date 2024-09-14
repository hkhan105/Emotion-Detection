import cv2
import mediapipe as mp
import os


img_bank = cv2.imread("C:/Users/tajam/Downloads/Image dataset/surprise")
previous_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=3, circle_radius=3)



def face(blendshape):
    print(blendshape[25], blendshape[21], blendshape[22], blendshape[3], blendshape[4], blendshape[5], blendshape[6])
    if blendshape[25] > 0.025 and blendshape[21] > 0.006 and blendshape[22] > 0. and blendshape[3] > 0.5 and blendshape[4] > 0.5 and blendshape[5] > 0.5 and blendshape[6] > 0.5:
        return "Surprise"
    #elif blendshape[36] < -0.3 and blendshape[37] < -0.3 and blendshape[1] < -0.35 and blendshape[2] < -0.35 and blendshape[50] > 0.1 and blendshape[51] > 0.1:
        #return "Anger"
    #elif blendshape[30] > 0.8 and blendshape[31] > 0.8 and blendshape[7] < 0.1 and blendshape[8] < 0.2 and blendshape[3] > 0.1:
       # return "Sadness"
    #elif blendshape[44] < -0.3 and blendshape[45] < -0.3 and blendshape[7] < -0.3 and blendshape[8] < -0.3:
        #return "Joy"
   # elif blendshape[3] > 0.5 and blendshape[21] > 0.45 and blendshape[22] > 0.45 and blendshape[25] > 0.4 and blendshape[7] <= 0.1 and blendshape[8] <= 0.1:
        #return "Fear"
    #elif blendshape[50] > 0.4 and blendshape[51] > 0.4 and blendshape[36] < 0 and blendshape[37] < 0 and blendshape[1] < -0.1 and blendshape[2] < -0.1:
        #return "Disgust"
    #elif blendshape[44] < -0.2 and blendshape[45] < -0.2 and blendshape[7] < -0.2 and blendshape[8] < -0.2 and blendshape[50] <= -0.1 and blendshape[51] <= -0.1:
        #return "Contempt"
    else:
        return "Neutral"

mood_colors = {
    "Joy": (215, 245, 66),
    "Surprise": (66, 176, 245),
    "Sadness": (84, 24, 14),
    "Anger": (40, 14, 84),
    "Fear": (84, 14, 77),
    "Disgust": (20, 3, 19),
    "Contempt": (230, 225, 229),
    "Neutral": (255,255,255)
}

while True:
    for photo in os.listdir("C:/Users/tajam/Downloads/Image dataset/surprise"):
        img_path = os.path.join("C:/Users/tajam/Downloads/Image dataset/surprise", photo)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, draw_spec, draw_spec)


                face_structure = {
                    1: face_landmarks.landmark[1].z,  # browDownLe
                    2: face_landmarks.landmark[2].z,  # browDownRight
                    3: face_landmarks.landmark[3].z,  # browInnerUp
                    4: face_landmarks.landmark[4].z,
                    5: face_landmarks.landmark[5].z,
                    6: face_landmarks.landmark[6].z,
                    7: face_landmarks.landmark[7].z,  # cheekSquintLe
                    8: face_landmarks.landmark[8].z,  # cheekSquintRight
                    9: face_landmarks.landmark[9].z,  # eyeBlinkLe
                    10: face_landmarks.landmark[10].z,  # eyeBlinkRight
                    21: face_landmarks.landmark[21].z,  # eyeWideLe
                    22: face_landmarks.landmark[22].z,  # eyeWideRight
                    25: face_landmarks.landmark[25].z,  # jawOpen
                    30: face_landmarks.landmark[30].z,  # mouthFrownLe
                    31: face_landmarks.landmark[31].z,  # mouthFrownRight
                    36: face_landmarks.landmark[36].z,  # mouthPressLe
                    37: face_landmarks.landmark[37].z,  # mouthPressRight
                    44: face_landmarks.landmark[44].z,  # mouthSmileLe
                    45: face_landmarks.landmark[45].z,  # mouthSmileRight
                    50: face_landmarks.landmark[50].z,  # noseSneerLe
                    51: face_landmarks.landmark[51].z,  # noseSneerRight
                    49: face_landmarks.landmark[49].z  # mouthUpperUpRight
                }

                emotion = face(face_structure)
                color = mood_colors.get(emotion, (255, 255, 255))
                cv2.putText(img, emotion, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                cv2.rectangle(img, (0, 0), (50, 50), color, thickness=-1)

        #current_time = time.time()
       #fps = 1 / (current_time - previous_time)
        #previous_time = current_time
        #cv2.putText(img, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Emotion Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Can't find emotion")
        break

cv2.destroyAllWindows()
