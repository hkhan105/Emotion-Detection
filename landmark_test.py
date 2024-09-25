#Importing necessary libraries
import cv2
import mediapipe as mp
import os

#Initializng image dataset
img = cv2.imread("//Contempt")

#Initializng face mesh for facial detection
mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=3, circle_radius=3)

#Function: Stating conditionals that contain ranges of blendshape values for each emotion
def face(blendshape):
    print(blendshape[44], blendshape[45], blendshape[7], blendshape[8], blendshape[50], blendshape[51])
    #elif (0.0070 < blendshape[25] < 0.08) and (0.06 < blendshape[21] < 0.2) and (-0.02< blendshape[22] < 0.05) and (-0.2 < blendshape[3] < -0.03) and (-0.3 < blendshape[4] < -0.06) and (-0.2 < blendshape[5] <-0.05) and (-0.2 < blendshape[6] <-0.003):
        #return "Surprise"
    if (-0.032 < blendshape[36] < 0.0039) and (-0.067 < blendshape[37] < -0.009) and (-0.195< blendshape[1] < -0.086) and (-0.086 < blendshape[2] < -0.033) and (-0.02 < blendshape[50] < 0.06) and (-0.21 < blendshape[51] <-0.08):
        return "Anger"
    #if (-0.012 < blendshape[30] < 0.048) and (0.015 < blendshape[31] < 0.59) and (0.002 < blendshape[7] < 0.054) and (-0.078 < blendshape[8] < 0.003) and (-0.11 < blendshape[3] < -0.05):
        #return "Sadness"
    #if (-0.132 < blendshape[44] < -0.059) and (-0.14 < blendshape[45] < -0.06) and (-0.023 < blendshape[7] < 0.034) and (-0.093 < blendshape[8] < -0.012):
       #return "Happy"
    #if (-0.096 < blendshape[3] < -0.02) and (0.015 < blendshape[21] < 0.193) and (-0.018 < blendshape[22] < 0.039) and (-0.017 < blendshape[25] < 0.07):
        #return "Fear"
    #if (-0.032 < blendshape[50] < 0.034) and (-0.1 < blendshape[51] < 0.03) and (-0.032 < blendshape[36] < 0.004) and (-0.055 < blendshape[37] < -0.015) and (-0.1 < blendshape[1] < 0.034) and (-0.11 < blendshape[2] < -0.0004):
        #return "Disgust"
    #if (-0.175 < blendshape[44] < -0.06) and (-0.185 < blendshape[45] < -0.06) and (-0.0041 < blendshape[7] < 0.04) and (-0.103 < blendshape[8] < -0.0009) and (-0.030 < blendshape[50] < 0.045) and (-0.17 < blendshape[51] < -0.05):
        #return "Contempt"
    #else:
        #return "Neutral"

#List of colors that correspond to emotion detected (RGB)
mood_colors = {
    "Happy": (255, 255, 0),
    "Surprise": (255, 255, 255),
    "Sadness": (255, 165, 0),
    "Anger": (160, 32, 240),
    "Fear": (0, 255, 255),
    "Disgust": (0, 255, 0),
    "Contempt": (255, 0, 0),
    "Neutral": (0, 0, 0)
}

#Looping through each image in the dataset to test its values
for photo in os.listdir("//Contempt"):
    img_path = os.path.join("//Contempt", photo)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

#Upon detecting landmarks and faces, apply mesh and run through each of the landmarks listed below
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, draw_spec, draw_spec)

            # List of blendshapes
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
            #Calling function to check if any values match defined emotions
            emotion = face(face_structure)
            print(emotion)

    #Display each image as it's processes
    cv2.imshow("Emotion Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
