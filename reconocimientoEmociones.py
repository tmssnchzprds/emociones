import cv2
import os
import numpy as np


def emotionImage(emotion):
    # Emojis
    if emotion == 'alicaido': image = cv2.imread('emojis/alicaido.jpeg')
    if emotion == 'aliviado': image = cv2.imread('emojis/aliviado.jpeg')
    if emotion == 'angustia': image = cv2.imread('emojis/angustia.jpeg')
    if emotion == 'asombrado': image = cv2.imread('emojis/asombrado.jpeg')
    if emotion == 'aturdido': image = cv2.imread('emojis/aturdido.jpeg')
    if emotion == 'besando': image = cv2.imread('emojis/besando.jpeg')
    if emotion == 'bostezando': image = cv2.imread('emojis/bostezando.jpeg')
    if emotion == 'caluroso': image = cv2.imread('emojis/caluroso.jpeg')
    if emotion == 'cansado': image = cv2.imread('emojis/cansado.jpeg')
    if emotion == 'confuso': image = cv2.imread('emojis/confuso.jpeg')
    if emotion == 'conmuecas': image = cv2.imread('emojis/conmuecas.jpeg')
    if emotion == 'deprimido': image = cv2.imread('emojis/deprimido.jpeg')
    if emotion == 'deslumbrado': image = cv2.imread('emojis/deslumbrado.jpeg')
    if emotion == 'disgustado': image = cv2.imread('emojis/disgustado.jpeg')
    if emotion == 'divertido': image = cv2.imread('emojis/divertido.jpeg')
    if emotion == 'dormido': image = cv2.imread('emojis/dormido.jpeg')
    if emotion == 'enamorado': image = cv2.imread('emojis/enamorado.jpeg')
    if emotion == 'enfado': image = cv2.imread('emojis/enfado.jpeg')
    if emotion == 'enfermo': image = cv2.imread('emojis/enfermo.jpeg')
    if emotion == 'enrojecido': image = cv2.imread('emojis/enrojecido.jpeg')
    if emotion == 'estornudando': image = cv2.imread('emojis/estornudando.jpeg')
    if emotion == 'felicidad': image = cv2.imread('emojis/felicidad.jpeg')
    if emotion == 'guay': image = cv2.imread('emojis/guay.jpeg')
    if emotion == 'hambriento': image = cv2.imread('emojis/hambriento.jpeg')
    if emotion == 'llorando': image = cv2.imread('emojis/llorando.jpeg')
    if emotion == 'mareado': image = cv2.imread('emojis/mareado.jpeg')
    if emotion == 'mentiroso': image = cv2.imread('emojis/mentiroso.jpeg')
    if emotion == 'nauseas': image = cv2.imread('emojis/nauseas.jpeg')
    if emotion == 'nerd': image = cv2.imread('emojis/nerd.jpeg')
    if emotion == 'neutral': image = cv2.imread('emojis/neutral.jpeg')
    if emotion == 'pensativo': image = cv2.imread('emojis/pensativo.jpeg')
    if emotion == 'perseverante': image = cv2.imread('emojis/perseverante.jpeg')
    if emotion == 'preocupado': image = cv2.imread('emojis/preocupado.jpeg')
    if emotion == 'singracia': image = cv2.imread('emojis/singracia.jpeg')
    if emotion == 'sonrisa': image = cv2.imread('emojis/sonrisa.jpeg')
    if emotion == 'sonrisamalvada': image = cv2.imread('emojis/sonrisamalvada.jpeg')
    if emotion == 'sorpresa': image = cv2.imread('emojis/sorpresa.jpeg')
    if emotion == 'suplicando': image = cv2.imread('emojis/suplicando.jpeg')
    if emotion == 'travieso': image = cv2.imread('emojis/travieso.jpeg')
    if emotion == 'tristeza': image = cv2.imread('emojis/tristeza.jpeg')
    if emotion == 'vomitando': image = cv2.imread('emojis/vomitando.jpeg')

    return image


# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
# method = 'EigenFaces'
# method = 'FisherFaces'
method = 'LBPH'

if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo' + method + '.xml')
# --------------------------------------------------------------------------------

dataPath = './caras'  # Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:

    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # EigenFaces
        if method == 'EigenFaces':
            if result[1] < 5700:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

        # FisherFace
        if method == 'FisherFaces':
            if result[1] < 500:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

        # LBPHFace
        if method == 'LBPH':
            if result[1] < 60:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    cv2.imshow('nFrame', nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
