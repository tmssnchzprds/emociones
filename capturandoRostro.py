import cv2
import os
import imutils

# Ir descomentando las emociones que se quieran añadir una a una
# ADVERETENCIA: Utilizar muchas emociones en un mismo entrenamiento puede producir lentitud en el proceso y mostrar peores resultados.
# NOTA: Para mejores resultados no escoger a la vez emociones que tengan expresiones parecidas

# emotionName = 'alicaido'
# emotionName = 'aliviado'
# emotionName = 'angustia'
# emotionName = 'asombrado'
# emotionName = 'aturdido'
# emotionName = 'besando'
# emotionName = 'bostezando'
# emotionName = 'caluroso'
# emotionName = 'cansado'
# emotionName = 'confuso'
# emotionName = 'conmuecas'
# emotionName = 'deprimido'
# emotionName = 'deslumbrado'
# emotionName = 'disgustado'
# emotionName = 'divertido'
# emotionName = 'dormido'
# emotionName = 'enamorado'
# emotionName = 'enfado'
# emotionName = 'enfermo'
# emotionName = 'enrojecido'
# emotionName = 'estornudando'
# emotionName = 'felicidad'
# emotionName = 'guay'
# emotionName = 'hambriento'
# emotionName = 'llorando'
# emotionName = 'mareado'
# emotionName = 'mentiroso'
# emotionName = 'nauseas'
# emotionName = 'nerd'
# emotionName = 'neutral'
# emotionName = 'pensativo'
# emotionName = 'perseverante'
# emotionName = 'preocupado'
# emotionName = 'singracia'
# emotionName = 'sonrisa'
# emotionName = 'sonrisamalvada'
# emotionName = 'sorpresa'
# emotionName = 'suplicando'
# emotionName = 'travieso'
# emotionName = 'tristeza'
# emotionName = 'vomitando'

dataPath = './Caras'  # Cambia a la ruta donde hayas almacenado Data
emotionsPath = dataPath + '/' + emotionName

if not os.path.exists(emotionsPath):
    print('Carpeta creada: ', emotionsPath)
    os.makedirs(emotionsPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:

    ret, frame = cap.read()
    if not ret: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(emotionsPath + '/cara_{}.jpg'.format(count), rostro)
        count += 1
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()
