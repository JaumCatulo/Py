# Py
pypy
import cv2

arqCasc = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(arqCasc)
classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)  # instancia o uso da webcam

img = cv2.imread('pessoas2.png')
imc = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


facesIdentificadas = classificador.detectMultiScale (img, scaleFactor=1.1,
                                                     minNeighbors=8,
                                                     minSize=(20, 20))

for (x, y, a, l) in facesIdentificadas:
   cv2.rectangle(img, (x, y), (x +a, y + l), (0, 255, 0), 2)

cv2.imshow('Sla', img)

while True:
    s, imagem = webcam.read()  # pega a imagem da webcam
    imagem = cv2.flip(imagem, 180)  # espelha a imagem

    faces = faceCascade.detectMultiScale(
        imagem,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(200, 200),
        maxSize=(200, 200)
    )


    # Desenha um retângulo nas faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', imagem)  # mostra a imagem captura na janela

    # para o código e fechar a janela com ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
webcam.release()
cv2.destroyAllWindows()
