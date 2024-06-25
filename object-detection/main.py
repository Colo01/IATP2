# Proyecto TP2 Red Neuronal CNN Pablo Calderon
# Deteccion de objetos en tiempo real a traves de webcam
# Proyecto totalmente ejecutable de manera local sin necesidad de ningun ejecutable en la nube
import numpy as np
import cv2

# Rutas de archivos
image_path = "maxresdefault.jpg"
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

# Configuración de confianza mínima
min_confidence = 0.2

# Lista de clases de objetos detectables
classes = [
    "background", "avion", "bicicleta", "pajaro", "bote", "botella", "colectivo", 
    "auto", "gato", "silla", "vaca", "mesa de cenar", "perro", "caballo", 
    "motocicleta", "persona", "planta en maceta", "oveja", "sofa", "tren", "monitor"
]

# Colores aleatorios para las clases
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Cargar el modelo de la red
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Iniciar la captura de video
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"{classes[class_id]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[class_id], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    cv2.imshow("Detected Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
