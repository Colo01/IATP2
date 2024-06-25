# Proyecto TP2 Red Neuronal CNN Pablo Calderon
# Deteccion de objetos en tiempo real a traves de webcam
# Proyecto totalmente ejecutable de manera local sin necesidad de ningun ejecutable en la nube
# Yo preferi para este tp buscar una red para entrenar y modificar localmente, ya que utilizar un entorno en la
# nube no me era comodo y era muy lento en mi maquina y con mi internet tambien.

# Importacion de librerias
import numpy as np
import cv2

# Rutas de los archivos, imagenes y modelos preentrnados
image_path = "maxresdefault.jpg"
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

# Configuración de confianza mínima para la deteccion de objetos
min_confidence = 0.2

# Lista de clases de objetos detectables
classes = [
    "background", "avion", "bicicleta", "pajaro", "bote", "botella", "colectivo", 
    "auto", "gato", "silla", "vaca", "mesa de cenar", "perro", "caballo", 
    "motocicleta", "persona", "planta en maceta", "oveja", "sofa", "tren", "monitor"
]

# Colores aleatorios para las clases de cada objeto
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Cargar el modelo que deseamos para la red
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Iniciar la captura de video con la camara web
capture = cv2.VideoCapture(0)

# Establecemos el tamaño de la ventana de la cámara web
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)  # Ancho deseado
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1300)  # Alto deseado

while True:
    # Leer un frame de la camara
    ret, frame = capture.read()
    if not ret:
        break
    # Obtener dimensiones del frame
    height, width = frame.shape[:2]
    # Crear un blob a partir del frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    # Configurar el blol como entrada de la red
    net.setInput(blob)
    # Realizar la deteccion
    detections = net.forward()
    # Aca es el punto fuerte, donde procesamos las detecciones
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
    # Mostramos el frame con las detecciones
    cv2.imshow("Detected Objects", frame)
    # Salimos de la camara aprentando la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Liberamos la captura y cerramos las ventanas
capture.release()
cv2.destroyAllWindows()
