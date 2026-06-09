import cv2

print("Buscando camaras... presiona una tecla para pasar a la siguiente.")
for indice in range(5):
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if cap.isOpened():
        frame = None
        for _ in range(10):          # leer 10 frames para que despierte
            ret, frame = cap.read()
        if frame is not None:
            print(f">>> Indice {indice}: ABIERTO. Revisa la ventana.")
            cv2.imshow(f"Indice {indice} (esta es la del telefono?)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"Indice {indice}: no disponible")
print("Busqueda terminada.")