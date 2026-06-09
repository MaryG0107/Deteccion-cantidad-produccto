import cv2
import numpy as np
from datetime import datetime

# ============================================================
#  CONFIGURACION
# ============================================================
RUTA_IMAGEN = "productos2.jpg"   # nombre de tu imagen (misma carpeta)
AREA_MINIMA = 800               # area minima para contar un objeto
OBJETOS_OSCUROS = True          # True si los productos son MAS OSCUROS que el fondo
                                # False si son MAS CLAROS que el fondo


def main():
    # --- Cargar la imagen ---
    imagen = cv2.imread(RUTA_IMAGEN)
    if imagen is None:
        print(f"ERROR: No se encontro la imagen '{RUTA_IMAGEN}'.")
        print("Coloca la imagen en la misma carpeta que este script.")
        return

    # Redimensionar si es muy grande (para que quepa en pantalla)
    alto, ancho = imagen.shape[:2]
    if ancho > 1000:
        escala = 1000 / ancho
        imagen = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))

    # --- 1) Preprocesamiento ---
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (7, 7), 0)

    # --- 2) Umbral automatico (metodo de Otsu) ---
    tipo = cv2.THRESH_BINARY_INV if OBJETOS_OSCUROS else cv2.THRESH_BINARY
    _, umbral = cv2.threshold(suavizada, 0, 255, tipo + cv2.THRESH_OTSU)

    # --- 3) Limpiar la mascara (quitar ruido y cerrar huecos) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)

    # --- 4) Encontrar contornos (objetos) ---
    contornos, _ = cv2.findContours(
        umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = 0
    for c in contornos:
        if cv2.contourArea(c) < AREA_MINIMA:
            continue  # ignorar manchas pequenas (ruido)
        total += 1
        x, y, w, h = cv2.boundingRect(c)
        # Rectangulo verde alrededor de cada objeto
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Numero del objeto
        cv2.putText(imagen, f"#{total}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- 5) Panel: total + fecha/hora ---
    fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    alto2, ancho2 = imagen.shape[:2]
    cv2.rectangle(imagen, (0, 0), (ancho2, 60), (0, 0, 0), -1)
    cv2.putText(imagen, f"Total de objetos: {total}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(imagen, f"Fecha/Hora: {fecha_hora}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    print(f"Total de objetos detectados: {total}")

    # --- 6) Guardar y mostrar el resultado ---
    cv2.imwrite("resultado.jpg", imagen)
    print("Resultado guardado como 'resultado.jpg'.")
    cv2.imshow("Deteccion y conteo de productos2", imagen)
    cv2.imshow("Mascara de deteccion", umbral)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()