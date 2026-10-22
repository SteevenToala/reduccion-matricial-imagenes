"""
=============================================================================
PROCESAMIENTO DIGITAL DE IMÁGENES - INTELIGENCIA ARTIFICIAL
Aplicación Web: Reducción de Imágenes sin funciones integradas (resize / imresize)
Backend: Flask (Python) | Frontend: Vue 3 + Modern CSS (3 Columnas)
=============================================================================
"""

import os
import sys
import base64
import webbrowser
import threading
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

DIR_CODIGO = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(DIR_CODIGO, "templates"))

# =============================================================================
# ALGORITMOS FUNDAMENTALES DE REDUCCIÓN (SIN USAR RESIZE NI IMRESIZE)
# =============================================================================
def generar_kernel_gaussiano(paso: int) -> np.ndarray:
    """Genera una máscara gaussiana 2D normalizada de paso x paso."""
    if paso == 2:
        # Ponderación 2x2 simétrica
        k = np.array([[0.25, 0.25], [0.25, 0.25]], dtype=np.float32)
        return k
    sigma = paso / 3.0
    ax = np.linspace(-(paso - 1) / 2.0, (paso - 1) / 2.0, paso)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def procesar_reduccion(imagen: np.ndarray, paso: int, metodo: str) -> np.ndarray:
    """
    Reduce la imagen matricialmente aplicando el método seleccionado:
    - 'saltos': Submuestreo directo (tomar píxeles cada 'paso').
    - 'media': Promedio aritmético de bloques paso x paso (redundancia espacial).
    - 'mediana': Filtro mediana de bloques paso x paso (robusto contra ruido).
    - 'gaussiana': Media ponderada con kernel gaussiano 2D.
    """
    alto, ancho = imagen.shape[:2]
    alto_n = alto // paso
    ancho_n = ancho // paso
    
    # Recorte a múltiplo exacto del paso
    recorte = imagen[:alto_n * paso, :ancho_n * paso]
    es_color = imagen.ndim == 3
    
    if metodo == 'saltos':
        # Submuestreo directo: fila[::paso], col[::paso]
        return recorte[::paso, ::paso].copy()
        
    elif metodo == 'media':
        # Promedio aritmético del bloque
        if not es_color:
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso)
            return np.round(bloques.mean(axis=(1, 3))).astype(np.uint8)
        else:
            canales = imagen.shape[2]
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso, canales)
            return np.round(bloques.mean(axis=(1, 3))).astype(np.uint8)
            
    elif metodo == 'mediana':
        # Mediana del bloque
        if not es_color:
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso)
            bloques_flat = bloques.transpose(0, 2, 1, 3).reshape(alto_n, ancho_n, paso * paso)
            return np.round(np.median(bloques_flat, axis=2)).astype(np.uint8)
        else:
            canales = imagen.shape[2]
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso, canales)
            bloques_flat = bloques.transpose(0, 2, 4, 1, 3).reshape(alto_n, ancho_n, canales, paso * paso)
            return np.round(np.median(bloques_flat, axis=3)).astype(np.uint8)
            
    elif metodo == 'gaussiana':
        # Convolución ponderada del bloque con kernel gaussiano
        kernel = generar_kernel_gaussiano(paso)
        if not es_color:
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso)
            res = np.tensordot(bloques, kernel, axes=([1, 3], [0, 1]))
            return np.clip(np.round(res), 0, 255).astype(np.uint8)
        else:
            canales = imagen.shape[2]
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso, canales)
            res = np.tensordot(bloques, kernel, axes=([1, 3], [0, 1]))
            return np.clip(np.round(res), 0, 255).astype(np.uint8)
            
    return recorte[::paso, ::paso].copy()


def array_a_base64(img_bgr: np.ndarray) -> str:
    """Convierte matriz NumPy BGR a data URI base64 PNG."""
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


# Imagen por defecto en memoria
candidatos_ruta = [
    os.path.join(DIR_CODIGO, "imagen_ejemplo.jpg"),
    os.path.join(os.path.dirname(DIR_CODIGO), "imagenes", "imagen_ejemplo.jpg"),
    os.path.join(DIR_CODIGO, "imagenes", "imagen_ejemplo.jpg"),
]
RUTA_DEF = next((r for r in candidatos_ruta if os.path.exists(r)), candidatos_ruta[0])
if not os.path.exists(RUTA_DEF):
    # Generar sintética si no existe
    img_synth = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(img_synth, (300, 200), 100, (50, 180, 255), -1)
    cv2.imwrite(RUTA_DEF, img_synth)

IMAGEN_ACTUAL = cv2.imread(RUTA_DEF)


# =============================================================================
# RUTAS Y ENDPOINTS FLASK
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cargar-ejemplo", methods=["POST"])
def cargar_ejemplo():
    global IMAGEN_ACTUAL
    IMAGEN_ACTUAL = cv2.imread(RUTA_DEF)
    return jsonify({"success": True, "image": array_a_base64(IMAGEN_ACTUAL)})


@app.route("/api/subir-imagen", methods=["POST"])
def subir_imagen():
    global IMAGEN_ACTUAL
    if 'archivo' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    archivo = request.files['archivo']
    file_bytes = np.frombuffer(archivo.read(), np.uint8)
    img_dec = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_dec is None:
        return jsonify({"error": "Archivo no válido como imagen"}), 400
    IMAGEN_ACTUAL = img_dec
    return jsonify({"success": True, "image": array_a_base64(IMAGEN_ACTUAL)})


@app.route("/api/procesar", methods=["POST"])
def api_procesar():
    global IMAGEN_ACTUAL
    data = request.json or {}
    paso = int(data.get("paso", 3))
    metodo = data.get("metodo", "media").lower()

    if IMAGEN_ACTUAL is None:
        return jsonify({"error": "No hay imagen cargada"}), 400

    alto_orig, ancho_orig = IMAGEN_ACTUAL.shape[:2]
    paso = max(2, min(paso, min(alto_orig, ancho_orig) // 2))

    # Reducir imagen
    reducida = procesar_reduccion(IMAGEN_ACTUAL, paso, metodo)
    alto_red, ancho_red = reducida.shape[:2]

    # Estadísticas
    total_orig = alto_orig * ancho_orig
    total_red = alto_red * ancho_red
    ahorro_pct = round((1.0 - (total_red / total_orig)) * 100.0, 1)

    # Extraer bloque de muestra para la matriz numérica (escala de grises)
    gris = cv2.cvtColor(IMAGEN_ACTUAL, cv2.COLOR_BGR2GRAY)
    bloque_tam = paso
    sub_bloque = gris[0:bloque_tam, 0:bloque_tam].astype(int).tolist()

    # Calcular explicación matemática del bloque según el método
    bloque_np = np.array(sub_bloque)
    valores_flat = bloque_np.flatten().tolist()
    
    if metodo == "saltos":
        resultado_val = int(sub_bloque[0][0])
        explicacion = f"Toma directamente el píxel en (0, 0): {resultado_val} (descartando los otros {len(valores_flat)-1} píxeles del bloque)."
    elif metodo == "media":
        suma = sum(valores_flat)
        n = len(valores_flat)
        prom = round(suma / n, 2)
        resultado_val = int(round(prom))
        explicacion = f"Suma ({' + '.join(map(str, valores_flat))}) = {suma} / {n} píxeles = {prom} ≈ {resultado_val}"
    elif metodo == "mediana":
        valores_ord = sorted(valores_flat)
        med = float(np.median(valores_flat))
        resultado_val = int(round(med))
        explicacion = f"Valores ordenados: [{', '.join(map(str, valores_ord))}]. Mediana central = {med} ≈ {resultado_val}"
    elif metodo == "gaussiana":
        k = generar_kernel_gaussiano(paso)
        val_g = float(np.sum(bloque_np * k))
        resultado_val = int(round(np.clip(val_g, 0, 255)))
        explicacion = f"Suma ponderada con pesos gaussianos 2D (mayor peso al centro) = {val_g:.2f} ≈ {resultado_val}"
    else:
        resultado_val = int(sub_bloque[0][0])
        explicacion = "Cálculo estándar."

    return jsonify({
        "success": True,
        "image_orig": array_a_base64(IMAGEN_ACTUAL),
        "image_red": array_a_base64(reducida),
        "orig_width": ancho_orig,
        "orig_height": alto_orig,
        "orig_pixels": total_orig,
        "new_width": ancho_red,
        "new_height": alto_red,
        "new_pixels": total_red,
        "savings_pct": ahorro_pct,
        "paso": paso,
        "metodo": metodo,
        "matrix_sample": sub_bloque,
        "matrix_result": resultado_val,
        "matrix_explanation": explicacion
    })


@app.route("/api/simular-matriz", methods=["POST"])
def api_simular_matriz():
    """Calcula el valor resultante para una matriz numérica editada directamente por el usuario."""
    data = request.json or {}
    matriz = np.array(data.get("matrix", [[100, 105, 102], [110, 115, 112], [108, 112, 110]]), dtype=np.float32)
    metodo = data.get("metodo", "media").lower()
    paso = matriz.shape[0]

    valores_flat = matriz.flatten().tolist()
    if metodo == "saltos":
        res = int(matriz[0, 0])
        exp = f"Píxel de la esquina (0, 0): {res}"
    elif metodo == "media":
        suma = sum(valores_flat)
        n = len(valores_flat)
        res = int(round(suma / n))
        exp = f"Suma ({suma}) / {n} = {suma/n:.2f} ≈ {res}"
    elif metodo == "mediana":
        res = int(round(float(np.median(valores_flat))))
        exp = f"Mediana de [{', '.join(map(str, sorted(valores_flat)))}] = {res}"
    elif metodo == "gaussiana":
        k = generar_kernel_gaussiano(paso)
        res = int(round(np.clip(np.sum(matriz * k), 0, 255)))
        exp = f"Ponderación Gaussiana = {res}"
    else:
        res = int(matriz[0, 0])
        exp = ""

    return jsonify({"result": res, "explanation": exp})


# =============================================================================
# INICIALIZACIÓN
# =============================================================================
def abrir_navegador(puerto):
    webbrowser.open_new_tab(f"http://127.0.0.1:{puerto}")


if __name__ == "__main__":
    puerto = 5050
    print("=" * 65)
    print(f" Servidor iniciado en: http://127.0.0.1:{puerto}")
    print(" Abriendo interfaz gráfica en tu navegador...")
    print("=" * 65)
    
    # Abrir navegador tras 1 segundo
    threading.Timer(1.0, abrir_navegador, args=[puerto]).start()
    
    # Iniciar Flask
    app.run(host="127.0.0.1", port=puerto, debug=False)
