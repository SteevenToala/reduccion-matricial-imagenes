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
from skimage.util import random_noise
from flask import Flask, render_template, request, jsonify
DIR_CODIGO = os.path.dirname(os.path.abspath(__file__))
if DIR_CODIGO not in sys.path:
    sys.path.insert(0, DIR_CODIGO)

from fourier_filtros import declarar_matriz_filtro, filtrar_canal_fourier, aplicar_filtro_fourier
from reducir_imagen import KERNEL_ACENTUADO_8, aplicar_filtro_acentuado

app = Flask(__name__, template_folder=os.path.join(DIR_CODIGO, "templates"))
app.config['TEMPLATES_AUTO_RELOAD'] = True


def colorear_espectro(esp_log: np.ndarray) -> np.ndarray:
    """Normaliza y mapea a pseudocolor inferno para visualización."""
    min_v, max_v = esp_log.min(), esp_log.max()
    if max_v - min_v > 1e-6:
        norm = ((esp_log - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(esp_log, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def aplicar_suavizado_espacial(imagen: np.ndarray, tipo: str = 'gaussiano', ksize: int = 3) -> np.ndarray:
    """Aplica suavizado espacial (media o gaussiana) post-Fourier para atenuar oscilaciones de Gibbs."""
    if tipo == 'media':
        k = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
        return cv2.filter2D(imagen, -1, k)
    elif tipo == 'mediana':
        return cv2.medianBlur(imagen, ksize)
    else:
        sigma = ksize / 3.0
        return cv2.GaussianBlur(imagen, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

# =============================================================================
# FUNCIÓN DE RUIDO IMPULSIVO DE BIBLIOTECA (NO PROGRAMADA MANUALMENTE)
# =============================================================================
def imnoise(imagen: np.ndarray, tipo: str = 'salt & pepper', amount: float = 0.05) -> np.ndarray:
    """
    Añade ruido impulsivo a una imagen utilizando la función estándar de biblioteca 
    'random_noise' de scikit-image (equivalente directo a la función imnoise de MATLAB / DIP).
    
    NOTA IMPORTANTE:
    Esta función NO está programada manualmente: utiliza directamente la función
    especializada 'skimage.util.random_noise' para garantizar rigor científico.
    """
    modo = 's&p' if tipo in ('salt & pepper', 'salt and pepper', 's&p', 'sal_y_pimienta') else tipo
    # Llamada directa a función de biblioteca (scikit-image), sin bucles manuales
    imagen_con_ruido = random_noise(imagen, mode=modo, amount=amount)
    return (np.clip(imagen_con_ruido, 0.0, 1.0) * 255).astype(np.uint8)

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
            
    elif metodo == 'acentuado':
        # Filtro de acentuado de bordes con la matriz Laplaciana 3x3:
        # [-1, -1, -1]
        # [-1,  8, -1]
        # [-1, -1, -1]
        if paso == 3:
            kernel = KERNEL_ACENTUADO_8
        else:
            kernel = -np.ones((paso, paso), dtype=np.float32)
            kernel[paso // 2, paso // 2] = float(paso * paso - 1)
            
        cy, cx = paso // 2, paso // 2
        
        if not es_color:
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso).astype(np.float32)
            centros = bloques[:, cy, :, cx]
            laplaciano = np.tensordot(bloques, kernel, axes=([1, 3], [0, 1]))
            res = centros + laplaciano
            return np.clip(np.round(res), 0, 255).astype(np.uint8)
        else:
            canales = imagen.shape[2]
            bloques = recorte.reshape(alto_n, paso, ancho_n, paso, canales).astype(np.float32)
            centros = bloques[:, cy, :, cx, :]
            laplaciano = np.tensordot(bloques, kernel, axes=([1, 3], [0, 1]))
            res = centros + laplaciano
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

IMAGEN_BASE = cv2.imread(RUTA_DEF)
IMAGEN_RUIDO = None
RUIDO_ACTIVO = False
CANTIDAD_RUIDO = 0.05
FOURIER_ACTIVO = False
FOURIER_TIPO = None
FOURIER_RADIO = 40.0
SUAVIZADO_POST_ACTIVO = False
SUAVIZADO_POST_TIPO = None
ACENTUADO_ACTIVO = False
ACENTUADO_MODO = 'acentuado'


def obtener_imagen_entrada() -> np.ndarray:
    """
    Retorna la imagen de entrada de referencia (Original limpia o con Ruido sal y pimienta).
    IMPORTANTE: La imagen de entrada es la referencia y NUNCA se altera con los filtros posteriores.
    """
    global IMAGEN_BASE, IMAGEN_RUIDO, RUIDO_ACTIVO, CANTIDAD_RUIDO
    if IMAGEN_BASE is None:
        return None
    if RUIDO_ACTIVO:
        if IMAGEN_RUIDO is None:
            IMAGEN_RUIDO = imnoise(IMAGEN_BASE, tipo='salt & pepper', amount=CANTIDAD_RUIDO)
        return IMAGEN_RUIDO.copy()
    return IMAGEN_BASE.copy()


def calcular_imagen_procesada() -> np.ndarray:
    """
    Cadena secuencial de procesamiento sobre la imagen de entrada:
    1. Entrada (Base o con Ruido)
    2. Fourier 2D (Pasa Bajos para mitigar ruido)
    3. Suavizado Post Pasa-Bajos (elimina oscilaciones de Gibbs, pero desenfoca ligeramente)
    4. Filtro Laplaciano 3x3 con matriz [-1,-1,-1; -1,8,-1; -1,-1,-1] aplicado SOBRE
       la imagen suavizada para ACLARAR y REALZAR la nitidez y los bordes atenuados.
    """
    entrada = obtener_imagen_entrada()
    if entrada is None:
        return None

    img = entrada.copy()

    # 1. Fourier 2D
    if FOURIER_ACTIVO and FOURIER_TIPO is not None:
        img = aplicar_filtro_fourier(img, radio_corte=FOURIER_RADIO, tipo=FOURIER_TIPO)

    # 2. Suavizado Post Pasa-Bajos (estrictamente tras Pasa Bajos)
    if FOURIER_ACTIVO and FOURIER_TIPO == 'pasa_bajos' and SUAVIZADO_POST_ACTIVO and SUAVIZADO_POST_TIPO:
        img = aplicar_suavizado_espacial(img, tipo=SUAVIZADO_POST_TIPO, ksize=3)

    # 3. Filtro Laplaciano 3x3: aclarado y realce estrictamente sobre la salida de Pasa Bajos / Suavizado
    # (No es un filtro para la imagen original: actúa sobre la imagen resultante tras salir de pasa bajos)
    if FOURIER_ACTIVO and FOURIER_TIPO == 'pasa_bajos' and ACENTUADO_ACTIVO and ACENTUADO_MODO:
        img = aplicar_filtro_acentuado(img, modo=ACENTUADO_MODO)

    return img


# =============================================================================
# RUTAS Y ENDPOINTS FLASK
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cargar-ejemplo", methods=["POST"])
def cargar_ejemplo():
    global IMAGEN_BASE, IMAGEN_RUIDO, RUIDO_ACTIVO, FOURIER_ACTIVO, FOURIER_TIPO, SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO, ACENTUADO_ACTIVO, ACENTUADO_MODO
    IMAGEN_BASE = cv2.imread(RUTA_DEF)
    IMAGEN_RUIDO = None
    RUIDO_ACTIVO = False
    FOURIER_ACTIVO = False
    FOURIER_TIPO = None
    SUAVIZADO_POST_ACTIVO = False
    SUAVIZADO_POST_TIPO = None
    ACENTUADO_ACTIVO = False
    ACENTUADO_MODO = 'acentuado'
    return jsonify({
        "success": True,
        "ruido_activo": False,
        "fourier_activo": False,
        "suavizado_post_activo": False,
        "acentuado_activo": False
    })


@app.route("/api/subir-imagen", methods=["POST"])
def subir_imagen():
    global IMAGEN_BASE, IMAGEN_RUIDO, RUIDO_ACTIVO, FOURIER_ACTIVO, FOURIER_TIPO, SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO, ACENTUADO_ACTIVO, ACENTUADO_MODO
    if 'archivo' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    archivo = request.files['archivo']
    file_bytes = np.frombuffer(archivo.read(), np.uint8)
    img_dec = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_dec is None:
        return jsonify({"error": "Archivo no válido como imagen"}), 400
    IMAGEN_BASE = img_dec
    IMAGEN_RUIDO = None
    RUIDO_ACTIVO = False
    FOURIER_ACTIVO = False
    FOURIER_TIPO = None
    SUAVIZADO_POST_ACTIVO = False
    SUAVIZADO_POST_TIPO = None
    ACENTUADO_ACTIVO = False
    ACENTUADO_MODO = 'acentuado'
    return jsonify({
        "success": True,
        "ruido_activo": False,
        "fourier_activo": False,
        "suavizado_post_activo": False,
        "acentuado_activo": False
    })


@app.route("/api/aplicar-ruido", methods=["POST"])
def api_aplicar_ruido():
    """Aplica imnoise('salt & pepper') a la imagen base usando scikit-image."""
    global IMAGEN_BASE, IMAGEN_RUIDO, RUIDO_ACTIVO, CANTIDAD_RUIDO
    if IMAGEN_BASE is None:
        return jsonify({"error": "No hay imagen cargada"}), 400

    data = request.json or {}
    cantidad = float(data.get("cantidad", 0.05))
    CANTIDAD_RUIDO = max(0.01, min(cantidad, 0.50))
    IMAGEN_RUIDO = imnoise(IMAGEN_BASE, tipo='salt & pepper', amount=CANTIDAD_RUIDO)
    RUIDO_ACTIVO = True

    return jsonify({
        "success": True,
        "ruido_activo": True,
        "cantidad": CANTIDAD_RUIDO
    })


@app.route("/api/quitar-ruido", methods=["POST"])
def api_quitar_ruido():
    """Restaura la imagen de entrada sin ruido."""
    global IMAGEN_RUIDO, RUIDO_ACTIVO
    IMAGEN_RUIDO = None
    RUIDO_ACTIVO = False

    return jsonify({
        "success": True,
        "ruido_activo": False
    })


@app.route("/api/aplicar-fourier", methods=["POST"])
def api_aplicar_fourier():
    """
    Configura el filtrado en Frecuencia con Fourier 2D.
    Calcula la matriz H(u, v) y los espectros para visualización didáctica.
    """
    global FOURIER_ACTIVO, FOURIER_TIPO, FOURIER_RADIO, SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO, ACENTUADO_ACTIVO
    data = request.json or {}
    tipo = data.get("tipo", "pasa_bajos").lower()
    radio = float(data.get("radio", 40.0))
    FOURIER_RADIO = max(5.0, min(radio, 300.0))
    FOURIER_TIPO = tipo
    FOURIER_ACTIVO = True

    if tipo == 'pasa_altos':
        SUAVIZADO_POST_ACTIVO = False
        SUAVIZADO_POST_TIPO = None
        ACENTUADO_ACTIVO = False

    # Generar visualizaciones didácticas: Matriz H y Espectros
    entrada = obtener_imagen_entrada()
    gris = cv2.cvtColor(entrada, cv2.COLOR_BGR2GRAY) if entrada.ndim == 3 else entrada
    _, H, esp_orig, esp_filt = filtrar_canal_fourier(gris, radio_corte=FOURIER_RADIO, tipo=tipo)

    H_vis = (H * 255.0).astype(np.uint8)
    esp_orig_vis = colorear_espectro(esp_orig)
    esp_filt_vis = colorear_espectro(esp_filt)

    return jsonify({
        "success": True,
        "fourier_activo": True,
        "fourier_tipo": tipo,
        "fourier_radio": FOURIER_RADIO,
        "suavizado_post_activo": SUAVIZADO_POST_ACTIVO,
        "acentuado_activo": ACENTUADO_ACTIVO,
        "matriz_h": array_a_base64(H_vis),
        "espectro_orig": array_a_base64(esp_orig_vis),
        "espectro_filt": array_a_base64(esp_filt_vis)
    })


@app.route("/api/quitar-fourier", methods=["POST"])
def api_quitar_fourier():
    """Desactiva Fourier y sus filtros posteriores dependientes (Suavizado y Aclarado)."""
    global FOURIER_ACTIVO, FOURIER_TIPO, SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO, ACENTUADO_ACTIVO
    FOURIER_ACTIVO = False
    FOURIER_TIPO = None
    SUAVIZADO_POST_ACTIVO = False
    SUAVIZADO_POST_TIPO = None
    ACENTUADO_ACTIVO = False

    return jsonify({
        "success": True,
        "fourier_activo": False,
        "suavizado_post_activo": False,
        "acentuado_activo": False
    })


@app.route("/api/aplicar-suavizado-post", methods=["POST"])
def api_aplicar_suavizado_post():
    """
    Aplica filtro de suavizado espacial DESPUÉS de Pasa Bajos.
    El suavizado se encadena sobre la salida de Pasa Bajos para mitigar oscilaciones de Gibbs.
    """
    global FOURIER_ACTIVO, FOURIER_TIPO, SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO
    data = request.json or {}
    tipo = data.get("tipo", "gaussiano").lower()
    SUAVIZADO_POST_TIPO = tipo
    SUAVIZADO_POST_ACTIVO = True

    # Asegura que Fourier Pasa Bajos esté activo
    if not FOURIER_ACTIVO or FOURIER_TIPO != 'pasa_bajos':
        FOURIER_ACTIVO = True
        FOURIER_TIPO = 'pasa_bajos'

    return jsonify({
        "success": True,
        "fourier_activo": True,
        "fourier_tipo": "pasa_bajos",
        "suavizado_post_activo": True,
        "suavizado_post_tipo": tipo
    })


@app.route("/api/quitar-suavizado-post", methods=["POST"])
def api_quitar_suavizado_post():
    """Desactiva el suavizado posterior."""
    global SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO
    SUAVIZADO_POST_ACTIVO = False
    SUAVIZADO_POST_TIPO = None

    return jsonify({
        "success": True,
        "suavizado_post_activo": False
    })


@app.route("/api/aplicar-acentuado", methods=["POST"])
def api_aplicar_acentuado():
    """
    Aplica el filtro Laplaciano 3x3:
    [-1, -1, -1]
    [-1,  8, -1]
    [-1, -1, -1]
    DESPUÉS de que la imagen sale de Pasa Bajos y Suavizado,
    para ACLARAR la imagen resultante y realzar los bordes atenuados por el filtro.
    NO es un filtro para la imagen original: se aplica sobre el resultado suavizado.
    """
    global FOURIER_ACTIVO, FOURIER_TIPO, SUAVIZADO_POST_ACTIVO, SUAVIZADO_POST_TIPO, ACENTUADO_ACTIVO, ACENTUADO_MODO
    data = request.json or {}
    modo = data.get("modo", "acentuado").lower()  # 'acentuado' (f + ∇²f) o 'bordes' (|∇²f|)
    ACENTUADO_MODO = modo
    ACENTUADO_ACTIVO = True

    # El filtro de aclarado Laplaciano se encadena tras Pasa Bajos
    if not FOURIER_ACTIVO or FOURIER_TIPO != 'pasa_bajos':
        FOURIER_ACTIVO = True
        FOURIER_TIPO = 'pasa_bajos'

    return jsonify({
        "success": True,
        "acentuado_activo": True,
        "acentuado_modo": modo,
        "fourier_activo": True,
        "fourier_tipo": "pasa_bajos",
        "suavizado_post_activo": SUAVIZADO_POST_ACTIVO
    })


@app.route("/api/quitar-acentuado", methods=["POST"])
def api_quitar_acentuado():
    """Desactiva el filtro de aclarado Laplaciano."""
    global ACENTUADO_ACTIVO
    ACENTUADO_ACTIVO = False

    return jsonify({
        "success": True,
        "acentuado_activo": False
    })


@app.route("/api/comparativa-fourier-5-etapas", methods=["POST"])
def api_comparativa_fourier_5_etapas():
    """
    Genera las 5 etapas canónicas de restauración y aclarado:
    1. Original: Imagen limpia de referencia
    2. Ruido: Inyección de sal y pimienta con imnoise
    3. Pasa Bajos: Mitigación de ruido en frecuencia con Fourier 2D
    4. Suavizado Post: Eliminación de oscilaciones de Gibbs (atenuación de bordes)
    5. Laplaciano 3x3: Aclarado y realce de nitidez sobre la imagen suavizada
    """
    global IMAGEN_BASE
    if IMAGEN_BASE is None:
        return jsonify({"error": "No hay imagen cargada"}), 400

    data = request.json or {}
    radio = float(data.get("radio", 40.0))
    cantidad_ruido = float(data.get("cantidad_ruido", 0.06))

    gris = cv2.cvtColor(IMAGEN_BASE, cv2.COLOR_BGR2GRAY) if IMAGEN_BASE.ndim == 3 else IMAGEN_BASE.copy()

    # 1. Original
    # 2. Con Ruido Sal y Pimienta
    ruidosa = imnoise(gris, tipo='salt & pepper', amount=cantidad_ruido)

    # 3. Pasa Bajos (Fourier 2D)
    img_bajos, H_bajos, _, _ = filtrar_canal_fourier(ruidosa, radio_corte=radio, tipo='pasa_bajos')

    # 4. Suavizado Post Pasa-Bajos (elimina ringing, imagen suave)
    img_suavizada = aplicar_suavizado_espacial(img_bajos, tipo='gaussiano', ksize=3)

    # 5. Laplaciano 3x3: Aclarado y realce de nitidez con la matriz [-1,-1,-1; -1,8,-1; -1,-1,-1]
    img_laplaciano = aplicar_filtro_acentuado(img_suavizada, modo='acentuado')

    # Espectro para visualización
    F = np.fft.fft2(ruidosa.astype(np.float32))
    F_shift = np.fft.fftshift(F)
    esp_log = np.log1p(np.abs(F_shift))
    esp_color = colorear_espectro(esp_log)

    return jsonify({
        "success": True,
        "img_orig": array_a_base64(gris),
        "img_ruido": array_a_base64(ruidosa),
        "img_bajos": array_a_base64(img_bajos),
        "img_suavizado": array_a_base64(img_suavizada),
        "img_laplaciano": array_a_base64(img_laplaciano),
        "img_espectro": array_a_base64(esp_color),
        "radio": radio,
        "cantidad_ruido": cantidad_ruido
    })


@app.route("/api/procesar", methods=["POST"])
def api_procesar():
    data = request.json or {}
    paso = int(data.get("paso", 3))
    metodo = data.get("metodo", "media").lower()

    entrada = obtener_imagen_entrada()
    if entrada is None:
        return jsonify({"error": "No hay imagen cargada"}), 400

    alto_orig, ancho_orig = entrada.shape[:2]
    paso = max(2, min(paso, min(alto_orig, ancho_orig) // 2))

    # 1. Ejecutar pipeline completo sobre la imagen
    # (Entrada -> Fourier Pasa Bajos -> Suavizado Post -> Aclarado Laplaciano 3x3)
    procesada = calcular_imagen_procesada()

    # 2. Reducir imagen procesada usando el método matricial seleccionado
    reducida = procesar_reduccion(procesada, paso, metodo)
    alto_red, ancho_red = reducida.shape[:2]

    # Estadísticas
    total_orig = alto_orig * ancho_orig
    total_red = alto_red * ancho_red
    ahorro_pct = round((1.0 - (total_red / total_orig)) * 100.0, 1)

    # Extraer bloque de muestra para la matriz numérica (escala de grises)
    gris = cv2.cvtColor(procesada, cv2.COLOR_BGR2GRAY) if procesada.ndim == 3 else procesada
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
    elif metodo == "acentuado":
        cy, cx = bloque_tam // 2, bloque_tam // 2
        if bloque_tam == 3:
            k = KERNEL_ACENTUADO_8
        else:
            k = -np.ones((bloque_tam, bloque_tam), dtype=np.float32)
            k[cy, cx] = float(bloque_tam * bloque_tam - 1)
        centro = int(sub_bloque[cy][cx])
        laplaciano = int(round(float(np.sum(bloque_np * k))))
        resultado_val = int(round(np.clip(centro + laplaciano, 0, 255)))
        explicacion = f"Centro={centro}. Laplaciano ∇² (8·centro - suma vecinos) = {laplaciano}. Píxel acentuado ({centro} + {laplaciano}) = {resultado_val}"
    else:
        resultado_val = int(sub_bloque[0][0])
        explicacion = "Cálculo estándar."

    return jsonify({
        "success": True,
        "image_orig": array_a_base64(entrada),     # ENTRADA DE REFERENCIA (INTACTA)
        "image_red": array_a_base64(reducida),     # RESULTADO FINAL PROCESADO Y REDUCIDO
        "image_proc_hd": array_a_base64(procesada),# RESOLUCIÓN COMPLETA RESTAURADA
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
        "matrix_explanation": explicacion,
        "ruido_activo": RUIDO_ACTIVO,
        "cantidad_ruido": CANTIDAD_RUIDO,
        "fourier_activo": FOURIER_ACTIVO,
        "fourier_tipo": FOURIER_TIPO,
        "fourier_radio": FOURIER_RADIO,
        "suavizado_post_activo": SUAVIZADO_POST_ACTIVO,
        "suavizado_post_tipo": SUAVIZADO_POST_TIPO,
        "acentuado_activo": ACENTUADO_ACTIVO,
        "acentuado_modo": ACENTUADO_MODO
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
    elif metodo == "acentuado":
        cy, cx = paso // 2, paso // 2
        if paso == 3:
            k = KERNEL_ACENTUADO_8
        else:
            k = -np.ones((paso, paso), dtype=np.float32)
            k[cy, cx] = float(paso * paso - 1)
        centro = int(matriz[cy, cx])
        lap = int(round(float(np.sum(matriz * k))))
        res = int(round(np.clip(centro + lap, 0, 255)))
        exp = f"Matriz [-1,-1,-1 / -1,8,-1 / -1,-1,-1] -> Centro={centro} | Laplaciano={lap} | Acentuado ({centro} + {lap}) = {res}"
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
