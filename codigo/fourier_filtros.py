"""
=============================================================================
PROCESAMIENTO DIGITAL DE IMÁGENES - INTELIGENCIA ARTIFICIAL
Tema: Filtrado en el Dominio de la Frecuencia con Transformada de Fourier 2D
Filtros: Pasa Bajos (Mitigación de Ruido) y Pasa Altos (Bordes y Detalles)
Método: Declaración de Matriz de Ceros y Multiplicación de Matrices en Frecuencia
=============================================================================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise


# =============================================================================
# 1. FUNCIÓN DE RUIDO DE BIBLIOTECA (imnoise)
# =============================================================================
def imnoise(imagen: np.ndarray, tipo: str = 'salt & pepper', amount: float = 0.05) -> np.ndarray:
    """
    Añade ruido impulsivo utilizando la biblioteca oficial scikit-image (no manual).
    """
    modo = 's&p' if tipo in ('salt & pepper', 'salt and pepper', 's&p', 'sal_y_pimienta') else tipo
    ruidosa = random_noise(imagen, mode=modo, amount=amount)
    return (np.clip(ruidosa, 0.0, 1.0) * 255).astype(np.uint8)


# =============================================================================
# 2. DECLARACIÓN DE LA MATRIZ DE CEROS Y CONSTRUCCIÓN DEL FILTRO H(u, v)
# =============================================================================
def declarar_matriz_filtro(alto: int, ancho: int, radio_corte: float = 40.0, tipo: str = 'pasa_bajos') -> np.ndarray:
    """
    PASO 1 Y 2: DECLARAR LA MATRIZ DE CEROS Y CREAR LA MÁSCARA DEL FILTRO.
    
    Parámetros:
    -----------
    alto, ancho : int
        Dimensiones exactas de la imagen a procesar.
    radio_corte : float (D0)
        Frecuencia de corte radial desde el centro del espectro.
    tipo : str
        'pasa_bajos' (Low-Pass Filter) o 'pasa_altos' (High-Pass Filter).
        
    Retorna:
    --------
    H : np.ndarray
        Matriz 2D de dimensiones (alto, ancho) con ceros y unos normalizados [0.0, 1.0].
    """
    # 1. DECLARAR MATRIZ DE CEROS DEL MISMO TAMAÑO
    H = np.zeros((alto, ancho), dtype=np.float32)

    # Coordenadas del centro del espectro centrado (u_0, v_0)
    centro_y = alto // 2
    centro_x = ancho // 2

    # Matriz de distancias euclidianas D(u, v) desde el centro:
    # D(u, v) = sqrt((u - M/2)^2 + (v - N/2)^2)
    y, x = np.ogrid[:alto, :ancho]
    distancia = np.sqrt((y - centro_y) ** 2 + (x - centro_x) ** 2)

    if tipo.lower() in ('pasa_bajos', 'low_pass', 'lpf', 'bajos'):
        # FILTRO PASA BAJOS IDEAL (ILPF):
        # H(u, v) = 1 si D(u, v) <= D0; 0 en el resto
        # Permite el paso de bajas frecuencias (fondo, formas suaves)
        # y bloquea altas frecuencias (ruido fino, cambios bruscos).
        H[distancia <= radio_corte] = 1.0

    elif tipo.lower() in ('pasa_altos', 'high_pass', 'hpf', 'altos'):
        # FILTRO PASA ALTOS IDEAL (IHPF):
        # H(u, v) = 1 si D(u, v) > D0; 0 en el centro
        # Bloquea frecuencias bajas continuas y resalta bordes y discontinuidades.
        H[distancia > radio_corte] = 1.0

    elif tipo.lower() in ('gaussiano_bajos', 'glpf'):
        # Filtro Pasa Bajos Gaussiano (suave, sin oscilaciones de Gibbs)
        H = np.exp(-(distancia ** 2) / (2.0 * (radio_corte ** 2)))

    elif tipo.lower() in ('gaussiano_altos', 'ghpf'):
        # Filtro Pasa Altos Gaussiano
        H = 1.0 - np.exp(-(distancia ** 2) / (2.0 * (radio_corte ** 2)))

    return H


# =============================================================================
# 3. ALGORITMO COMPLETO DE TRANSFORMADA DE FOURIER Y MULTIPLICACIÓN
# =============================================================================
def filtrar_canal_fourier(canal_2d: np.ndarray, radio_corte: float = 40.0, tipo: str = 'pasa_bajos'):
    """
    Aplica el ciclo completo de Fourier sobre una matriz 2D:
    f(x, y) -> FFT -> F(u, v) -> fftshift -> F_shift * H -> ifftshift -> iFFT -> g(x, y)
    
    Retorna:
    --------
    canal_filtrado : np.ndarray (uint8)
    H : np.ndarray (matriz de ceros/unos del filtro)
    espectro_orig_log : np.ndarray (espectro centrado original en escala log)
    espectro_filt_log : np.ndarray (espectro centrado tras multiplicar por H)
    """
    alto, ancho = canal_2d.shape[:2]

    # 1. Transformada Rápida de Fourier 2D (FFT)
    # Convierte la señal del dominio espacial al dominio frecuencial complejo
    F = np.fft.fft2(canal_2d.astype(np.float32))

    # 2. Desplazamiento del cero de frecuencia al centro de la matriz (fftshift)
    F_shift = np.fft.fftshift(F)

    # 3. Declarar la matriz de ceros y configurar la máscara H(u, v)
    H = declarar_matriz_filtro(alto, ancho, radio_corte=radio_corte, tipo=tipo)

    # 4. MULTIPLICAR LAS MATRICES (Producto elemento a elemento en frecuencia)
    # Teorema de Convolución: f(x, y) * h(x, y) <=> F(u, v) x H(u, v)
    G_shift = F_shift * H

    # 5. Deshacer el centrado (ifftshift)
    G = np.fft.ifftshift(G_shift)

    # 6. Transformada Inversa de Fourier (iFFT) para volver al dominio espacial
    g_espacial = np.fft.ifft2(G)

    # Extraer la parte real y recortar a rango válido [0, 255]
    canal_filtrado = np.clip(np.real(g_espacial), 0, 255).astype(np.uint8)

    # Espectros de magnitud en decibelios (escala logarítmica) para visualizar
    espectro_orig_log = np.log1p(np.abs(F_shift))
    espectro_filt_log = np.log1p(np.abs(G_shift))

    return canal_filtrado, H, espectro_orig_log, espectro_filt_log


def aplicar_filtro_fourier(imagen: np.ndarray, radio_corte: float = 40.0, tipo: str = 'pasa_bajos') -> np.ndarray:
    """
    Aplica el filtro de Fourier a una imagen completa (escala de grises o BGR/RGB).
    """
    if imagen.ndim == 2:
        res, _, _, _ = filtrar_canal_fourier(imagen, radio_corte, tipo)
        return res
    else:
        # Procesar canal por canal (R, G, B)
        canales = cv2.split(imagen)
        canales_filtrados = []
        for c in canales:
            c_f, _, _, _ = filtrar_canal_fourier(c, radio_corte, tipo)
            canales_filtrados.append(c_f)
        return cv2.merge(canales_filtrados)


# =============================================================================
# 4. DEMOSTRACIÓN VISUAL Y CIENTÍFICA
# =============================================================================
def demostrar_fourier(ruta_imagen: str = None, radio_corte: float = 35.0, aplicar_ruido_previo: bool = True, ruta_guardar: str = None, mostrar: bool = True):
    """
    Demostración visual completa en una ventana de Matplotlib:
    Muestra:
    1. Imagen con Ruido (o Entrada Original)
    2. Espectro de Fourier Original centrado |F(u, v)|
    3. Matriz del Filtro Pasa Bajos H_LP (Círculo de 1s sobre matriz de 0s)
    4. Espectro Multiplicado G_LP = F * H_LP
    5. Imagen Resultante Pasa Bajos (Mitigación del Ruido)
    6. Matriz del Filtro Pasa Altos H_HP
    7. Imagen Resultante Pasa Altos (Extracción de Bordes)
    """
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    if ruta_imagen is None or not os.path.exists(ruta_imagen):
        candidatos = [
            os.path.join(dir_actual, "imagen_ejemplo.jpg"),
            os.path.join(os.path.dirname(dir_actual), "imagenes", "imagen_ejemplo.jpg")
        ]
        ruta_imagen = next((c for c in candidatos if os.path.exists(c)), candidatos[0])

    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"[ERROR] No se pudo cargar la imagen: {ruta_imagen}")
        return

    # Trabajar con escala de grises para máxima claridad matemática
    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Inyectar ruido si está solicitado (para demostrar la mitigación)
    if aplicar_ruido_previo:
        print("[+] Inyectando ruido Sal y Pimienta con imnoise() de biblioteca...")
        img_entrada = imnoise(img_gris, tipo='salt & pepper', amount=0.06)
    else:
        img_entrada = img_gris

    print(f"[+] Aplicando Transformada de Fourier 2D...")
    print(f"    1. Declarando matriz de ceros H de tamaño {img_entrada.shape}...")
    print(f"    2. Construyendo filtros Pasa Bajos y Pasa Altos con D0 = {radio_corte} px...")
    print(f"    3. Multiplicando matrices en frecuencia: G(u,v) = F(u,v) * H(u,v)...")
    print(f"    4. Calculando Transformada Inversa de Fourier (iFFT)...")

    # 1. Pasa Bajos (Mitigación de ruido)
    img_bajos, H_bajos, esp_orig, esp_bajos = filtrar_canal_fourier(img_entrada, radio_corte, tipo='pasa_bajos')

    # 2. Pasa Altos (Detección de bordes)
    img_altos, H_altos, _, esp_altos = filtrar_canal_fourier(img_entrada, radio_corte, tipo='pasa_altos')

    # Construir figura explicativa
    fig, axs = plt.subplots(2, 4, figsize=(18, 9), facecolor='white')

    # Fila 1: Entrada, Espectro, Filtro Pasa Bajos y Salida Pasa Bajos
    axs[0, 0].imshow(img_entrada, cmap='gray')
    axs[0, 0].set_title(f"1. Entrada {'con Ruido (imnoise)' if aplicar_ruido_previo else ''}", fontweight='bold')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(esp_orig, cmap='inferno')
    axs[0, 1].set_title("2. Espectro Centrado |F(u, v)|\n(Logarítmico)", fontweight='bold')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(H_bajos, cmap='gray')
    axs[0, 2].set_title(f"3. Matriz Filtro Pasa Bajos H\n(Matriz de 0s con círculo de 1s)", fontweight='bold', color='#15803d')
    axs[0, 2].axis('off')

    axs[0, 3].imshow(img_bajos, cmap='gray')
    axs[0, 3].set_title(f"4. Salida Pasa Bajos (iFFT)\n★ MITIGACIÓN DEL RUIDO", fontweight='bold', color='#15803d')
    axs[0, 3].axis('off')

    # Fila 2: Espectro Multiplicado Bajos, Filtro Pasa Altos, Espectro Multiplicado Altos, Salida Pasa Altos
    axs[1, 0].imshow(esp_bajos, cmap='inferno')
    axs[1, 0].set_title("5. Espectro G = F * H_LP\n(Altas frec. eliminadas)", fontweight='bold')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(H_altos, cmap='gray')
    axs[1, 1].set_title(f"6. Matriz Filtro Pasa Altos H\n(Centro 0s, periferia 1s)", fontweight='bold', color='#1d4ed8')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(esp_altos, cmap='inferno')
    axs[1, 2].set_title("7. Espectro G = F * H_HP\n(Solo altas frecuencias)", fontweight='bold')
    axs[1, 2].axis('off')

    axs[1, 3].imshow(img_altos, cmap='gray')
    axs[1, 3].set_title(f"8. Salida Pasa Altos (iFFT)\n(Bordes y altas frecuencias)", fontweight='bold', color='#1d4ed8')
    axs[1, 3].axis('off')

    plt.suptitle(
        f"Filtrado Frecuencial con Transformada de Fourier 2D\n"
        f"G(u, v) = F(u, v) · H(u, v)  |  Radio de Corte D0 = {radio_corte} px",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    if ruta_guardar:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_guardar)), exist_ok=True)
        plt.savefig(ruta_guardar, dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico de Fourier guardado en: {ruta_guardar}")

    if mostrar:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 5. TIRA COMPARATIVA CANÓNICA DE 5 ETAPAS (ORIGINAL -> RUIDO -> FFT -> PASA BAJOS -> PASA ALTOS)
# =============================================================================
def generar_comparativa_5_etapas(ruta_imagen: str = None, radio_corte: float = 35.0, cantidad_ruido: float = 0.06, ruta_guardar: str = None, mostrar: bool = True):
    """
    Genera la tira comparativa canónica con exactamente las 5 etapas solicitadas:
    1. Imagen Original
    2. Imagen con Ruido (imnoise 'salt & pepper')
    3. Espectro FFT 2D centrado |F(u, v)|
    4. Imagen Filtrada Pasa Bajos (Mitigación del ruido)
    5. Imagen Filtrada Pasa Altos (Extracción de bordes)
    """
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    if ruta_imagen is None or not os.path.exists(ruta_imagen):
        candidatos = [
            os.path.join(dir_actual, "imagen_ejemplo.jpg"),
            os.path.join(os.path.dirname(dir_actual), "imagenes", "imagen_ejemplo.jpg")
        ]
        ruta_imagen = next((c for c in candidatos if os.path.exists(c)), candidatos[0])

    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"[ERROR] No se pudo leer {ruta_imagen}")
        return None

    # Trabajar con escala de grises para máxima inteligibilidad matemática
    img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # 1. Contaminar con Ruido Sal y Pimienta usando función de biblioteca (imnoise)
    img_ruido = imnoise(img_orig, tipo='salt & pepper', amount=cantidad_ruido)

    # 2. Transformada de Fourier 2D del estado ruidoso y espectro centrado
    F = np.fft.fft2(img_ruido.astype(np.float32))
    F_shift = np.fft.fftshift(F)
    espectro_log = np.log1p(np.abs(F_shift))

    # 3. Filtrado Pasa Bajos: Matriz de ceros H y multiplicación G = F * H
    img_bajos, H_bajos, _, _ = filtrar_canal_fourier(img_ruido, radio_corte=radio_corte, tipo='pasa_bajos')

    # 4. Filtrado Pasa Altos
    img_altos, H_altos, _, _ = filtrar_canal_fourier(img_ruido, radio_corte=radio_corte, tipo='pasa_altos')

    # Construir visualización en una sola fila con 5 paneles
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.8), facecolor='white')

    # Panel 1: Original
    axs[0].imshow(img_orig, cmap='gray')
    axs[0].set_title(f"1. Imagen Original\n({img_orig.shape[1]}x{img_orig.shape[0]} px)", fontsize=11, fontweight='bold', color='#0f172a')
    axs[0].axis('off')

    # Panel 2: Con Ruido
    axs[1].imshow(img_ruido, cmap='gray')
    axs[1].set_title(f"2. Con Ruido (imnoise)\nSal y Pimienta al {int(cantidad_ruido*100)}%", fontsize=11, fontweight='bold', color='#b45309')
    axs[1].axis('off')

    # Panel 3: Espectro FFT
    axs[2].imshow(espectro_log, cmap='inferno')
    axs[2].set_title("3. Espectro FFT 2D\n|F(u, v)| (Logarítmico)", fontsize=11, fontweight='bold', color='#4338ca')
    axs[2].axis('off')

    # Panel 4: Pasa Bajos
    axs[3].imshow(img_bajos, cmap='gray')
    axs[3].set_title(f"4. Filtrada Pasa Bajos\n★ Mitiga Ruido (D0={radio_corte:.0f} px)", fontsize=11, fontweight='bold', color='#15803d')
    axs[3].axis('off')

    # Panel 5: Pasa Altos
    axs[4].imshow(img_altos, cmap='gray')
    axs[4].set_title(f"5. Filtrada Pasa Altos\n(Bordes y altas frec.)", fontsize=11, fontweight='bold', color='#1d4ed8')
    axs[4].axis('off')

    plt.suptitle("Procesamiento Digital de Imágenes en Frecuencia: Original → Ruido → Espectro FFT → Pasa Bajos → Pasa Altos", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()

    if ruta_guardar:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_guardar)), exist_ok=True)
        plt.savefig(ruta_guardar, dpi=300, bbox_inches='tight')
        print(f"[OK] Tira comparativa de 5 etapas guardada en: {ruta_guardar}")

    if mostrar:
        plt.show()
    else:
        plt.close()

    return {
        "img_orig": img_orig,
        "img_ruido": img_ruido,
        "espectro_log": espectro_log,
        "img_bajos": img_bajos,
        "img_altos": img_altos
    }


# =============================================================================
# 5. MENÚ DE CONSOLA
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("   UNIVERSIDAD TÉCNICA DE AMBATO - INTELIGENCIA ARTIFICIAL")
    print("   FILTRADO CON TRANSFORMADA DE FOURIER 2D (PASA BAJOS / PASA ALTOS)")
    print("   Algoritmo: Declarar matriz de ceros H y multiplicar: G = F * H")
    print("=" * 70)

    dir_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_def = os.path.join(dir_actual, "imagen_ejemplo.jpg")
    if not os.path.exists(ruta_def):
        ruta_def = os.path.join(os.path.dirname(dir_actual), "imagenes", "imagen_ejemplo.jpg")

    ruta = input(f"\nRuta de la imagen [def: {ruta_def}]: ").strip() or ruta_def
    radio_corte_str = input("Radio de corte D0 en píxeles [def: 40]: ").strip() or "40"
    radio_corte = float(radio_corte_str)
    con_ruido = input("¿Inyectar ruido sal y pimienta con imnoise() para mitigar? (S/n): ").strip().lower() != 'n'

    demostrar_fourier(ruta, radio_corte=radio_corte, aplicar_ruido_previo=con_ruido)
