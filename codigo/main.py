"""
=============================================================================
UNIVERSIDAD TÉCNICA DE AMBATO
FACULTAD DE INGENIERÍA EN SISTEMAS, ELECTRÓNICA E INDUSTRIAL
CARRERA DE SOFTWARE - INTELIGENCIA ARTIFICIAL

TEMA: REDUCCIÓN DE IMÁGENES MATRICIAL SIN FUNCIONES DE LIBRERÍA (CV2.RESIZE / IMRESIZE)
MÉTODOS: SALTOS (DECIMACIÓN), PROMEDIO (MEAN POOLING), MEDIANA Y GAUSSIANA
MATRICES IMPARES: 3x3, 5x5, 7x7

AUTORES: Toala Steeven, Chavez Michael
DOCENTE: Ing. Mg. Ruben Nogales
=============================================================================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.util import random_noise

DIR_ACTUAL_MAIN = os.path.dirname(os.path.abspath(__file__))
if DIR_ACTUAL_MAIN not in sys.path:
    sys.path.insert(0, DIR_ACTUAL_MAIN)

from fourier_filtros import demostrar_fourier, aplicar_filtro_fourier, declarar_matriz_filtro
from reducir_imagen import KERNEL_ACENTUADO_8, aplicar_filtro_acentuado, reducir_por_acentuado_manual


# =============================================================================
# FUNCIÓN DE RUIDO DE BIBLIOTECA (NO PROGRAMADA MANUALMENTE)
# =============================================================================
def imnoise(imagen: np.ndarray, tipo: str = 'salt & pepper', amount: float = 0.05) -> np.ndarray:
    """
    Añade ruido impulsivo a una imagen utilizando la función estándar de biblioteca 
    'random_noise' de scikit-image (equivalente directo al imnoise de MATLAB / DIP).
    
    NOTA IMPORTANTE:
    Esta función NO está programada manualmente: utiliza directamente la función
    especializada 'skimage.util.random_noise' para garantizar rigor científico.
    """
    modo = 's&p' if tipo in ('salt & pepper', 'salt and pepper', 's&p', 'sal_y_pimienta') else tipo
    # Llamada directa a función de biblioteca (scikit-image), sin bucles manuales
    imagen_con_ruido = random_noise(imagen, mode=modo, amount=amount)
    return (np.clip(imagen_con_ruido, 0.0, 1.0) * 255).astype(np.uint8)


# =============================================================================
# 1. GENERACIÓN DEL KERNEL GAUSSIANO 2D NORMALIZADO
# =============================================================================
def generar_kernel_gaussiano(paso_k: int) -> np.ndarray:
    """
    Genera una máscara gaussiana bidimensional de tamaño (k x k).
    Garantiza que la suma total de los pesos sea exactamente 1.0 (normalización),
    preservando la escala de luminosidad original de la imagen.
    """
    desviacion_estandar = paso_k / 3.0
    rango_ejes = np.linspace(-(paso_k - 1) / 2.0, (paso_k - 1) / 2.0, paso_k)
    curva_gaussiana = np.exp(-0.5 * np.square(rango_ejes) / np.square(desviacion_estandar))
    kernel_bidimensional = np.outer(curva_gaussiana, curva_gaussiana)
    kernel_normalizado = kernel_bidimensional / np.sum(kernel_bidimensional)
    return kernel_normalizado


# =============================================================================
# 2. MÉTODO 1: REDUCCIÓN POR SUBMUESTREO (DECIMACIÓN ESPACIAL / SALTOS)
# =============================================================================
def reducir_por_saltos(imagen_original: np.ndarray, paso_k: int) -> np.ndarray:
    """
    Muestreo espacial directo dando saltos de 'paso_k' píxeles en filas y columnas.
    Fórmula matemática:
        I_reducida(i, j) = I_original(i * k, j * k)
    
    Descarta los píxeles intermedios y toma directamente la muestra discreta.
    """
    alto_original, ancho_original = imagen_original.shape[:2]
    alto_nuevo = alto_original // paso_k
    ancho_nuevo = ancho_original // paso_k

    if imagen_original.ndim == 2:  # Escala de grises (matriz 2D)
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                imagen_reducida[fila, columna] = imagen_original[fila * paso_k, columna * paso_k]
    else:  # Color RGB/BGR (matriz 3D)
        canales = imagen_original.shape[2]
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                for c in range(canales):
                    imagen_reducida[fila, columna, c] = imagen_original[fila * paso_k, columna * paso_k, c]

    return imagen_reducida


# =============================================================================
# 3. MÉTODO 2: REDUCCIÓN POR PROMEDIO DE BLOQUES (MEAN POOLING / REDUNDANCIA)
# =============================================================================
def reducir_por_promedio(imagen_original: np.ndarray, paso_k: int) -> np.ndarray:
    """
    Aprovecha la redundancia espacial de píxeles adyacentes promediando
    cada vecindad de tamaño (k x k).
    Fórmula matemática:
        I_reducida(i, j) = round( (1 / k^2) * sum_{u=0}^{k-1} sum_{v=0}^{k-1} I_original(i*k + u, j*k + v) )
    """
    alto_original, ancho_original = imagen_original.shape[:2]
    alto_nuevo = alto_original // paso_k
    ancho_nuevo = ancho_original // paso_k
    total_pixeles_bloque = paso_k * paso_k

    if imagen_original.ndim == 2:  # Escala de grises
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                suma_intensidades = 0.0
                for delta_fila in range(paso_k):
                    for delta_col in range(paso_k):
                        suma_intensidades += float(imagen_original[fila * paso_k + delta_fila, columna * paso_k + delta_col])
                promedio = int(round(suma_intensidades / total_pixeles_bloque))
                imagen_reducida[fila, columna] = np.clip(promedio, 0, 255)
    else:  # Color
        canales = imagen_original.shape[2]
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                for c in range(canales):
                    suma_intensidades = 0.0
                    for delta_fila in range(paso_k):
                        for delta_col in range(paso_k):
                            suma_intensidades += float(imagen_original[fila * paso_k + delta_fila, columna * paso_k + delta_col, c])
                    promedio = int(round(suma_intensidades / total_pixeles_bloque))
                    imagen_reducida[fila, columna, c] = np.clip(promedio, 0, 255)

    return imagen_reducida


# =============================================================================
# 4. MÉTODO 3: REDUCCIÓN POR FILTRO MEDIANA (MEDIAN POOLING)
# =============================================================================
def reducir_por_mediana(imagen_original: np.ndarray, paso_k: int) -> np.ndarray:
    """
    Ordena los k^2 píxeles de cada vecindad y toma el elemento central.
    Al utilizar ventanas impares (3x3, 5x5, 7x7), el total de muestras es impar
    (9, 25, 49), garantizando un valor mediano real existente de la escena.
    Excelente resistencia frente a ruido impulsivo de tipo sal y pimienta.
    """
    alto_original, ancho_original = imagen_original.shape[:2]
    alto_nuevo = alto_original // paso_k
    ancho_nuevo = ancho_original // paso_k

    if imagen_original.ndim == 2:
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                bloque = imagen_original[fila * paso_k : (fila + 1) * paso_k, columna * paso_k : (columna + 1) * paso_k]
                mediana_valor = int(round(float(np.median(bloque))))
                imagen_reducida[fila, columna] = mediana_valor
    else:
        canales = imagen_original.shape[2]
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                for c in range(canales):
                    bloque_canal = imagen_original[fila * paso_k : (fila + 1) * paso_k, columna * paso_k : (columna + 1) * paso_k, c]
                    mediana_valor = int(round(float(np.median(bloque_canal))))
                    imagen_reducida[fila, columna, c] = mediana_valor

    return imagen_reducida


# =============================================================================
# 5. MÉTODO 4: REDUCCIÓN POR FILTRADO GAUSSIANO PONDERADO 2D
# =============================================================================
def reducir_por_gaussiana(imagen_original: np.ndarray, paso_k: int) -> np.ndarray:
    """
    Realiza una convolución ponderada con el kernel gaussiano discreto.
    Otorga mayor peso al píxel central y disminuye suavemente hacia los extremos,
    eliminando frecuencias altas y atenuando el efecto de aliasing o pixelado.
    """
    alto_original, ancho_original = imagen_original.shape[:2]
    alto_nuevo = alto_original // paso_k
    ancho_nuevo = ancho_original // paso_k
    kernel = generar_kernel_gaussiano(paso_k)

    if imagen_original.ndim == 2:
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                bloque = imagen_original[fila * paso_k : (fila + 1) * paso_k, columna * paso_k : (columna + 1) * paso_k].astype(np.float64)
                valor_ponderado = np.sum(bloque * kernel)
                imagen_reducida[fila, columna] = int(round(np.clip(valor_ponderado, 0, 255)))
    else:
        canales = imagen_original.shape[2]
        imagen_reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for fila in range(alto_nuevo):
            for columna in range(ancho_nuevo):
                for c in range(canales):
                    bloque_canal = imagen_original[fila * paso_k : (fila + 1) * paso_k, columna * paso_k : (columna + 1) * paso_k, c].astype(np.float64)
                    valor_ponderado = np.sum(bloque_canal * kernel)
                    imagen_reducida[fila, columna, c] = int(round(np.clip(valor_ponderado, 0, 255)))

    return imagen_reducida


# =============================================================================
# 6. INSPECCIÓN MATRICIAL PASO A PASO EN TERMINAL
# =============================================================================
def imprimir_inspeccion_bloque(imagen_original: np.ndarray, paso_k: int):
    """
    Imprime en consola la matriz numérica real del primer bloque (0, 0)
    y muestra cómo cada uno de los 4 métodos calcula su píxel resultante.
    """
    gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY) if imagen_original.ndim == 3 else imagen_original
    bloque = gris[0:paso_k, 0:paso_k].astype(int)
    valores_flat = bloque.flatten().tolist()
    
    # 1. Saltos
    pixel_saltos = int(bloque[0, 0])
    
    # 2. Promedio
    suma = sum(valores_flat)
    promedio = suma / (paso_k * paso_k)
    pixel_promedio = int(round(promedio))
    
    # 3. Mediana
    ordenados = sorted(valores_flat)
    mediana = float(np.median(valores_flat))
    pixel_mediana = int(round(mediana))
    
    # 4. Gaussiana
    kernel = generar_kernel_gaussiano(paso_k)
    val_gauss = float(np.sum(bloque * kernel))
    pixel_gauss = int(round(np.clip(val_gauss, 0, 255)))

    print("\n" + "=" * 75)
    print(f" DEMOSTRACIÓN MATRICIAL DETALLADA DEL BLOQUE (0, 0) - PASO k = {paso_k}")
    print("=" * 75)
    print(f"Matriz de entrada de píxeles reales ({paso_k}x{paso_k}):")
    for f in range(paso_k):
        fila_str = "  ".join([f"{bloque[f, c]:>4}" for c in range(paso_k)])
        print(f"   Fila {f}: [ {fila_str} ]")

    print("\n--- RESULTADOS ARITMÉTICOS POR MÉTODO ---")
    print(f"1. SUBMUESTREO (SALTOS):")
    print(f"   -> Selecciona directamente la esquina superior izquierda I(0, 0): {pixel_saltos}")
    print(f"   -> Píxeles omitidos del bloque: {len(valores_flat) - 1} de {len(valores_flat)}")

    print(f"\n2. PROMEDIO DE BLOQUE (MEAN POOLING):")
    print(f"   -> Suma de intensidades = {suma}")
    print(f"   -> Promedio = {suma} / {paso_k * paso_k} = {promedio:.2f} -> Redondeo: {pixel_promedio}")

    print(f"\n3. FILTRO MEDIANA:")
    print(f"   -> Valores ordenados: {ordenados}")
    print(f"   -> Elemento central en índice {(paso_k*paso_k)//2}: {mediana} -> Redondeo: {pixel_mediana}")

    print(f"\n4. FILTRO GAUSSIANO 2D:")
    print(f"   -> Kernel gaussiano centrado con pesos sumando 1.0")
    print(f"   -> Suma producto bloque * kernel = {val_gauss:.2f} -> Redondeo: {pixel_gauss}")
    print("=" * 75 + "\n")


# =============================================================================
# 7. GENERACIÓN DE FIGURAS Y GRÁFICOS CIENTÍFICOS PARA EL INFORME
# =============================================================================
def generar_graficos_informe(ruta_imagen: str, carpeta_salida_informe: str):
    """
    Genera automáticamente todas las figuras científicas en alta resolución
    requeridas para el informe de laboratorio en LaTeX.
    """
    os.makedirs(carpeta_salida_informe, exist_ok=True)
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"[ERROR] No se pudo leer la imagen: '{ruta_imagen}'")
        return

    alto_orig, ancho_orig = imagen.shape[:2]
    total_pixeles_orig = alto_orig * ancho_orig
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    print("\n[+] Generando figuras para el informe de laboratorio...")

    # -------------------------------------------------------------------------
    # FIGURA 1: COMPARATIVA DE LOS 4 MÉTODOS PARA PASO k = 3 y k = 5
    # -------------------------------------------------------------------------
    for paso in [3, 5]:
        red_saltos = reducir_por_saltos(imagen, paso)
        red_promedio = reducir_por_promedio(imagen, paso)
        red_mediana = reducir_por_mediana(imagen, paso)
        red_gaussiana = reducir_por_gaussiana(imagen, paso)

        h_red, w_red = red_saltos.shape[:2]
        pix_red = h_red * w_red
        ahorro = (1.0 - (pix_red / total_pixeles_orig)) * 100.0

        fig, axs = plt.subplots(1, 5, figsize=(20, 5), facecolor='white')
        
        # 1. Original
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"1. Original\n{ancho_orig}x{alto_orig} px\n({total_pixeles_orig:,} px - 100%)", fontsize=11, fontweight='bold', color='#0f172a')
        axs[0].axis('off')

        # 2. Saltos
        axs[1].imshow(cv2.cvtColor(red_saltos, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"2. Submuestreo (Saltos)\n{w_red}x{h_red} px (Paso {paso}x{paso})\n(Ahorro: {ahorro:.1f}%)", fontsize=11, fontweight='bold', color='#1e3a8a')
        axs[1].axis('off')

        # 3. Promedio
        axs[2].imshow(cv2.cvtColor(red_promedio, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f"3. Promedio (Redundancia)\n{w_red}x{h_red} px (Mean Pooling)\n(Ahorro: {ahorro:.1f}%)", fontsize=11, fontweight='bold', color='#065f46')
        axs[2].axis('off')

        # 4. Mediana
        axs[3].imshow(cv2.cvtColor(red_mediana, cv2.COLOR_BGR2RGB))
        axs[3].set_title(f"4. Filtro Mediana\n{w_red}x{h_red} px (Antirruido)\n(Ahorro: {ahorro:.1f}%)", fontsize=11, fontweight='bold', color='#92400e')
        axs[3].axis('off')

        # 5. Gaussiana
        axs[4].imshow(cv2.cvtColor(red_gaussiana, cv2.COLOR_BGR2RGB))
        axs[4].set_title(f"5. Ponderación Gaussiana\n{w_red}x{h_red} px (Suavizado)\n(Ahorro: {ahorro:.1f}%)", fontsize=11, fontweight='bold', color='#6b21a8')
        axs[4].axis('off')

        plt.suptitle(f"Evaluación Comparativa de Métodos de Reducción Matricial sin Resize (Paso k = {paso})", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        ruta_fig1 = os.path.join(carpeta_salida_informe, f"comparativa_paso_{paso}.png")
        plt.savefig(ruta_fig1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" [OK] Figura guardada: {ruta_fig1}")

    # -------------------------------------------------------------------------
    # FIGURA 2: DIAGRAMA DIDÁCTICO DEL BLOQUE MATRICIAL Y OPERACIÓN ARITMÉTICA
    # -------------------------------------------------------------------------
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    k = 3
    bloque = gris[0:k, 0:k].astype(int)
    valores_flat = bloque.flatten().tolist()
    p_salto = int(bloque[0, 0])
    p_prom = int(round(sum(valores_flat) / 9.0))
    p_med = int(round(float(np.median(valores_flat))))
    kernel = generar_kernel_gaussiano(3)
    p_gauss = int(round(np.clip(np.sum(bloque * kernel), 0, 255)))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Título
    ax.text(5, 5.6, "Mapeo Matricial Discreto: Bloque 3x3 a Píxel Resultante Único", fontsize=13, fontweight='bold', ha='center', color='#0f172a')
    ax.text(5, 5.25, "Demostración didáctica del paso de muestreo espacial y aprovechamiento de redundancia", fontsize=9.5, ha='center', color='#64748b', style='italic')

    # Cuadrícula 3x3
    x_base = 0.8
    y_base = 1.3
    tam_c = 0.95
    for f in range(k):
        for c in range(k):
            val = bloque[f, c]
            rect = patches.Rectangle((x_base + c * tam_c, y_base + (2 - f) * tam_c), tam_c, tam_c,
                                     linewidth=1.5, edgecolor='#3b82f6', facecolor='#e0f2fe' if (f==0 and c==0) else '#f1f5f9')
            ax.add_patch(rect)
            ax.text(x_base + c * tam_c + tam_c/2, y_base + (2 - f) * tam_c + tam_c/2, str(val),
                    fontsize=12, fontweight='bold', ha='center', va='center', color='#1e293b')

    ax.text(x_base + 1.5 * tam_c, y_base - 0.35, f"Bloque original ({k}x{k} = 9 píxeles)", ha='center', fontsize=10, fontweight='bold', color='#1e3a8a')

    # Flecha de transformación
    ax.annotate('', xy=(5.0, 2.7), xytext=(4.0, 2.7),
                arrowprops=dict(facecolor='#3b82f6', edgecolor='#1d4ed8', width=3, headwidth=10))
    ax.text(4.5, 3.1, "Operación de\nReducción", ha='center', fontsize=9.5, fontweight='bold', color='#1d4ed8')

    # Resultados a la derecha
    resultados = [
        ("Submuestreo (Saltos):", f"Píxel I(0, 0) = {p_salto}", "#1e3a8a", 4.1),
        ("Promedio (Pooling):", f"Media de los 9 px = {p_prom}", "#065f46", 3.2),
        ("Filtro Mediana:", f"Valor central = {p_med}", "#92400e", 2.3),
        ("Filtro Gaussiano:", f"Ponderación 2D = {p_gauss}", "#6b21a8", 1.4)
    ]

    for etiqueta, desc, color, y_pos in resultados:
        rect_res = patches.Rectangle((5.5, y_pos - 0.35), 4.1, 0.7,
                                     linewidth=1.2, edgecolor=color, facecolor='#f8fafc')
        ax.add_patch(rect_res)
        ax.text(5.7, y_pos, etiqueta, fontsize=10, fontweight='bold', color=color, va='center')
        ax.text(7.6, y_pos, desc, fontsize=10, fontweight='bold', color='#0f172a', va='center')

    ruta_fig2 = os.path.join(carpeta_salida_informe, "matriz_bloque_explicacion.png")
    plt.savefig(ruta_fig2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Figura guardada: {ruta_fig2}")

    # -------------------------------------------------------------------------
    # FIGURA 3: GRÁFICO DE BARRAS DE REDUCCIÓN DE DATOS Y PÍXELES PARA k = 3, 5, 7
    # -------------------------------------------------------------------------
    pasos_evaluados = [3, 5, 7]
    pixeles_totales = []
    porcentajes_ahorro = []
    etiquetas = []

    for p in pasos_evaluados:
        h_n = alto_orig // p
        w_n = ancho_orig // p
        tot = h_n * w_n
        ah = (1.0 - (tot / total_pixeles_orig)) * 100.0
        pixeles_totales.append(tot)
        porcentajes_ahorro.append(ah)
        etiquetas.append(f"Paso {p}x{p}\n({w_n}x{h_n})")

    fig, ax1 = plt.subplots(figsize=(8, 4.8), facecolor='white')

    x = np.arange(len(etiquetas))
    ancho_barra = 0.38

    color_barras = '#3b82f6'
    barras = ax1.bar(x - ancho_barra/2, [p / 1000.0 for p in pixeles_totales], ancho_barra, label='Miles de Píxeles (kpx)', color=color_barras, edgecolor='#1d4ed8', zorder=3)
    ax1.set_ylabel('Total de Píxeles (en Miles)', color='#1e3a8a', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas, fontsize=10.5, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    # Añadir valores sobre barras
    for rect in barras:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f} kpx", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2 = ax1.twinx()
    color_ahorro = '#10b981'
    linea = ax2.plot(x + ancho_barra/2, porcentajes_ahorro, color=color_ahorro, marker='o', linewidth=2.5, markersize=8, label='% de Ahorro Espacial', zorder=4)
    ax2.set_ylabel('% de Reducción / Ahorro de Memoria', color='#047857', fontsize=11, fontweight='bold')
    ax2.set_ylim(80, 100)

    for i, txt in enumerate(porcentajes_ahorro):
        ax2.annotate(f"{txt:.2f}%", (x[i] + ancho_barra/2, txt), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=9.5, fontweight='bold', color='#047857')

    plt.title(f"Ahorro de Memoria y Disminución de Píxeles según Paso Impar (Original: {ancho_orig}x{alto_orig} px)", fontsize=11.5, fontweight='bold', pad=14)
    plt.tight_layout()

    ruta_fig3 = os.path.join(carpeta_salida_informe, "grafico_ahorro_datos.png")
    plt.savefig(ruta_fig3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Figura guardada: {ruta_fig3}")

    # -------------------------------------------------------------------------
    # FIGURA 4: ROBUSTEZ ANTIRRUIDO FRENTE A IMNOISE (SALT & PEPPER)
    # -------------------------------------------------------------------------
    paso_ruido = 3
    img_ruidosa = imnoise(imagen, tipo='salt & pepper', amount=0.06)
    red_saltos_r = reducir_por_saltos(img_ruidosa, paso_ruido)
    red_promedio_r = reducir_por_promedio(img_ruidosa, paso_ruido)
    red_mediana_r = reducir_por_mediana(img_ruidosa, paso_ruido)
    red_gauss_r = reducir_por_gaussiana(img_ruidosa, paso_ruido)

    fig, axs = plt.subplots(1, 5, figsize=(20, 5), facecolor='white')
    axs[0].imshow(cv2.cvtColor(img_ruidosa, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"1. Entrada con imnoise()\n(Salt & Pepper 6%)", fontsize=11, fontweight='bold', color='#b45309')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(red_saltos_r, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"2. Saltos k={paso_ruido}\n(Puntos corruptos)", fontsize=11, fontweight='bold', color='#1e3a8a')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(red_promedio_r, cv2.COLOR_BGR2RGB))
    axs[2].set_title(f"3. Promedio k={paso_ruido}\n(Manchas difusas)", fontsize=11, fontweight='bold', color='#065f46')
    axs[2].axis('off')

    axs[3].imshow(cv2.cvtColor(red_mediana_r, cv2.COLOR_BGR2RGB))
    axs[3].set_title(f"4. Mediana k={paso_ruido}\n(★ RUIDO ELIMINADO)", fontsize=11, fontweight='bold', color='#15803d')
    axs[3].axis('off')

    axs[4].imshow(cv2.cvtColor(red_gauss_r, cv2.COLOR_BGR2RGB))
    axs[4].set_title(f"5. Gaussiana k={paso_ruido}\n(Artefactos atenuados)", fontsize=11, fontweight='bold', color='#6b21a8')
    axs[4].axis('off')

    plt.suptitle("Demostración Científica: Inmunidad del Filtro Mediana ante Ruido Impulsivo imnoise('salt & pepper')", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    ruta_fig4 = os.path.join(carpeta_salida_informe, "figura_robustez_ruido_imnoise.png")
    plt.savefig(ruta_fig4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Figura guardada: {ruta_fig4}")


# =============================================================================
# 8. MENÚ Y EJECUCIÓN PRINCIPAL
# =============================================================================
def main():
    print("=" * 70)
    print("   UNIVERSIDAD TÉCNICA DE AMBATO - CARRERA DE SOFTWARE")
    print("   PROCESAMIENTO DE IMÁGENES: REDUCCIÓN MATRICIAL PURA")
    print("=" * 70)

    # Determinar rutas
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    dir_raiz = os.path.dirname(dir_actual)
    ruta_imagen_def = os.path.join(dir_raiz, "imagenes", "imagen_ejemplo.jpg")
    carpeta_informe_img = os.path.join(dir_raiz, "informe", "imagenes")

    if not os.path.exists(ruta_imagen_def):
        ruta_imagen_def = os.path.join(dir_actual, "imagen_ejemplo.jpg")

    print("\nOpciones disponibles:")
    print("  1. Inspección interactiva de matrices en consola y comparativa gráfica")
    print("  2. Generar todas las figuras de alta resolución para el informe LaTeX")
    print("  3. Filtrado con Transformada de Fourier 2D (Pasa Bajos para ruido / Pasa Altos para bordes)")
    print("  4. Filtro de Acentuado 3×3 (Matriz Laplaciana [-1,-1,-1 / -1,8,-1 / -1,-1,-1])")
    print("  5. Salir")

    opcion = input("\nSeleccione opción [1, 2, 3, 4 o 5, def: 2]: ").strip() or "2"

    if opcion == "1":
        ruta_img = input(f"Ruta de la imagen [def: {ruta_imagen_def}]: ").strip() or ruta_imagen_def
        paso_str = input("Paso impar k (3 para 3x3, 5 para 5x5, 7 para 7x7) [def: 3]: ").strip() or "3"
        paso_k = int(paso_str)

        img = cv2.imread(ruta_img)
        if img is None:
            print(f"[ERROR] No se pudo leer la imagen: '{ruta_img}'")
            return

        aplicar_ruido_opt = input("\n¿Desea inyectar ruido sal y pimienta con imnoise() de biblioteca? (s/N): ").strip().lower() == 's'
        if aplicar_ruido_opt:
            densidad_str = input("Densidad de ruido [def: 0.05 / 5%]: ").strip() or "0.05"
            densidad = float(densidad_str)
            img = imnoise(img, tipo='salt & pepper', amount=densidad)
            print(f"[OK] Ruido inyectado con imnoise('salt & pepper', amount={densidad}) vía scikit-image.")

        imprimir_inspeccion_bloque(img, paso_k)

        # Mostrar comparativa con matplotlib
        red_saltos = reducir_por_saltos(img, paso_k)
        red_promedio = reducir_por_promedio(img, paso_k)
        red_mediana = reducir_por_mediana(img, paso_k)
        red_gaussiana = reducir_por_gaussiana(img, paso_k)
        red_acentuado = reducir_por_acentuado_manual(img, paso_k)

        fig, axs = plt.subplots(1, 6, figsize=(22, 4.5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original {'(ruido)' if aplicar_ruido_opt else ''}\n({img.shape[1]}x{img.shape[0]})", fontweight='bold')
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(red_saltos, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Saltos k={paso_k}", fontweight='bold')
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(red_promedio, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f"Promedio k={paso_k}", fontweight='bold')
        axs[2].axis('off')

        axs[3].imshow(cv2.cvtColor(red_mediana, cv2.COLOR_BGR2RGB))
        axs[3].set_title(f"Mediana k={paso_k}", fontweight='bold')
        axs[3].axis('off')

        axs[4].imshow(cv2.cvtColor(red_gaussiana, cv2.COLOR_BGR2RGB))
        axs[4].set_title(f"Gaussiana k={paso_k}", fontweight='bold')
        axs[4].axis('off')

        axs[5].imshow(cv2.cvtColor(red_acentuado, cv2.COLOR_BGR2RGB))
        axs[5].set_title(f"Acentuado k={paso_k}\n(Kernel 3x3)", fontweight='bold', color='#ec4899')
        axs[5].axis('off')

        plt.suptitle(f"Reducción Manual de Imagen - Paso k = {paso_k} (Incluye Acentuado)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    elif opcion == "2":
        generar_graficos_informe(ruta_imagen_def, carpeta_informe_img)
        # Generar también la figura de Fourier
        ruta_fig_fourier = os.path.join(carpeta_informe_img, "demostracion_fourier_filtros.png")
        demostrar_fourier(ruta_imagen_def, radio_corte=40.0, aplicar_ruido_previo=True, ruta_guardar=ruta_fig_fourier, mostrar=False)
        
        # Generar figura de demostración de Acentuado
        img_base = cv2.imread(ruta_imagen_def)
        if img_base is not None:
            img_bordes = aplicar_filtro_acentuado(img_base, modo='bordes')
            img_acent = aplicar_filtro_acentuado(img_base, modo='acentuado')
            fig_ac, axs_ac = plt.subplots(1, 3, figsize=(16, 5))
            axs_ac[0].imshow(cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB))
            axs_ac[0].set_title("1. Imagen Original", fontweight='bold')
            axs_ac[0].axis('off')
            axs_ac[1].imshow(cv2.cvtColor(img_bordes, cv2.COLOR_BGR2RGB))
            axs_ac[1].set_title("2. Bordes Laplaciano 3x3 (∇²)\n[-1 -1 -1 / -1 8 -1 / -1 -1 -1]", fontweight='bold', color='#a855f7')
            axs_ac[1].axis('off')
            axs_ac[2].imshow(cv2.cvtColor(img_acent, cv2.COLOR_BGR2RGB))
            axs_ac[2].set_title("3. Imagen Acentuada (f + ∇²)\n★ Realce de Contornos", fontweight='bold', color='#ec4899')
            axs_ac[2].axis('off')
            plt.suptitle("Filtro Espacial de Acentuado mediante Máscara Laplaciana de 8 Vecinos", fontsize=13, fontweight='bold')
            plt.tight_layout()
            ruta_fig_ac = os.path.join(carpeta_informe_img, "demostracion_filtro_acentuado.png")
            plt.savefig(ruta_fig_ac, dpi=300, bbox_inches='tight')
            plt.close()
            print(f" [OK] Figura guardada: {ruta_fig_ac}")

        print("\n[ÉXITO] Todas las figuras científicas fueron generadas en 'informe/imagenes/' correctamente.")

    elif opcion == "3":
        ruta_img = input(f"Ruta de la imagen [def: {ruta_imagen_def}]: ").strip() or ruta_imagen_def
        radio_str = input("Radio de corte D0 [def: 40]: ").strip() or "40"
        radio_corte = float(radio_str)
        con_ruido = input("¿Inyectar ruido con imnoise() para mitigar? (S/n): ").strip().lower() != 'n'
        demostrar_fourier(ruta_img, radio_corte=radio_corte, aplicar_ruido_previo=con_ruido, mostrar=True)

    elif opcion == "4":
        ruta_img = input(f"Ruta de la imagen [def: {ruta_imagen_def}]: ").strip() or ruta_imagen_def
        img = cv2.imread(ruta_img)
        if img is None:
            print(f"[ERROR] No se pudo leer la imagen: '{ruta_img}'")
            return
            
        print("\n[+] Aplicando Filtro de Acentuado 3x3:")
        print("    Kernel Laplaciano K:")
        print("    [ -1  -1  -1 ]")
        print("    [ -1   8  -1 ]")
        print("    [ -1  -1  -1 ]")
        
        img_bordes = aplicar_filtro_acentuado(img, modo='bordes')
        img_acent = aplicar_filtro_acentuado(img, modo='acentuado')
        
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("1. Original", fontweight='bold')
        axs[0].axis('off')
        
        axs[1].imshow(cv2.cvtColor(img_bordes, cv2.COLOR_BGR2RGB))
        axs[1].set_title("2. Laplaciano ∇² (Bordes)\nK = [-1 -1 -1 / -1 8 -1 / -1 -1 -1]", fontweight='bold', color='#a855f7')
        axs[1].axis('off')
        
        axs[2].imshow(cv2.cvtColor(img_acent, cv2.COLOR_BGR2RGB))
        axs[2].set_title("3. Imagen Acentuada (f + ∇²)\n★ Contornos y Detalles Realzados", fontweight='bold', color='#ec4899')
        axs[2].axis('off')
        
        plt.suptitle("Demostración del Filtro de Acentuado Espacial", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    else:
        print("Saliendo...")


if __name__ == "__main__":
    main()
