"""
=============================================================================
PROCESAMIENTO DE IMÁGENES - INTELIGENCIA ARTIFICIAL
Tema: Reducción de Imágenes sin usar funciones integradas (cv2.resize / imresize)
Autor: Estudiante de IA
=============================================================================
Descripción:
Este script implementa dos algoritmos fundamentales para la reducción de imágenes:
1. Submuestreo por Saltos (Decimación espacial / Stride sampling)
2. Promedio de Bloques Redundantes (Reducción por media aritmética / Mean Pooling)

Permite procesar imágenes completas e inspeccionar/interactuar directamente con 
las matrices numéricas de píxeles (0 a 255).
=============================================================================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# =============================================================================
# MÉTODO 1: REDUCCIÓN POR SUBMUESTREO (SALTOS DE PÍXEL EN PÍXEL)
# =============================================================================
def reducir_por_saltos_manual(imagen: np.ndarray, paso: int) -> np.ndarray:
    """
    Reduce la imagen tomando muestras cada 'paso' píxeles en filas y columnas.
    Implementado con bucles explícitos para demostrar el algoritmo básico.
    """
    alto_orig, ancho_orig = imagen.shape[:2]
    alto_nuevo = alto_orig // paso
    ancho_nuevo = ancho_orig // paso

    if imagen.ndim == 2:  # Escala de grises o matriz 2D
        reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                # Se salta i*paso en filas y j*paso en columnas
                reducida[i, j] = imagen[i * paso, j * paso]
    else:  # Imagen a color (RGB / BGR)
        canales = imagen.shape[2]
        reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                for c in range(canales):
                    reducida[i, j, c] = imagen[i * paso, j * paso, c]

    return reducida


# =============================================================================
# MÉTODO 2: REDUCCIÓN POR PROMEDIO DE BLOQUES REDUNDANTES (POOLING ARITMÉTICO)
# =============================================================================
def reducir_por_promedio_manual(imagen: np.ndarray, paso: int) -> np.ndarray:
    """
    Aprovecha la redundancia de los píxeles vecinos.
    Toma un bloque de tamaño (paso x paso) y calcula su promedio aritmético
    para representarlo con un único píxel en la imagen reducida.
    """
    alto_orig, ancho_orig = imagen.shape[:2]
    alto_nuevo = alto_orig // paso
    ancho_nuevo = ancho_orig // paso
    total_pixeles_bloque = paso * paso

    if imagen.ndim == 2:  # Escala de grises o matriz 2D
        reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                suma = 0.0
                for di in range(paso):
                    for dj in range(paso):
                        suma += float(imagen[i * paso + di, j * paso + dj])
                promedio = int(round(suma / total_pixeles_bloque))
                reducida[i, j] = np.clip(promedio, 0, 255)
    else:  # Color (3 canales)
        canales = imagen.shape[2]
        reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                for c in range(canales):
                    suma = 0.0
                    for di in range(paso):
                        for dj in range(paso):
                            suma += float(imagen[i * paso + di, j * paso + dj, c])
                    promedio = int(round(suma / total_pixeles_bloque))
                    reducida[i, j, c] = np.clip(promedio, 0, 255)

    return reducida


# =============================================================================
# MÉTODO 3: REDUCCIÓN POR FILTRO MEDIANA
# =============================================================================
def reducir_por_mediana_manual(imagen: np.ndarray, paso: int) -> np.ndarray:
    """Toma la mediana de los valores del bloque paso x paso."""
    alto_orig, ancho_orig = imagen.shape[:2]
    alto_nuevo = alto_orig // paso
    ancho_nuevo = ancho_orig // paso

    if imagen.ndim == 2:
        reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                bloque = imagen[i * paso : (i + 1) * paso, j * paso : (j + 1) * paso]
                reducida[i, j] = int(round(float(np.median(bloque))))
    else:
        canales = imagen.shape[2]
        reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                for c in range(canales):
                    bloque = imagen[i * paso : (i + 1) * paso, j * paso : (j + 1) * paso, c]
                    reducida[i, j, c] = int(round(float(np.median(bloque))))
    return reducida


# =============================================================================
# MÉTODO 4: REDUCCIÓN POR MÁSCARA GAUSSIANA PONDERADA
# =============================================================================
def reducir_por_gaussiana_manual(imagen: np.ndarray, paso: int) -> np.ndarray:
    """Aplica una ponderación gaussiana al bloque paso x paso."""
    alto_orig, ancho_orig = imagen.shape[:2]
    alto_nuevo = alto_orig // paso
    ancho_nuevo = ancho_orig // paso

    # Crear kernel gaussiano normalizado
    sigma = paso / 3.0
    ax = np.linspace(-(paso - 1) / 2.0, (paso - 1) / 2.0, paso)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)

    if imagen.ndim == 2:
        reducida = np.zeros((alto_nuevo, ancho_nuevo), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                bloque = imagen[i * paso : (i + 1) * paso, j * paso : (j + 1) * paso].astype(np.float32)
                val = np.sum(bloque * kernel)
                reducida[i, j] = int(round(np.clip(val, 0, 255)))
    else:
        canales = imagen.shape[2]
        reducida = np.zeros((alto_nuevo, ancho_nuevo, canales), dtype=np.uint8)
        for i in range(alto_nuevo):
            for j in range(ancho_nuevo):
                for c in range(canales):
                    bloque = imagen[i * paso : (i + 1) * paso, j * paso : (j + 1) * paso, c].astype(np.float32)
                    val = np.sum(bloque * kernel)
                    reducida[i, j, c] = int(round(np.clip(val, 0, 255)))
    return reducida


# =============================================================================
# INSPECTOR VISUAL DE MATRICES NUMÉRICAS EN CONSOLA
# =============================================================================
def mostrar_matrices_consola(orig: np.ndarray, saltos: np.ndarray, promedio: np.ndarray, paso: int, tam: int = 6):
    """
    Muestra en consola las matrices numéricas reales de los píxeles
    y el desglose aritmético paso a paso de los bloques.
    """
    # Si es imagen a color, tomamos el primer canal para mostrar números claros
    if orig.ndim == 3:
        m_orig = orig[:tam, :tam, 0]
        m_saltos = saltos[:tam // paso, :tam // paso, 0]
        m_prom = promedio[:tam // paso, :tam // paso, 0]
        etiqueta_canal = " (Canal Azul / Canal 0)"
    else:
        m_orig = orig[:tam, :tam]
        m_saltos = saltos[:tam // paso, :tam // paso]
        m_prom = promedio[:tam // paso, :tam // paso]
        etiqueta_canal = ""

    h_sub, w_sub = m_orig.shape

    print("\n" + "=" * 70)
    print(f" [INSPECCION MATRICIAL: VALORES NUMERICOS DE LOS PIXELES{etiqueta_canal}]")
    print("=" * 70)
    
    print(f"\n[1] MATRIZ ORIGINAL ({h_sub}x{w_sub} pixeles):")
    print("      " + " ".join([f"Col_{j:<3}" for j in range(w_sub)]))
    for i in range(h_sub):
        fila_str = " ".join([f"{int(val):>6}" for val in m_orig[i]])
        print(f"Fila_{i}: {fila_str}")

    print(f"\n[2] MATRIZ REDUCIDA POR SALTOS ({h_sub // paso}x{w_sub // paso} con paso k={paso}):")
    print(f"    (Se toman directamente los indices [i*{paso}, j*{paso}])")
    for i in range(h_sub // paso):
        fila_str = " ".join([f"{int(val):>6}" for val in m_saltos[i]])
        print(f"Fila_{i}: {fila_str}")

    print(f"\n[3] MATRIZ REDUCIDA POR PROMEDIO ({h_sub // paso}x{w_sub // paso} con paso k={paso}):")
    print(f"    (Media aritmetica de los bloques de {paso}x{paso} pixeles redundantes)")
    for i in range(h_sub // paso):
        fila_str = " ".join([f"{int(val):>6}" for val in m_prom[i]])
        print(f"Fila_{i}: {fila_str}")

    # Mostrar desglose aritmético del bloque (0, 0)
    print("\n>>> DESGLOSE MATEMATICO DEL BLOQUE SUPERIOR IZQUIERDO (0, 0):")
    bloque_00 = m_orig[0:paso, 0:paso]
    p_salto = m_orig[0, 0]
    p_prom = m_prom[0, 0]
    elementos = [str(int(x)) for x in bloque_00.flatten()]
    suma = int(np.sum(bloque_00))
    n = paso * paso
    
    print(f"  * Por Saltos:   Pixel tomado en (0, 0) = {p_salto}")
    print(f"  * Por Promedio: Pixeles del bloque = [{', '.join(elementos)}]")
    print(f"                  Suma = {suma}")
    print(f"                  Promedio = {suma}/{n} = {suma/n:.2f} -> {p_prom}")
    print("=" * 70 + "\n")


# =============================================================================
# MODO DE MATRIZ PERSONALIZADA EN CONSOLA
# =============================================================================
def modo_matriz_manual():
    print("\n" + "=" * 65)
    print("   MODO 2: SIMULADOR DE MATRIZ NUMÉRICA MANUAL")
    print("=" * 65)
    print("Permite al ingeniero definir una pequeña matriz y ver cómo se reduce.\n")
    
    try:
        tam = int(input("Ingrese tamaño de la matriz cuadrada (ej. 6 para 6x6, 9 para 9x9) [def: 9]: ").strip() or "9")
    except ValueError:
        tam = 9

    try:
        paso = int(input(f"¿De cuánto por cuánto quiere saltar? (impar: 3, 5, 7) [def: 3]: ").strip() or "3")
    except ValueError:
        paso = 3

    print(f"\nGenerando matriz de prueba de {tam}x{tam} con bloques redundantes...")
    # Crear matriz con redundancia evidente
    matriz = np.zeros((tam, tam), dtype=np.uint8)
    for bi in range(0, tam, paso):
        for bj in range(0, tam, paso):
            base = int((bi * 40 + bj * 30 + 50) % 230)
            for di in range(min(paso, tam - bi)):
                for dj in range(min(paso, tam - bj)):
                    matriz[bi + di, bj + dj] = np.clip(base + (di + dj) * 3, 0, 255)

    saltos = reducir_por_saltos_manual(matriz, paso)
    promedio = reducir_por_promedio_manual(matriz, paso)

    mostrar_matrices_consola(matriz, saltos, promedio, paso, tam=tam)


# =============================================================================
# ESTADÍSTICAS Y GRÁFICO COMPARATIVO
# =============================================================================
def imprimir_estadisticas(nombre: str, orig: np.ndarray, reducida: np.ndarray, paso: int):
    h_o, w_o = orig.shape[:2]
    h_r, w_r = reducida.shape[:2]
    pix_o = h_o * w_o
    pix_r = h_r * w_r
    reduccion_pct = (1.0 - (pix_r / pix_o)) * 100.0

    print("-" * 65)
    print(f" RESULTADOS: {nombre.upper()}")
    print("-" * 65)
    print(f"- Factor de salto (k):             {paso} (salta de {paso} en {paso} pixeles)")
    print(f"- Dimensiones Originales:          {w_o} x {h_o} (Ancho x Alto)")
    print(f"- Dimensiones Reducidas:           {w_r} x {h_r} (Ancho x Alto)")
    print(f"- Total de pixeles originales:     {pix_o:,} px")
    print(f"- Total de pixeles reducidos:      {pix_r:,} px")
    print(f"- Reduccion de datos / pixeles:    {reduccion_pct:.2f}% menos datos")
    print("-" * 65)


def mostrar_comparativa(img_original: np.ndarray, img_saltos: np.ndarray, img_promedio: np.ndarray, paso: int):
    es_color = img_original.ndim == 3
    if es_color:
        rgb_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        rgb_saltos = cv2.cvtColor(img_saltos, cv2.COLOR_BGR2RGB)
        rgb_promedio = cv2.cvtColor(img_promedio, cv2.COLOR_BGR2RGB)
        cmap = None
    else:
        rgb_orig = img_original
        rgb_saltos = img_saltos
        rgb_promedio = img_promedio
        cmap = "gray"

    h_o, w_o = img_original.shape[:2]
    h_r, w_r = img_saltos.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    try:
        fig.canvas.manager.set_window_title(f"Reducción de Imagen (Paso = {paso}) - Sin Resize")
    except Exception:
        pass

    axes[0].imshow(rgb_orig, cmap=cmap)
    axes[0].set_title(f"1. ORIGINAL\n{w_o} x {h_o} px\n(100% de píxeles)", fontsize=11, fontweight="bold", color="#1E3A8A")
    axes[0].axis("on")

    axes[1].imshow(rgb_saltos, cmap=cmap)
    axes[1].set_title(f"2. POR SALTOS (Submuestreo)\n{w_r} x {h_r} px (paso={paso})\n(Píxeles tomados cada {paso})", fontsize=11, fontweight="bold", color="#047857")
    axes[1].axis("on")

    axes[2].imshow(rgb_promedio, cmap=cmap)
    axes[2].set_title(f"3. POR PROMEDIO (Redundancia)\n{w_r} x {h_r} px (bloques {paso}x{paso})\n(Media de píxeles redundantes)", fontsize=11, fontweight="bold", color="#B45309")
    axes[2].axis("on")

    plt.suptitle(f"Demostración de Reducción de Imagen sin función 'resize' | Factor de salto = {paso}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    os.makedirs("imagenes", exist_ok=True)
    nombre_comparativa = os.path.join("imagenes", f"comparativa_paso_{paso}.png")
    plt.savefig(nombre_comparativa, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Gráfico comparativo guardado como: '{nombre_comparativa}'")
    
    try:
        plt.show()
    except Exception:
        pass


# =============================================================================
# PROGRAMA PRINCIPAL CON MENÚ
# =============================================================================
def main():
    print("=" * 65)
    print("   DEBER DE IA: REDUCCIÓN DE IMÁGENES Y ANÁLISIS MATRICIAL")
    print("=" * 65)
    print("Seleccione una opción:")
    print("  1. Procesar imagen de archivo (con inspección de matriz numérica)")
    print("  2. Modo simulador de matriz numérica (para interactuar con números)")
    
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        opcion = "1"
    else:
        opcion = input("\nIngrese opción [1 o 2, def: 1]: ").strip() or "1"

    if opcion == "2":
        modo_matriz_manual()
        return

    # Opción 1: Procesar imagen
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        ruta_imagen = sys.argv[1]
    else:
        ruta_default = os.path.join("imagenes", "imagen_ejemplo.jpg")
        if not os.path.exists(ruta_default):
            ruta_default = "imagen_ejemplo.jpg"
        entrada = input(f"\nIngrese la ruta de la imagen [Presione ENTER para '{ruta_default}']: ").strip()
        ruta_imagen = entrada if entrada else ruta_default

    if not os.path.exists(ruta_imagen):
        print(f"[ERROR] No se encontró el archivo: '{ruta_imagen}'")
        return

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"[ERROR] No se pudo decodificar la imagen: '{ruta_imagen}'")
        return

    alto_orig, ancho_orig = imagen.shape[:2]
    print(f"\n[OK] Imagen cargada: '{ruta_imagen}' ({ancho_orig}x{alto_orig} px)")

    while True:
        try:
            if len(sys.argv) > 2:
                paso = int(sys.argv[2])
            else:
                entrada_paso = input("\n¿De cuánto por cuánto quiere saltar? (impar: 3 para 3x3, 5 para 5x5, 7 para 7x7) [def: 3]: ").strip()
                paso = int(entrada_paso) if entrada_paso else 3

            if paso < 1:
                print("El paso debe ser al menos 1.")
                continue
            if paso > min(alto_orig, ancho_orig):
                print(f"El paso no puede ser mayor a {min(alto_orig, ancho_orig)}.")
                continue
            break
        except ValueError:
            print("Ingrese un número entero válido.")

    print(f"\n>>> Reduciendo imagen con salto k = {paso}...")
    reducida_saltos = reducir_por_saltos_manual(imagen, paso)
    reducida_promedio = reducir_por_promedio_manual(imagen, paso)

    # Mostrar la inspección numérica de matrices en la consola
    mostrar_matrices_consola(imagen, reducida_saltos, reducida_promedio, paso, tam=6)

    # Imprimir estadísticas
    imprimir_estadisticas("Método 1: Submuestreo por Saltos", imagen, reducida_saltos, paso)
    imprimir_estadisticas("Método 2: Promedio de Redundancia", imagen, reducida_promedio, paso)

    # Guardar imágenes en la carpeta 'imagenes/'
    os.makedirs("imagenes", exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(ruta_imagen))
    ruta_salida_saltos = os.path.join("imagenes", f"{base}_reducida_saltos_x{paso}.png")
    ruta_salida_prom = os.path.join("imagenes", f"{base}_reducida_promedio_x{paso}.png")
    cv2.imwrite(ruta_salida_saltos, reducida_saltos)
    cv2.imwrite(ruta_salida_prom, reducida_promedio)
    print(f"[OK] Imágenes guardadas en la carpeta 'imagenes/' exitosamente.")

    print("\nAbriendo gráfico comparativo...")
    mostrar_comparativa(imagen, reducida_saltos, reducida_promedio, paso)


if __name__ == "__main__":
    main()
