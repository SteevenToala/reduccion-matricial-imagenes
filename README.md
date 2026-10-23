# 🖼️ Reducción Matricial de Imágenes sin `imresize` ni `cv2.resize`
### Procesamiento Digital de Imágenes | Inteligencia Artificial
**Universidad Técnica de Ambato (UTA) — Facultad de Ingeniería en Sistemas, Electrónica e Industrial (FISEI)**  
**Carrera de Software — Ciclo: Octubre 2026 – Marzo 2027**  
**Docente:** Ing. Mg. Ruben Nogales | **Autores:** Toala Steeven, Chavez Michael

---

Proyecto desarrollado para demostrar rigurosamente los principios de **muestreo espacial discreto**, **manipulación de índices matriciales** y **redundancia espacial de píxeles**, prescindiendo en su totalidad de funciones automáticas de interpolación de caja negra como `imresize` (MATLAB) o `cv2.resize` (OpenCV).

---

## 📁 Estructura Completa del Proyecto

```text
ReducirImagen/
├── codigo/
│   └── main.py                     # Algoritmos matriciales vectorizados (Saltos, Promedio, Mediana, Gaussiana)
│                                   # y generador automático de figuras científicas para el informe
├── informe/
│   ├── main.tex                    # Documento formal en LaTeX con formato oficial UTA - FISEI
│   ├── main.pdf                    # Informe académico completo compilado (14 páginas)
│   ├── referencias.bib             # Bibliografía en formato IEEE (Gonzalez & Woods, Shannon, etc.)
│   └── imagenes/                   # Figuras científicas de alta resolución y logos institucionales
│       ├── uta.png                 # Escudo institucional UTA
│       ├── fisei.png               # Logotipo FISEI
│       ├── comparativa_paso_3.png  # Comparativa visual de los 4 métodos con paso 3x3
│       ├── comparativa_paso_5.png  # Comparativa visual de los 4 métodos con paso 5x5
│       ├── matriz_bloque_explicacion.png # Diagrama del bloque matricial 3x3 y síntesis de píxel
│       └── grafico_ahorro_datos.png      # Gráfica cuantitativa de ahorro de memoria y píxeles
├── templates/
│   └── index.html                  # Interfaz Web interactiva (Alpine.js + CSS Moderno)
├── imagenes/                       # Banco de imágenes de prueba y resultados
│   └── imagen_ejemplo.jpg
├── app.py                          # Servidor Web Flask con simulador de matrices numéricas en vivo
├── EXPLICACION_VISUAL_FORMULAS.md  # Guía didáctica explicada paso a paso en lenguaje claro
├── explicacion_formulas.html       # Dashboard interactivo con KaTeX y banco de preguntas
└── README.md                       # Documentación centralizada del proyecto
```

---

## 🚀 Cómo Ejecutar

### 1. Generador de Figuras y Menú de Consola (`codigo/main.py`)
Ejecuta el script principal para inspeccionar el bloque numérico en la terminal o regenerar las figuras del informe:
```bash
python codigo/main.py
```
*Opciones del menú:*
- `1`: Inspección interactiva en consola (muestra la cuadrícula de píxeles de $0$ a $255$ y la traza aritmética paso a paso) junto con visualización gráfica en Matplotlib.
- `2`: Generación automática de todas las figuras científicas en `informe/imagenes/` a $300$ DPI.

### 2. Aplicación Web Interactiva (`app.py`)
Abre una interfaz gráfica moderna en tu navegador (`http://127.0.0.1:5050`):
```bash
python app.py
```
- **Panel Izquierdo:** Comparativa visual en tiempo real entre la imagen original y la reducida.
- **Panel Derecho:**
  - Selector de pasos impares exclusivos ($3\times3, 5\times5, 7\times7$).
  - Selector de método (Saltos, Promedio, Mediana, Gaussiana).
  - **Matriz de píxeles editable en vivo:** cambia cualquier número de la cuadrícula y observa el píxel resultante al instante.
  - Estadísticas de ahorro porcentual y botón para descargar el resultado en PNG.

### 3. Compilación del Informe en LaTeX (`informe/`)
El informe ya se encuentra compilado en [`informe/main.pdf`](file:///d:/2Proyectos/INTELIGENCIA%20ARTIFICIAL/ReducirImagen/informe/main.pdf). Si deseas recompilarlo manualmente:
```bash
cd informe
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

---

## 🧠 Fundamentos Teóricos y Métodos Implementados

### ¿Por qué matrices impares ($3\times3, 5\times5, 7\times7$)?
1. **Centro Geométrico Simétrico Entero:** Una matriz impar tiene un centro exacto en $(\frac{k-1}{2}, \frac{k-1}{2})$. En $3\times3$ es $(1, 1)$, en $5\times5$ es $(2, 2)$. Las matrices pares carecen de píxel físico en el centro.
2. **Mediana Pura sin Promedios:** Al haber $k^2$ celdas (impar: $9, 25, 49$), la mediana es un **píxel real existente de la escena**.
3. **Distribución Gaussiana Isotrópica:** Permite colocar el vértice de la campana en el centro y distribuir los pesos simétricamente.

---

### Resumen Matemático de los 4 Métodos

| Método | Operación Matemática | Característica Principal |
| :--- | :--- | :--- |
| **1. Submuestreo por Saltos** | $I(i, j) = I_{\text{orig}}(i\cdot k, j\cdot k)$ | Decimación rápida; propenso a *aliasing* si no hay filtro previo. |
| **2. Promedio de Bloques** | $I(i, j) = \frac{1}{k^2} \sum_{u} \sum_{v} I_{\text{orig}}$ | Aprovecha la redundancia espacial; preserva la energía luminosa media. |
| **3. Filtro Mediana** | $I(i, j) = \text{mediana}(B_{k\times k})$ | Excelente robustez contra ruido impulsivo (sal y pimienta); bordes nítidos. |
| **4. Ponderación Gaussiana** | $I(i, j) = \sum_{u} \sum_{v} I_{\text{orig}} \cdot K(u, v)$ | Suavizado natural imitando el sistema visual humano; mayor peso al centro. |

---

## 📊 Tabla Cuantitativa de Ahorro y Reducción

$$\text{Factor de Escala Bidimensional} = \frac{1}{k} \times \frac{1}{k} = \frac{1}{k^2}$$

| Configuración | Factor $k$ | Resolución | Total Píxeles | Fracción | \% Ahorro de Memoria |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | -- | $800 \times 600$ | $480.000$ | $1.0000$ | $0.00\%$ |
| **Paso $3\times3$** | $k = 3$ | $266 \times 200$ | $53.200$ | $\approx 1/9$ | **$88.92\%$** |
| **Paso $5\times5$** | $k = 5$ | $160 \times 120$ | $19.200$ | $1/25$ | **$96.00\%$** |
| **Paso $7\times7$** | $k = 7$ | $114 \times 85$ | $9.690$ | $\approx 1/49$ | **$97.98\%$** |

---

## 🎓 Banco de Preguntas para la Sustentación

1. **¿Por qué saltar de 3 en 3 reduce la imagen a la novena parte ($11.1\%$) y no a la tercera parte?**  
   *Porque la reducción ocurre simultáneamente en filas ($1/3$) y columnas ($1/3$), multiplicando ambas dimensiones: $(1/3) \times (1/3) = 1/9$.*

2. **¿Por qué la imagen sigue reconociéndose si eliminamos el $89\%$ o $96\%$ de los datos?**  
   *Por la **redundancia espacial**: los píxeles adyacentes pertenecen a superficies continuas y comparten valores lumínicos similares. El sistema visual humano reconstruye la forma a partir de las bajas frecuencias espaciales.*

3. **¿Cuál es la diferencia entre el método de saltos y el de promedio?**  
   *Saltos descarta píxeles directamente sin procesar; promedio calcula la media de los vecinos actuando como filtro de paso bajo que preserva la luminosidad y atenúa el aliasing.*

4. **¿Por qué se utilizó `imnoise('salt & pepper')` de biblioteca y no programada manualmente?**  
   *Porque las reducciones matriciales se programaron desde cero (sin `resize` / `imresize`), pero para evaluar la respuesta ante perturbaciones se utilizó la función estándar de biblioteca `skimage.util.random_noise` (equivalente oficial al `imnoise` de MATLAB). Esto garantiza una prueba científica imparcial donde se demuestra que el **Filtro Mediana** elimina el 100% del ruido impulsivo al enviar los ceros y 255 a los extremos del bloque ordenado.*
