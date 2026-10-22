# 🖼️ Reducción de Imágenes sin `imresize` ni `cv2.resize`
### Procesamiento Digital de Imágenes | Inteligencia Artificial

Proyecto desarrollado para demostrar el principio fundamental del **muestreo espacial**, la **redundancia de píxeles** y la **manipulación directa de matrices**, sin recurrir a funciones automáticas de interpolación de caja negra como `imresize` (MATLAB) o `cv2.resize` (OpenCV).

---

## 📁 Estructura del Proyecto Organizada

```text
ReducirImagen/
├── app.py                # Servidor Web y API (Flask) con los algoritmos matemáticos en NumPy
├── reducir_imagen.py     # Script ejecutable en consola con menú interactivo y gráficos
├── templates/
│   └── index.html        # Interfaz Web moderna y reactiva (Alpine.js + CSS Moderno)
├── imagenes/             # Carpeta centralizada de imágenes y resultados
│   ├── imagen_ejemplo.jpg
│   ├── comparativa_paso_2.png
│   ├── imagen_ejemplo_reducida_saltos_x2.png
│   └── imagen_ejemplo_reducida_promedio_x2.png
└── README.md             # Documentación unificada, guía de uso y sustento teórico
```

---

## 🚀 Cómo Ejecutar

### Opción 1: Aplicación Web Interactiva (Recomendada)
La interfaz web permite visualizar la imagen, seleccionar los saltos impares ($3\times3$, $5\times5$, $7\times7$), cambiar el método en tiempo real e inspeccionar/editar los valores numéricos de la matriz de píxeles:
```bash
python app.py
```
*Se abrirá automáticamente en tu navegador (`http://127.0.0.1:5050`):*
- **Panel Izquierdo:** Imagen original y resultado modificado frente a frente.
- **Panel Derecho:** Selector de paso impar ($k = 3, 5, 7$), método de reducción, **matriz de píxeles editable en vivo** ($0$ a $255$), estadísticas de ahorro y botón para descargar en PNG.

> **¿Por qué matrices impares ($3\times3, 5\times5, 7\times7$)?**
> 1. **Píxel central exacto:** Tienen un centro geométrico simétrico $(\frac{k-1}{2}, \frac{k-1}{2})$.
> 2. **Mediana pura sin promedios:** Al tener un número impar de celdas ($9, 25, 49$), la mediana es un píxel real exacto de la muestra.
> 3. **Convolución Gaussiana balanceada:** Los pesos se distribuyen simétricamente alrededor del centro.

### Opción 2: Script de Consola (Terminal)
Permite procesar imágenes directamente por línea de comandos e inspeccionar matrices en texto:
```bash
python reducir_imagen.py
```

---

## 🧠 Marco Teórico para la Sustentación

### 1. ¿Qué es una imagen digital a nivel matricial?
Una imagen es una función discreta $f(x, y)$ representada como una matriz bidimensional (escala de grises) o tridimensional (RGB / BGR):
- Cada celda contiene la intensidad luminosa de un **píxel** entre $[0, 255]$ (`uint8`).
- Las dimensiones espaciales son $Alto \times Ancho$ píxeles.

### 2. ¿Por qué el profesor prohibió `imresize` / `cv2.resize`?
Porque esas funciones de librería aplican interpolaciones automáticas (bilineal, bicúbica, spline o Lanczos) que realizan aproximaciones numéricas complejas, ocultando el concepto didáctico que el docente busca evaluar:
> **Demostrar cómo se manipulan los índices de una matriz bidimensional y cómo opera el muestreo espacial.**

---

## ⚙️ Métodos de Reducción Implementados

### Método 1: Submuestreo por Saltos (Decimación Espacial)
> *"de cuánto por cuánto quiere saltar :v ... saco, por ejemplo, de dos en dos"*

- **Funcionamiento:** Se recorre la matriz original dando saltos de tamaño $k$ en filas y columnas.
- **Matemáticamente:**  
  $$I_{reducida}(i, j) = I_{original}(i \cdot k, j \cdot k)$$
- En Python (slicing directo): `imagen[::k, ::k]`
- **Efecto dimensional:**  
  $Alto_{nuevo} = \lfloor Alto / k \rfloor$, $Ancho_{nuevo} = \lfloor Ancho / k \rfloor$.  
  Para $k = 2$, la cantidad total de píxeles pasa a ser $\frac{1}{4}$ (25% del original), logrando un **75% de reducción de datos**.

---

### Método 2: Promedio de Bloques (Mean Pooling)
> *"¿Por qué? Porque mira, aquí yo tengo píxeles que son redundantes... ¿cuál es la diferencia de coger estos dos píxeles y hacerle uno solo en un valor aritmético?"*

- **Funcionamiento:** En vez de descartar píxeles intermedios, se toma cada bloque de $k \times k$ píxeles vecinos y se calcula su media aritmética:
  $$I_{reducida}(i, j) = \frac{1}{k^2} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} I_{original}(i \cdot k + u, j \cdot k + v)$$
- **Ventaja:** Aprovecha la redundancia espacial, preserva mejor la luminosidad promedio y minimiza el efecto de *aliasing* (bordes dentados).

---

### Métodos Adicionales: Mediana y Gaussiana
- **Mediana:** Calcula el valor central del bloque $k \times k$. Es robusto frente a ruido de tipo "sal y pimienta".
- **Gaussiana:** Convoluciona el bloque con una máscara ponderada 2D, dando más peso a los píxeles centrales y suavizando la transición.

---

## 👁️ ¿Por qué la imagen reducida no pierde visibilidad?
Debido a la **Redundancia Espacial**:
1. En las imágenes del mundo real, los objetos son continuos y uniformes (paredes, piel, cielo).
2. Los píxeles contiguos suelen tener valores de color prácticamente idénticos.
3. Al saltar o promediar píxeles adyacentes, la estructura geométrica, bordes y formas macroscópicas se conservan nítidamente ante la percepción del ojo humano.

---

## 🎓 Banco de Preguntas para el Examen / Sustentación

| Pregunta del Docente | Respuesta Clave |
|---|---|
| **¿Por qué divides las dimensiones para $k$?** | Porque al saltar de $k$ en $k$, la cantidad de muestras que caben en el nuevo arreglo es exactamente la dimensión original dividida para el paso de muestreo. |
| **Si saltamos de 2 en 2, ¿a cuánto se reduce la imagen?** | Se reduce a la **cuarta parte (25%)** del número original de píxeles, porque la reducción ocurre simultáneamente en filas y columnas: $(1/2) \times (1/2) = 1/4$. |
| **¿Qué ocurre si usamos un paso muy alto como $k = 10$?** | Se produce un sobremuestreo excesivo perdiendo frecuencias espaciales altas (detalles finos) y la imagen se pixela notablemente. |
| **¿Cuál es la diferencia entre el método de saltos y el de promedio?** | El método de saltos descarta información directamente; el método de promedio integra la información de los píxeles vecinos redundantes en un único valor medio representativo. |
| **¿Cómo se maneja una imagen a color frente a una en escala de grises?** | En escala de grises la matriz es $H \times W$ (2D). A color es un tensor $H \times W \times 3$ (canales Azul, Verde y Rojo); la reducción se aplica idénticamente a cada canal. |
