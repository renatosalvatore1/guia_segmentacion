'''
Carga de imagenes de resonancia, aplicacion de preprosesamiento, detecta tumores, 
analiza texturas, localiza anatomicamente y los remarca con referencias a zonas 
funcionales del cerebro.  
Uso:
    python mri_tumor_analyzer.py
    → Se abre un explorador de archivos para elegir el .nii / .nii.gz
 
Estructura de carpetas esperada (puede ajustarse):
    data/
      BraTS-GLI-02185-102/
        BraTS-GLI-02185-102-t1c.nii.gz   ← contraste T1 (mejor para tumores)
        BraTS-GLI-02185-102-t2f.nii.gz   ← FLAIR (opcional)
        BraTS-GLI-02185-102-seg.nii.gz   ← máscara de referencia (opcional)

Crtl + C para cortar ejecucion
'''
import os # interactuar con el sistema operativo: buscar archivos, construir rutas, verificar si una carpeta existe
import sys #controlar la ejecución del programa (salir, manejar argumentos)
import glob #buscar archivos por patrón
import warnings#suprimir advertencias molestas que no son errores reales (librerías de imagen suelen generar muchas)
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib #leer archivos NIfTI (.nii, .nii.gz), formato estándar de resonancias. Te da los datos del volumen y la matriz affine que relaciona vóxeles con coordenadas reales en milímetros
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #dibujar formas sobre la imagen 
from matplotlib.widgets import Slider, Button , RadioButtons #los controles interactivos
from matplotlib.colors import ListedColormap #crear el mapa de color personalizado para el overlay del tumor (transparente donde no hay tumor, rojo semitransparente donde sí)
from scipy import ndimage
from scipy.signal import wiener
from skimage import filters, morphology, measure, exposure ,feature
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.feature import graycomatrix, graycoprops

'''
Atlas anatomico
basado en divisiones lobulares y zonas funcionales estandar

El volumen tiene coordenadas normalizadas de 0 a 1 en cada eje
Eje X → izquierda (0) a derecha (1)
Eje Y → posterior (0) a anterior (1)  
Eje Z → inferior (0) a superior (1)

Frontal ocupa ~el tercio anterior del cerebro  → Y: 0.5 a 1.0
Occipital está en la parte posterior           → Y: 0.0 a 0.35
Cerebelo está abajo y atrás                   → Z: 0.0 a 0.4
Tronco encefálico está en el centro inferior  → Z: 0.1 a 0.35

'''
BRAIN_ATLAS = {
    # (nombre, x_min, x_max, y_min, y_max, z_min, z_max, descripcion_funcional)
    "Lóbulo Frontal (izq)":      (0.0,  0.45, 0.5,  1.0,  0.4,  1.0,  "Motor primario, lenguaje expresivo (Broca), funciones ejecutivas, personalidad"),
    "Lóbulo Frontal (der)":      (0.55, 1.0,  0.5,  1.0,  0.4,  1.0,  "Control inhibitorio, atención, planificación"),
    "Área de Broca":             (0.55, 0.75, 0.55, 0.75, 0.35, 0.55, "⚠️ CRÍTICO: Producción del lenguaje hablado"),
    "Área de Wernicke":          (0.55, 0.75, 0.4,  0.6,  0.45, 0.65, "⚠️ CRÍTICO: Comprensión del lenguaje"),
    "Corteza Motora Primaria":   (0.2,  0.8,  0.55, 0.75, 0.5,  0.75, "⚠️ CRÍTICO: Control motor voluntario (contralateral)"),
    "Corteza Somatosensorial":   (0.2,  0.8,  0.45, 0.65, 0.5,  0.75, "⚠️ CRÍTICO: Sensibilidad táctil y propioceptiva"),
    "Lóbulo Parietal (izq)":     (0.0,  0.45, 0.35, 0.6,  0.5,  0.8,  "Integración sensorial, lectura, escritura, cálculo"),
    "Lóbulo Parietal (der)":     (0.55, 1.0,  0.35, 0.6,  0.5,  0.8,  "Orientación espacial, atención visuoespacial"),
    "Lóbulo Temporal (izq)":     (0.0,  0.4,  0.3,  0.6,  0.2,  0.5,  "⚠️ CRÍTICO: Memoria, lenguaje, audición"),
    "Lóbulo Temporal (der)":     (0.6,  1.0,  0.3,  0.6,  0.2,  0.5,  "Reconocimiento facial, música, procesamiento emocional"),
    "Lóbulo Occipital":          (0.2,  0.8,  0.0,  0.35, 0.3,  0.7,  "Procesamiento visual primario"),
    "Cerebelo":                  (0.15, 0.85, 0.0,  0.3,  0.0,  0.4,  "⚠️ CRÍTICO: Coordinación motora, equilibrio, marcha"),
    "Tronco Encefálico":         (0.35, 0.65, 0.15, 0.4,  0.1,  0.35, "⚠️ CRÍTICO: Funciones vitales (respiración, conciencia, pares craneales)"),
    "Hipocampo (izq)":           (0.15, 0.4,  0.3,  0.5,  0.25, 0.45, "⚠️ CRÍTICO: Memoria episódica, navegación espacial"),
    "Hipocampo (der)":           (0.6,  0.85, 0.3,  0.5,  0.25, 0.45, "Memoria espacial"),
    "Ganglios Basales":          (0.3,  0.7,  0.4,  0.6,  0.3,  0.55, "⚠️ CRÍTICO: Control motor, hábitos, recompensa"),
    "Cuerpo Calloso":            (0.3,  0.7,  0.45, 0.65, 0.45, 0.65, "Comunicación inter-hemisférica"),
    "Corteza Visual Primaria":   (0.2,  0.8,  0.0,  0.2,  0.35, 0.65, "⚠️ CRÍTICO: Visión"),
    "Ínsula":                    (0.1,  0.4,  0.4,  0.6,  0.3,  0.55, "Dolor, interoceptión, conciencia"),
}

CRITICAL_ZONES = {k for k, v in BRAIN_ATLAS.items() if "⚠️" in v[-1]}
'''Flujo completo de clase
MRITumorAnalyzer(filepath)
        │
        ▼
   __init__()          ← arranca todo
        │
        ├─→ _preprocess()        ← limpia la imagen
        │
        ├─→ _detect_tumor()      ← busca el tumor
        │
        ├─→ _analyze_regions()   ← mide y localiza
        │
        ├─→ _analyze_textures()  ← texturas GLCM
        │
        └─→ _print_report()      ← imprime en consola

   .show()             ← abre el visualizador (lo llamás aparte)
'''

# ═══════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL
# ═══════════════════════════════════════════════════════════════
class MRITumorAnalyzer: # empezamos definiendo la clase principal que va a manejar todo el proceso de análisis de la imagen de resonancia. 
    #Esta clase se encargará de cargar la imagen, preprocesarla, detectar tumores, analizar las regiones detectadas, extraer texturas y 
    # finalmente mostrar un reporte y una visualización interactiva.
    # la maquina se llama MRITumorAnalyzer porque es un analizador de tumores en 
    # imágenes de resonancia magnética (MRI)., que ya bien mencione que carga resonancias , precesa, detecta tumores y las muestra con referencias anatómicas.

    def __init__(self, filepath: str):#Metodo que se ejecutara automaticamente cuando se construye la maquina. Es el primerio en correr
        # y organiza todo el pipeline en orden. Para cambiar el orden de pasos, o saltear, es donde se edita
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        print(f"\n{'═'*60}")
        print(f"  Cargando: {self.filename}")
        print(f"{'═'*60}")

        img = nib.load(filepath)
        self.data_raw = img.get_fdata() # datos crudos de ESTA resonancia
        self.affine   = img.affine
        self.shape    = self.data_raw.shape

        print(f"  Dimensiones: {self.shape}")
        print(f"  Rango original: [{self.data_raw.min():.1f}, {self.data_raw.max():.1f}]")

        # Pipeline completo
        self.data_clean  = self._preprocess()
        self.tumor_mask  = self._detect_tumor()
        self.regions     = self._analyze_regions()
        self.textures    = self._analyze_textures_all_regions()
        self._print_report()

#self.data_raw    # imagen original en RAM
#self.data_clean  # imagen preprocesada
#self.tumor_mask  # máscara booleana del tumor (True/False por vóxel)
#self.brain_mask  # máscara del cerebro
#self.regions     # lista con info de cada región tumoral
#self.textures    # diccionario con rasgos GLCM por región

#Cualquier cosa guardada con self. es accesible desde 
#cualquier metodo de la clase. Es la memoria compartida de la maquina

    # ──────────────────────────────────────────
    #  1. PREPROCESAMIENTO
    # ──────────────────────────────────────────
    def _preprocess(self) -> np.ndarray: # defino la función de preprocesamiento que va a limpiar y mejorar 
        #la calidad de la imagen para facilitar la detección de tumores.
        # el self es porque esta funcion es un método de la clase MRITumorAnalyzer, y va 
        # a operar sobre los datos que ya cargamos en el __init__.
        """
        Cadena de preprocesamiento:
          1. Normalización robusta (percentil)
          2. Eliminación de cráneo (skull stripping simplificado por umbral)
          3. Filtro Wiener por corte (reducción de ruido + modelo AR implícito)
          4. Ecualización adaptativa del histograma (CLAHE)
        """
        print("\n  [1/4] Preprocesando imagen...")
        vol = self.data_raw.copy().astype(np.float64)

        # Normalización robusta (ignora outliers)
        p2, p98 = np.percentile(vol[vol > 0], [2, 98])
        vol = np.clip(vol, p2, p98)
        vol = (vol - p2) / (p98 - p2 + 1e-9)

        # Skull stripping básico: máscara de tejido cerebral
        brain_mask = vol > 0.05
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        # Mantener sólo el componente más grande (cabeza)
        labeled, n = ndimage.label(brain_mask)
        if n > 0:
            sizes = ndimage.sum(brain_mask, labeled, range(1, n+1))
            biggest = np.argmax(sizes) + 1
            brain_mask = (labeled == biggest)
        self.brain_mask = brain_mask # la mascara de ESTE cerebro la guardo como atributo para usarla 
        #en otras partes del análisis (detección de tumor, localización anatómica)

        # # 3. Filtro Gaussiano suave (reemplaza Wiener que genera halos)
    #    sigma=0.8 → suavizado leve, preserva bordes importantes
        from scipy.ndimage import gaussian_filter
        vol_filt = np.zeros_like(vol)
        for z in range(vol.shape[2]):
            sl = vol[:, :, z]
            if sl.max() > 0.01:
                vol_filt[:, :, z] = gaussian_filter(sl, sigma=0.8)
            else:
                vol_filt[:, :, z] = sl

        # 4. Sharpening suave (realza bordes sin quemar)
    #    Resta una versión más suavizada → recupera detalle
        vol_sharp = np.zeros_like(vol_filt)
        for z in range(vol.shape[2]):
            sl = vol_filt[:, :, z]
            blurred = gaussian_filter(sl, sigma=2.0)
            vol_sharp[:, :, z] = np.clip(sl + 0.3 * (sl - blurred), 0, 1)

        # Aplicar máscara cerebral
        vol_sharp *= brain_mask.astype(float)

        print("     Preprocesamiento completado ✓")
        return vol_sharp

    # ──────────────────────────────────────────
    #  2. DETECCIÓN DE TUMOR
    # ──────────────────────────────────────────
    def _detect_tumor(self) -> np.ndarray:
        """
        Pipeline de segmentación:
          1. Umbral de Otsu global → regiones hiperintensas (realce de contraste en T1c)
          2. Filtro de simetría bilateral → detecta asimetrías (característica de tumores)
          3. Morphological cleanup
          4. Análisis de componentes conectados → retiene regiones candidatas
        """
        print("  [2/4] Detectando tumor...")
        vol = self.data_clean

        # ── Umbral adaptativo por Otsu ──────────────────────────
        # Trabajo solo en vóxeles cerebrales
        brain_vals = vol[self.brain_mask]
        thresh_otsu = filters.threshold_otsu(brain_vals)

        # Umbral más exigente para detectar hiperseñal (zonas realzadas = tumor activo)
        # En T1c, el tumor realzado está entre ~85-99 percentil
        thresh_high = np.percentile(brain_vals, 87) # agarra mas zona de edema
        # Esto agarra todo lo que está arriba del 87% de brillo
        # → materia blanca normal también es brillante → falsos positivos
        thresh = max(thresh_otsu * 1.15, thresh_high)

        bright_mask = (vol > thresh) & self.brain_mask

        # ── Simetría bilateral ──────────────────────────────────
        # Voltear horizontalmente y restar → resaltar asimetrías
        vol_flip = np.flip(vol, axis=0)
        asymmetry = np.abs(vol - vol_flip)
        asym_mask = asymmetry > np.percentile(asymmetry[self.brain_mask], 90)

        # Combinar: región brillante Y asimétrica
        candidate = bright_mask & asym_mask & self.brain_mask

        # ── Gradiente de Sobel → bordes ────────────────────────
        # Excluir zonas de borde de cráneo (falsos positivos)
        edge_vol = np.zeros_like(vol)
        for z in range(vol.shape[2]):
            edge_vol[:, :, z] = filters.sobel(vol[:, :, z])
        edge_mask = edge_vol > np.percentile(edge_vol[self.brain_mask], 97)

        # Quitar bordes puros del cráneo
        candidate = candidate & ~(edge_mask & ~self._is_interior())

        # ── Limpieza morfológica ────────────────────────────────
        candidate = morphology.binary_opening(candidate, morphology.ball(2))
        candidate = ndimage.binary_fill_holes(candidate)
        candidate = morphology.binary_closing(candidate, morphology.ball(3))

        # ── Filtro de tamaño (eliminar puntos aislados) ─────────
        labeled, n_regions = ndimage.label(candidate)
        final_mask = np.zeros_like(candidate)

        min_voxels = 50   # mínimo para ser considerado región de interés
        for i in range(1, n_regions + 1):
            region = labeled == i
            size = region.sum()
            if size >= min_voxels:
                # Verificar que esté dentro del cerebro (no en los bordes)
                region_vals = vol[region]
                if region_vals.mean() > thresh * 0.85:
                    final_mask |= region

        self.n_tumor_regions = ndimage.label(final_mask)[1]
        print(f"     Regiones candidatas encontradas: {self.n_tumor_regions} ✓")
        return final_mask

    def _is_interior(self) -> np.ndarray:
        """Máscara de interior del cerebro (erosión del brain mask)."""
        return morphology.binary_erosion(self.brain_mask, morphology.ball(5))

    # ──────────────────────────────────────────
    #  3. ANÁLISIS DE REGIONES
    # ──────────────────────────────────────────
    def _analyze_regions(self) -> list:
        """
        Para cada región tumoral detectada:
          - Centroide en vóxeles y coordenadas normalizadas
          - Volumen estimado
          - Intensidad media y máxima
          - Localización anatómica
          - Proximidad a zonas críticas
        """
        print("  [3/4] Analizando regiones...")
        labeled, n = ndimage.label(self.tumor_mask)
        regions = []

        sx, sy, sz = self.shape[:3]

        for i in range(1, n + 1):
            region_mask = labeled == i
            size = int(region_mask.sum())
            if size < 50:
                continue

            # Centroide
            coords = np.array(np.where(region_mask)).T
            centroid = coords.mean(axis=0)  # [x, y, z]

            # Normalizar a [0,1]
            cx_n = centroid[0] / sx
            cy_n = centroid[1] / sy
            cz_n = centroid[2] / sz

            # Volumen (asumiendo vóxeles isométricos ~1mm si no hay info)
            voxel_vol_mm3 = 1.0  # aproximación
            volume_mm3 = size * voxel_vol_mm3

            # Intensidades
            intensities = self.data_clean[region_mask]
            mean_int = float(intensities.mean())
            max_int  = float(intensities.max())

            # Bounding box
            zs = coords[:, 2]
            slices_span = (int(zs.min()), int(zs.max()))

            # Localización anatómica
            locations, critical = self._locate_anatomical(cx_n, cy_n, cz_n)

            # Hemisferio
            hemisphere = "Izquierdo" if cx_n < 0.5 else "Derecho"

            # Posición AP (anterior-posterior)
            if cy_n > 0.65:
                ap_pos = "Anterior (Frontal)"
            elif cy_n > 0.45:
                ap_pos = "Central"
            elif cy_n > 0.25:
                ap_pos = "Posterior (Parieto-Occipital)"
            else:
                ap_pos = "Muy Posterior / Occipital"

            # Posición SI (superior-inferior)
            if cz_n > 0.7:
                si_pos = "Superior"
            elif cz_n > 0.45:
                si_pos = "Medio"
            else:
                si_pos = "Inferior / Basal"

            regions.append({
                "id":          i,
                "size_voxels": size,
                "volume_mm3":  volume_mm3,
                "centroid":    centroid,
                "centroid_n":  (cx_n, cy_n, cz_n),
                "mean_int":    mean_int,
                "max_int":     max_int,
                "slices":      slices_span,
                "hemisphere":  hemisphere,
                "ap_position": ap_pos,
                "si_position": si_pos,
                "locations":   locations,
                "critical":    critical,
            })

        # Ordenar por tamaño descendente
        regions.sort(key=lambda r: r["size_voxels"], reverse=True)
        print(f"     {len(regions)} región(es) analizadas ✓")
        return regions

    def _locate_anatomical(self, cx, cy, cz) -> tuple:
        """
        Devuelve lista de zonas anatómicas donde cae el centroide
        y si alguna es zona crítica.
        """
        matched = []
        for zone_name, bounds in BRAIN_ATLAS.items():
            xmin, xmax, ymin, ymax, zmin, zmax, _ = bounds
            if xmin <= cx <= xmax and ymin <= cy <= ymax and zmin <= cz <= zmax:
                matched.append(zone_name)

        if not matched:
            matched = ["Zona no especificada en atlas"]

        critical = [z for z in matched if z in CRITICAL_ZONES]
        return matched, critical

    # ──────────────────────────────────────────
    #  4. ANÁLISIS DE TEXTURAS
    # ──────────────────────────────────────────
    def _analyze_textures_all_regions(self) -> dict:
        """
        GLCM (Gray-Level Co-occurrence Matrix) → rasgos de Haralick:
          - Contraste, correlación, energía, homogeneidad, disimilaridad
        Por cada región tumoral en su corte central.
        """
        print("  [4/4] Analizando texturas (GLCM/Haralick)...")
        labeled, _ = ndimage.label(self.tumor_mask)
        results = {}

        for reg in self.regions:
            rid = reg["id"]
            # Corte central de la región
            z_center = int(reg["centroid"][2])
            sl = self.data_clean[:, :, z_center]
            sl_uint8 = (sl * 255).astype(np.uint8)

            region_2d = (labeled[:, :, z_center] == rid)
            if region_2d.sum() < 10:
                continue

            # Bounding box 2D
            rows = np.where(region_2d.any(axis=1))[0]
            cols = np.where(region_2d.any(axis=0))[0]
            if len(rows) < 3 or len(cols) < 3:
                continue

            patch = sl_uint8[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
            if patch.size < 16:
                continue

            # GLCM en 4 ángulos
            glcm = graycomatrix(patch, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)

            results[rid] = {
                "contraste":    float(graycoprops(glcm, 'contrast').mean()),
                "correlacion":  float(graycoprops(glcm, 'correlation').mean()),
                "energia":      float(graycoprops(glcm, 'energy').mean()),
                "homogeneidad": float(graycoprops(glcm, 'homogeneity').mean()),
                "disimilaridad":float(graycoprops(glcm, 'dissimilarity').mean()),
            }

        print("     Análisis de texturas completado ✓")
        return results

    # ──────────────────────────────────────────
    #  REPORTE EN CONSOLA
    # ──────────────────────────────────────────
    def _print_report(self):
        print(f"\n{'═'*60}")
        print(f"  REPORTE DE ANÁLISIS: {self.filename}")
        print(f"{'═'*60}")

        if not self.regions:
            print("  ✅ No se detectaron regiones anómalas significativas.")
            return

        print(f"  ⚠️  Se detectaron {len(self.regions)} región(es) de interés:\n")

        for i, reg in enumerate(self.regions, 1):
            rid = reg["id"]
            print(f"  {'─'*55}")
            print(f"  REGIÓN {i}  (ID interno: {rid})")
            print(f"  {'─'*55}")
            print(f"  • Tamaño:     {reg['size_voxels']:,} vóxeles (~{reg['volume_mm3']:.0f} mm³)")
            print(f"  • Cortes:     {reg['slices'][0]} – {reg['slices'][1]}")
            print(f"  • Hemisferio: {reg['hemisphere']}")
            print(f"  • Posición:   {reg['ap_position']}, {reg['si_position']}")
            print(f"  • Señal:      media={reg['mean_int']:.3f}  máx={reg['max_int']:.3f}")
            print(f"\n  Localización anatómica:")
            for loc in reg["locations"]:
                marker = "⚠️ " if loc in CRITICAL_ZONES else "   "
                desc = BRAIN_ATLAS.get(loc, ("","","","","","",""))[6]
                print(f"    {marker}{loc}")
                if desc:
                    print(f"       → {desc}")

            if reg["critical"]:
                print(f"\n  🔴 ZONAS CRÍTICAS INVOLUCRADAS:")
                for cz in reg["critical"]:
                    desc = BRAIN_ATLAS[cz][6]
                    print(f"     ⚠️  {cz}: {desc}")

            if rid in self.textures:
                tx = self.textures[rid]
                print(f"\n  Texturas (GLCM/Haralick):")
                print(f"     Contraste:     {tx['contraste']:.4f}")
                print(f"     Correlación:   {tx['correlacion']:.4f}")
                print(f"     Energía:       {tx['energia']:.4f}")
                print(f"     Homogeneidad:  {tx['homogeneidad']:.4f}")
                print(f"     Disimilaridad: {tx['disimilaridad']:.4f}")
            print()

    # ──────────────────────────────────────────
    #  VISUALIZACIÓN INTERACTIVA
    # ──────────────────────────────────────────
    def show(self):
        """
        Visualizador interactivo con:
          - Slider de cortes axiales
          - Toggle: imagen original / preprocesada
          - Overlay del tumor en rojo semitransparente
          - Anotación de región y zona anatómica en cada corte
          - Panel lateral con info de región activa
        """
        # Mapa de colores para overlay
        tumor_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.1, 0.1, 0.6)]) # con todos ceros es un false, entonces transparente, con un true es rojo semitransparente

        fig = plt.figure(figsize=(18,9), facecolor='#1a1a2e') # tamaño visual ventana interactiva
        fig.suptitle(f"MRI Tumor Analyzer  |  {self.filename}",
                     color='white', fontsize=13, fontweight='bold', y=0.98)

        # Layout: imagen izq, panel info der
        ax_img  = fig.add_axes([0.03, 0.18, 0.58, 0.76]) # esto maneja la posicion del eje de cada item, poniendo 
        #desde donde empieza respecto al eje de la figura total, y hasta donde llega el nuevo eje
        ax_info = fig.add_axes([0.64, 0.18, 0.34, 0.76])
        ax_slid = fig.add_axes([0.08, 0.07, 0.5,  0.04])
        ax_tog  = fig.add_axes([0.64, 0.07, 0.12, 0.05])
        ax_prev = fig.add_axes([0.79, 0.07, 0.08, 0.05])
        ax_next = fig.add_axes([0.88, 0.07, 0.08, 0.05])
        '''
        ax_img  = fig.add_axes([0.03, 0.18, 0.58, 0.76])
                                   ↑     ↑     ↑     ↑
                                left  bottom  ancho  alto
                            (todo en porcentaje de 0 a 1)

        '''

        for a in [ax_img, ax_info]:
            a.set_facecolor('#0d0d1a')
        ax_info.axis('off')

        labeled, _ = ndimage.label(self.tumor_mask)

        # Estado compartido
        state = {"use_clean": True}

        def get_slice(z):
            vol = self.data_clean if state["use_clean"] else self.data_raw
            return vol[:, :, z], self.tumor_mask[:, :, z], labeled[:, :, z]

        z0 = self.shape[2] // 2
        sl, tm, lb = get_slice(z0)

        im_ax   = ax_img.imshow(sl.T, cmap='gray', origin='lower', aspect='auto') # ajustamos propiedades de la imagen base
        ov_ax   = ax_img.imshow(tm.T, cmap=tumor_cmap, origin='lower', aspect='auto',
                                vmin=0, vmax=1, alpha=0.7)#dibuja encima de im_ax, en el mismo eje ax_img, la mascara del tumor en rojo semitrasparente, con transparencia alpha=0.7
        ax_img.axis('off')

        title_ax = ax_img.set_title(f"Corte axial Z = {z0}", color='white', fontsize=11)

        # ── Slider ──────────────────────────────────
        slider = Slider(ax_slid, 'Corte Z', 0, self.shape[2]-1,
                        valinit=z0, valstep=1, color='#4a90d9')
        slider.label.set_color('white')
        slider.valtext.set_color('white')

        # ── Botón toggle ────────────────────────────
        btn_tog = Button(ax_tog, 'Preprocesada', color='#2d4a7a', hovercolor='#3d6aaa')
        btn_tog.label.set_color('white')

        # ── Botones de región ───────────────────────
        btn_prev = Button(ax_prev, '◀ Reg', color='#2d4a7a', hovercolor='#3d6aaa')
        btn_next = Button(ax_next, 'Reg ▶', color='#2d4a7a', hovercolor='#3d6aaa')
        btn_prev.label.set_color('white')
        btn_next.label.set_color('white')

        region_idx = [0]  # mutable para closures

        def update_info():
            ax_info.clear()
            ax_info.axis('off')
            ax_info.set_facecolor('#0d0d1a')

            if not self.regions:
                ax_info.text(0.1, 0.9, "✅ Sin anomalías detectadas",
                             color='lime', fontsize=11, transform=ax_info.transAxes)
                return

            reg = self.regions[region_idx[0]]
            rid = reg["id"]
            y = 0.97

            def txt(s, color='white', size=9, bold=False):
                nonlocal y
                weight = 'bold' if bold else 'normal'
                ax_info.text(0.03, y, s, color=color, fontsize=size,
                             fontweight=weight, transform=ax_info.transAxes,
                             verticalalignment='top', wrap=True)
                y -= 0.04

            txt(f"REGIÓN {region_idx[0]+1} / {len(self.regions)}", '#4a90d9', 10, True)
            txt(f"Tamaño: {reg['size_voxels']:,} vóxeles", 'white')
            txt(f"Cortes: {reg['slices'][0]}–{reg['slices'][1]}", 'white')
            txt(f"Hemisferio: {reg['hemisphere']}", 'white')
            txt(f"Pos AP: {reg['ap_position']}", '#cccccc', 8)
            txt(f"Pos SI: {reg['si_position']}", '#cccccc', 8)
            txt(f"Señal media: {reg['mean_int']:.3f}", '#cccccc', 8)
            y -= 0.02
            txt("Localización:", '#4a90d9', 9, True)
            for loc in reg["locations"]:
                is_crit = loc in CRITICAL_ZONES
                col = '#ff6b6b' if is_crit else '#aaaaaa'
                prefix = "⚠ " if is_crit else "• "
                txt(f"{prefix}{loc}", col, 8)

            if reg["critical"]:
                y -= 0.02
                txt("🔴 ZONAS CRÍTICAS:", '#ff4444', 9, True)
                for cz in reg["critical"]:
                    desc = BRAIN_ATLAS[cz][6].replace("⚠️ CRÍTICO: ", "")
                    txt(f"  {desc}", '#ff8888', 7)

            if rid in self.textures:
                y -= 0.02
                txt("Texturas GLCM:", '#4a90d9', 9, True)
                tx = self.textures[rid]
                for k, v in tx.items():
                    txt(f"  {k}: {v:.4f}", '#bbbbbb', 7)

            fig.canvas.draw_idle()

        def update_image(z):
            z = int(z)
            sl, tm, lb = get_slice(z)
            im_ax.set_data(sl.T)
            im_ax.set_clim(sl.min(), sl.max())
            ov_ax.set_data(tm.T)

            lbl = "Preprocesada" if state["use_clean"] else "Original"
            tumor_txt = f" | ⚠ Tumor en este corte" if tm.any() else ""
            ax_img.set_title(f"Corte axial Z = {z}  [{lbl}]{tumor_txt}",
                             color='white', fontsize=10)

            # Anotaciones de región en el corte
            for ann in list(ax_img.texts):
                ann.remove()

            unique_ids = np.unique(lb[lb > 0])
            for uid in unique_ids:
                reg_match = next((r for r in self.regions if r["id"] == uid), None)
                if reg_match is None:
                    continue
                coords_2d = np.where(lb == uid)
                if len(coords_2d[0]) == 0:
                    continue
                cx2 = int(np.mean(coords_2d[0]))
                cy2 = int(np.mean(coords_2d[1]))
                zones_short = reg_match["locations"][0] if reg_match["locations"] else "?"
                is_crit = bool(reg_match["critical"])
                color = '#ff4444' if is_crit else '#ffcc00'
                ax_img.text(cy2, cx2, f"R{uid}\n{zones_short[:20]}",
                            color=color, fontsize=6.5, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

            fig.canvas.draw_idle()

        def on_slider(val):
            update_image(val)

        def on_toggle(event):
            state["use_clean"] = not state["use_clean"]
            lbl = "Preprocesada" if state["use_clean"] else "Original"
            btn_tog.label.set_text(lbl)
            update_image(slider.val)

        def on_prev(event):
            if self.regions:
                region_idx[0] = (region_idx[0] - 1) % len(self.regions)
                update_info()
                # Ir al corte central de la región
                z_c = int(self.regions[region_idx[0]]["centroid"][2])
                slider.set_val(z_c)

        def on_next(event):
            if self.regions:
                region_idx[0] = (region_idx[0] + 1) % len(self.regions)
                update_info()
                z_c = int(self.regions[region_idx[0]]["centroid"][2])
                slider.set_val(z_c)

        slider.on_changed(on_slider)
        btn_tog.on_clicked(on_toggle)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        update_info()
        # Ir al corte con más tumor al inicio
        if self.regions:
            z_start = int(self.regions[0]["centroid"][2])
            slider.set_val(z_start)
        else:
            update_image(z0)

        # Leyenda
        patch_tumor = mpatches.Patch(color=(1, 0.1, 0.1, 0.7), label='Región sospechosa')
        ax_img.legend(handles=[patch_tumor], loc='lower left',
                      facecolor='#1a1a2e', labelcolor='white', fontsize=8)

        plt.show()


# ═══════════════════════════════════════════════════════════════
#  UTILIDADES DE CARGA
# ═══════════════════════════════════════════════════════════════

def find_nifti_files(data_dir: str = "./data") -> list:
    """Busca recursivamente archivos .nii y .nii.gz en data_dir."""
    patterns = ["**/*.nii.gz", "**/*.nii"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    # Preferir archivos T1c (contraste) que son los mejores para tumores
    t1c = [f for f in files if "t1c" in os.path.basename(f).lower()]
    others = [f for f in files if f not in t1c]
    return t1c + others


def select_file_cli(files: list) -> str:
    """Selección por consola si hay múltiples archivos."""
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    print("\n  Archivos NIfTI encontrados:")
    for i, f in enumerate(files):
        print(f"    [{i}] {f}")
    while True:
        try:
            idx = int(input(f"\n  Seleccioná un archivo [0-{len(files)-1}]: "))
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, KeyboardInterrupt):
            return files[0]


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  MRI TUMOR ANALYZER")
    print("  Detección y localización anatómica de tumores en RMN")
    print("═"*60)

    # ── Buscar archivos ──────────────────────────────────────
    # Intentar primero la ruta del ejemplo del usuario
    search_dirs = ["./data", "./data/MRI-images", "../data", "."]
    nifti_files = []

    for d in search_dirs:
        if os.path.isdir(d):
            found = find_nifti_files(d)
            if found:
                nifti_files = found
                print(f"  Encontrados {len(found)} archivo(s) en '{d}'")
                break

    if not nifti_files:
        # Pedir ruta manualmente
        print("\n  No se encontraron archivos NIfTI automáticamente.")
        path = input("  Ingresá la ruta al archivo .nii/.nii.gz: ").strip()
        if not path or not os.path.isfile(path):
            print("  ❌ Archivo no encontrado. Saliendo.")
            sys.exit(1)
        nifti_files = [path]

    filepath = select_file_cli(nifti_files)

    # ── Análisis ─────────────────────────────────────────────
    try:
        analyzer = MRITumorAnalyzer(filepath) #construyo una maquina concreta a partir del molde, pasandole el archivo a analizar
        analyzer.show()
    except Exception as e:
        print(f"\n  ❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()