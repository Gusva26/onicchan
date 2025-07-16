import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import pytz
from PIL import Image, Image as PILImage

# Configuración de la página
st.set_page_config(page_title="Diagnóstico de Enfermedades en Hojas de Maíz", layout="wide")

# Inicializar variables de sesión
if 'analisis_realizado' not in st.session_state:
    st.session_state.analisis_realizado = False
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('models/best_grape_model.h5')

modelo = cargar_modelo()

# Información de enfermedades del maíz actualizada
INFORMACION_ENFERMEDADES = {
    0: {
        "nombre": "Hoja Sana",
        "tratamiento": "No se requiere tratamiento. Continuar con buenas prácticas agrícolas.",
        "sintomas": "Color verde uniforme, sin manchas ni decoloraciones",
        "color": "#4CAF50"  # Verde saludable
    },
    1: {
        "nombre": "Roya Común (Puccinia sorghi)",
        "tratamiento": "1. Aplicar fungicidas triazoles (Tebuconazole, Propiconazole)\n2. Usar variedades resistentes\n3. Rotación de cultivos",
        "sintomas": "Pequeñas pústulas circulares de color rojizo-marrón en ambas caras de la hoja",
        "color": "#8B4513"  # Marrón rojizo
    },
    2: {
        "nombre": "Tizón de la Hoja (Exserohilum turcicum)",
        "tratamiento": "1. Fungicidas a base de Estrobilurinas\n2. Eliminar residuos de cosecha\n3. Evitar riego por aspersión",
        "sintomas": "Lesiones alargadas color café con márgenes amarillentos (forma de cigarro)",
        "color": "#A0522D"  # Siena
    },
    3: {
        "nombre": "Mancha Gris (Cercospora zeae-maydis)",
        "tratamiento": "1. Aplicar Clorotalonil o Mancozeb\n2. Reducir densidad de siembra\n3. Balancear fertilización nitrogenada",
        "sintomas": "Manchas rectangulares grisáceas con bordes definidos entre las venas",
        "color": "#808080"  # Gris
    }
}

# Función para generar PDF
def generar_reporte_pdf(imagen, diagnostico, tratamiento, clases, probabilidades, modelo_usado="CNN Personalizado"):
    import time
    verde_oscuro = (0, 100, 0)
    verde_marco = (0, 128, 0)

    from fpdf import FPDF
    import matplotlib.pyplot as plt
    from datetime import datetime
    import pytz
    import os

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    colores = ["#6A0DAD", "#000000", "#8B4513", "#CD5C5C"]
    bars = ax.bar(clases, probabilidades, color=colores)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', (bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    ruta_grafico = "temp_chart.png"
    plt.savefig(ruta_grafico, dpi=150)
    plt.close()

    # Guardar imagen
    ruta_imagen = "temp_diag.png"
    imagen.save(ruta_imagen)

    pdf = FPDF()
    pdf.add_page()

    # Encabezado
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(*verde_oscuro)
    pdf.cell(0, 10, txt="REPORTE DE DIAGNÓSTICO DE HOJA DE MAÍZ", ln=1, align='C')

    # Contenido del reporte
    pdf.set_xy(10, 30)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt="MODELO UTILIZADO:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=modelo_usado, ln=1)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="FECHA DE ANÁLISIS:", ln=1)
    pdf.set_font("Arial", '', 12)
    tz_peru = pytz.timezone('America/Lima')
    fecha_hora_peru = datetime.now(tz_peru).strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, txt=fecha_hora_peru, ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(*verde_oscuro)
    pdf.cell(0, 10, txt="DIAGNÓSTICO:", ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=diagnostico, ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="TRATAMIENTO RECOMENDADO:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(100, 10, txt=tratamiento)

    # Imagen con marco
    x_imagen = 120
    y_imagen = 30
    ancho_imagen = 80
    grosor_marco = 1

    pdf.set_draw_color(*verde_marco)
    pdf.set_line_width(grosor_marco)
    pdf.rect(
        x_imagen - grosor_marco, 
        y_imagen - grosor_marco, 
        ancho_imagen + (2 * grosor_marco), 
        ancho_imagen + (2 * grosor_marco)
    )
    pdf.image(ruta_imagen, x=x_imagen, y=y_imagen, w=ancho_imagen)

    # Gráfico de probabilidades
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(*verde_oscuro)
    pdf.cell(0, 10, txt="PROBABILIDADES POR CLASE:", ln=1)
    pdf.ln(5)
    pdf.image(ruta_grafico, x=20, w=170)

    ruta_pdf = "reporte_diagnostico.pdf"
    pdf.output(ruta_pdf)

    # Devolver el archivo PDF
    with open(ruta_pdf, "rb") as f:
        pdf_bytes = f.read()

    os.remove(ruta_pdf)
    os.remove(ruta_imagen)
    os.remove(ruta_grafico)
    
    return pdf_bytes

# Preprocesar imagen
def preprocesar_imagen(imagen):
    img = np.array(imagen)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    imagen_mejorada = cv2.merge((cl, a, b))
    img = cv2.cvtColor(imagen_mejorada, cv2.COLOR_LAB2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Mostrar resultados del análisis
def mostrar_resultados():
    if st.session_state.analisis_realizado and st.session_state.resultados:
        resultados = st.session_state.resultados
        info = resultados['info']

        st.markdown(f"### Resultado: **{info['nombre']}** ({resultados['confianza']:.1f}%)")
        st.markdown(f"**Síntomas:** {info['sintomas']}")
        st.markdown(f"**Tratamiento recomendado:**\n```\n{info['tratamiento']}\n```")

        st.subheader("Distribución de Probabilidades")
        figura, ejes = plt.subplots()
        barras = ejes.bar(resultados['etiquetas'], resultados['probabilidades'], 
                         color=resultados['colores'])
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        for barra in barras:
            altura = barra.get_height()
            ejes.annotate(f'{altura:.1f}%', (barra.get_x() + barra.get_width() / 2, altura),
                          ha='center', va='bottom')
        st.pyplot(figura)
        
        # Generar PDF y mostrar botón de descarga
        pdf_bytes = generar_reporte_pdf(
            imagen=resultados['imagen'],
            diagnostico=f"{info['nombre']} ({resultados['confianza']:.1f}%)",
            tratamiento=info['tratamiento'],
            clases=resultados['etiquetas'],
            probabilidades=resultados['probabilidades']
        )
        
        st.download_button(
            label="📥 Descargar Reporte PDF",
            data=pdf_bytes,
            file_name="reporte_diagnostico.pdf",
            mime="application/pdf"
        )

# Interfaz de la aplicación
st.title("🌽 Detector de Enfermedades en Hojas de Maíz")
modo = st.sidebar.selectbox("Modo", ["Diagnóstico", "Reporte Comparativo", "Guía de Enfermedades", "Reportes"])

# Sección de Diagnóstico
if modo == "Diagnóstico":
    st.header("🔍 Diagnóstico por Imagen")
    archivo_subido = st.file_uploader("Sube una imagen de hoja de maíz", type=["jpg", "jpeg", "png"])

    if archivo_subido is not None:
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption="Imagen subida", use_container_width=True)

        if st.button("Analizar"):
            with st.spinner("Procesando..."):
                imagen_procesada = preprocesar_imagen(imagen)
                prediccion = modelo.predict(imagen_procesada)
                clase_predicha = np.argmax(prediccion[0])
                confianza = np.max(prediccion[0]) * 100
                info = INFORMACION_ENFERMEDADES[clase_predicha]

                st.session_state.resultados = {
                    'imagen': imagen,
                    'info': info,
                    'confianza': confianza,
                    'etiquetas': [INFORMACION_ENFERMEDADES[i]['nombre'] for i in range(len(prediccion[0]))],
                    'probabilidades': prediccion[0] * 100,
                    'colores': [INFORMACION_ENFERMEDADES[i]['color'] for i in range(len(prediccion[0]))]
                }
                st.session_state.analisis_realizado = True

    mostrar_resultados()

elif modo == "Reporte Comparativo":
    st.header("📊 Reporte Comparativo de Modelos")
    
    matrices = {
    "CNN Personalizado": [
        [371, 0, 35, 18],
        [9, 633, 523, 16],
        [2, 19, 1362, 1],
        [4, 13, 114, 946]
    ],
    "InceptionV3": [
        [308, 23, 58, 35],
        [52, 504, 565, 60],
        [87, 111, 1090, 96],
        [74, 83, 190, 730]
    ],
    "ResNet50": [
        [355, 3, 40, 26],
        [25, 588, 540, 28],
        [27, 38, 1300, 19],
        [20, 32, 125, 900]
    ]
}

    labels = ["hoja_sana", "roya_comun", "tizon_de_hoja", "mancha_gris"]

# Mostrar matriz
    matrix_images = []
    for modelo, matrix_data in matrices.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        matrix = np.array(matrix_data)
        cax = ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.5)
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, f"{val}", ha='center', va='center', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        plt.title(f"Matriz de Confusión - {modelo}")
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        img_buffer.seek(0)
        matrix_images.append((modelo, img_buffer))

    st.subheader("Vista Previa")
    cols = st.columns(len(matrix_images))
    for idx, (modelo, img_buffer) in enumerate(matrix_images):
        with cols[idx]:
            img_buffer.seek(0)
            st.image(img_buffer, caption=f"Matriz {modelo}", use_container_width=True)

    if st.button("📄 Generar Reporte Comparativo PDF"):
        pdf = FPDF()
        pdf.add_page()

        pdf.set_fill_color(245, 245, 245)
        pdf.rect(0, 0, 210, 297, 'F')

        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, "REPORTE COMPARATIVO DE MODELOS", ln=1, align='C')
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "DIAGNÓSTICO DE ENFERMEDADES EN MAÍZ", ln=1, align='C')

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        fecha = datetime.now(pytz.timezone("America/Lima")).strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(0, 10, f"Fecha: {fecha}", ln=1)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "MÉTRICAS DE RENDIMIENTO POR MODELO", ln=1)
        pdf.set_font("Arial", '', 11)

        data = [
            ["Modelo", "Precisión", "Sensibilidad", "Especificidad", "F1-Score", "MCC"],
            ["CNN Personalizado", "0.815", "0.818", "0.887", "0.830", "0.762"],
            ["InceptionV3", "0.647", "0.655", "0.664", "0.644", "0.519"],
            ["ResNet50", "0.773", "0.778", "0.824", "0.779", "0.701"]
        ]
        col_width = pdf.w / 6.5
        row_height = pdf.font_size * 1.5
        for row in data:
            for item in row:
                pdf.cell(col_width, row_height, item, border=1)
            pdf.ln(row_height)

        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, """
- Precisión: proporción de predicciones correctas entre todas las muestras.
- Sensibilidad (Recall): capacidad del modelo para identificar correctamente los positivos.
- Especificidad: capacidad del modelo para identificar correctamente los negativos.
- F1-Score: media armónica entre precisión y sensibilidad.
- MCC: medida robusta de calidad de clasificación, incluso con clases desbalanceadas.
""")

        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, "ANÁLISIS ESTADÍSTICO COMPARATIVO", ln=1)
        pdf.set_text_color(0, 0, 0)

        comparaciones = [
            ("CNN Personalizado vs InceptionV3", 4.2300, 0.0392),
            ("CNN Personalizado vs ResNet50", 1.0200, 0.3123),
            ("InceptionV3 vs ResNet50", 0.7800, 0.3769)
        ]
        for nombre, estadistico, pval in comparaciones:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 10, f"Prueba de McNemar: {nombre}", ln=1)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 10, f"Estadístico: {estadistico:.4f}, p-valor: {pval:.4f}", ln=1)
            if pval < 0.05:
                pdf.set_text_color(0, 128, 0)
                pdf.cell(0, 10, "Resultado: Diferencia significativa (< 0.05)", ln=1)
            else:
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 10, "Resultado: No hay diferencia significativa (> 0.05)", ln=1)
            pdf.set_text_color(0, 0, 0)

        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, "RECOMENDACIONES", ln=1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 11)
        recomendaciones = [
            "Modelo recomendado: CNN Personalizado",
            "- MCC alto (0.762): fuerte correlación entre predicciones y etiquetas verdaderas.",
            "- Precisión de 0.815: comete pocos errores.",
            "- F1-Score de 0.830: equilibrio entre precisión y sensibilidad."
        ]
        for r in recomendaciones:
            pdf.multi_cell(0, 7, r)

        pdf.add_page()

        matrix_pils = [PILImage.open(buf[1]) for buf in matrix_images]
        altura_comun = 250
        matrices_redimensionadas = []
        for img in matrix_pils:
            factor = altura_comun / img.height
            nueva_img = img.resize((int(img.width * factor), altura_comun))
            matrices_redimensionadas.append(nueva_img)

        ancho_total = sum(img.width for img in matrices_redimensionadas)
        combinada = PILImage.new("RGB", (ancho_total, altura_comun), color=(255, 255, 255))
        x_offset = 0
        for img in matrices_redimensionadas:
            combinada.paste(img, (x_offset, 0))
            x_offset += img.width

        ruta_matriz_combinada = "matrices_comb.png"
        combinada.save(ruta_matriz_combinada)

        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 10, "MATRICES DE CONFUSIÓN", ln=1, align='C')
        pdf.ln(5)
        pdf.image(ruta_matriz_combinada, x=10, w=pdf.w - 20)
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "Desarrollado por Luis Silva, Joel Florian, Maricielo Moreno © 2025", ln=1, align='C')
        pdf.cell(0, 10, "Sistema de Diagnóstico Inteligente para Enfermedades en Hojas de Maíz", ln=1, align='C')
        os.remove(ruta_matriz_combinada)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("⬇️ Descargar Reporte Comparativo PDF", data=pdf_bytes, file_name="reporte_comparativo_maiz.pdf", mime="application/pdf")
        
# Sección de Guía
elif modo == "Guía de Enfermedades":
    st.header("📚 Guía Visual")
    for indice, info in INFORMACION_ENFERMEDADES.items():
        st.subheader(info['nombre'])
        st.markdown(f"**Síntomas:** {info['sintomas']}")
        st.markdown(f"**Tratamiento:**\n```\n{info['tratamiento']}\n```")
        ruta_ejemplo = f"data/examples/{indice}.jpg"
        if os.path.exists(ruta_ejemplo):
            st.image(ruta_ejemplo, caption=f"Ejemplo de {info['nombre']}")

# Sección de Reportes
elif modo == "Reportes":
    st.header("📈 Reportes del Modelo")
    st.image("reports/model_comparison.png", caption="Comparación de Modelos")
    st.image("reports/confusion_matrix.png", caption="Matriz de Confusión")
    st.image("reports/learning_curves.png", caption="Curvas de Precisión")