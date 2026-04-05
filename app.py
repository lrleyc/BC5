# ============================================================
# CABECERA
# ============================================================
# Alumno: Luis Ricardo Ley Castro
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un analista de datos experto trabajando sobre un dataset de historial de Spotify.

Tu tarea es generar código Python que analice un DataFrame llamado `df` y produzca visualizaciones usando Plotly.

====================
CONTEXTO DEL DATASET
====================
El dataset contiene reproducciones musicales entre {fecha_min} y {fecha_max}.

Columnas disponibles en df:
- ts: datetime de la reproducción
- ms_played: milisegundos reproducidos
- track: nombre de la canción
- artist: artista principal
- album: álbum
- platform: plataforma ({plataformas})
- reason_start: motivo de inicio ({reason_start_values})
- reason_end: motivo de fin ({reason_end_values})
- shuffle: booleano
- skipped: booleano (True si se saltó)

Columnas derivadas disponibles:
- minutes_played: minutos reproducidos
- hour: hora del día (0-23)
- day_of_week: día de la semana (Monday, Tuesday, etc.)
- month: mes (1-12)
- month_name: nombre del mes
- is_weekend: True si es fin de semana
- season: estación (Winter, Spring, Summer, Fall)

IMPORTANTE:
- Los valores nulos en track o artist corresponden a podcasts → ignóralos para análisis musicales

====================
INSTRUCCIONES
====================
1. SIEMPRE responde en formato JSON válido con esta estructura:

{{ 
  "tipo": "grafico" o "fuera_de_alcance",
  "codigo": "código Python ejecutable",
  "interpretacion": "explicación clara en lenguaje natural"
}}

2. Si la pregunta es válida:
- "tipo" = "grafico"
- Genera código que:
  - Use df directamente
  - Cree una figura Plotly llamada `fig`
  - No imprima nada
  - No use datos externos

3. Si la pregunta NO se puede responder con el dataset:
- "tipo" = "fuera_de_alcance"
- "codigo" = ""
- Explica brevemente por qué

====================
BUENAS PRÁCTICAS
====================
- Usa groupby para agregaciones
- Ordena resultados cuando tenga sentido (top rankings)
- Usa minutos (minutes_played) para medir tiempo
- Usa títulos claros en gráficos
- Usa labels claros en ejes
- Limita rankings a Top 10 máximo si no se especifica
- Cuando compares periodos de distinta duración (por ejemplo, fin de semana vs entre semana), utiliza métricas normalizadas como promedios en lugar de totales para evitar sesgos.
- Cuando hagas comparaciones entre periodos, mantén consistencia en la definición de "top" (por ejemplo, seleccionar primero los artistas más escuchados en total y luego comparar su comportamiento entre periodos).

====================
TIPOS DE PREGUNTAS QUE DEBES RESOLVER
====================
A. Rankings (top artistas, canciones, etc.)
B. Evolución temporal (por mes)
C. Patrones (hora, día, plataforma)
D. Comportamiento (skips, shuffle)
E. Comparaciones (periodos, estaciones, etc.)

====================
REGLAS IMPORTANTES
====================
- NO uses print()
- NO modifiques el df
- NO uses librerías fuera de pandas, plotly
- El resultado SIEMPRE debe ser una figura llamada `fig`
- Filtra podcasts cuando analices música
- Maneja ambigüedad de forma razonable

====================
EJEMPLO DE RESPUESTA
====================
{{
  "tipo": "grafico",
  "codigo": "top = df.groupby('artist')['minutes_played'].sum().sort_values(ascending=False).head(10).reset_index(); fig = px.bar(top, x='artist', y='minutes_played', title='Top artistas')",
  "interpretacion": "Estos son los artistas más escuchados en minutos."
}}

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    # Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # Renombrar columnas para simplificar el código
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album"
    })

    # Eliminar registros de podcasts
    df = df[df["track"].notna()]

    # Convertir milisegundos a minutos
    df["minutes_played"] = df["ms_played"] / 60000

    # Time-based features
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.day_name()
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()

    # Indicador de fin de semana
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])

    # Indicador de temporadas
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["season"] = df["month"].apply(get_season)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La aplicación sigue una arquitectura text-to-code en la que el LLM no recibe los datos directamente,
# sino una descripción estructurada del dataset. A partir de la pregunta del usuario y el system prompt,
# el modelo genera código Python como texto. Este código se ejecuta localmente sobre el df previamente cargado. 
# El LLM devuelve un JSON con tres campos: tipo, codigo e interpretacion. El código se ejecuta en el entorno local 
# de la app y genera una figura de Plotly que se muestra en Streamlit. El LLM no recibe los datos directamente para 
# evitar problemas de privacidad, reducir consumo de tokens y garantizar que el análisis se realice de forma reproducible.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#   El system prompt le da al LLM una descripción detallada del dataset, su estructura, las columnas disponibles y sus significados. 
# Además, establece las reglas de formato deseado para las respuestas, incluyendo guardrails para lidiar con preguntas fuera de alcance.
# Por ejemplo, la pregunta "¿Cuáles son mis artistas más escuchados?" funciona gracias a que el prompt le indica al LLM que debe usar groupby para agregaciones y limitar a Top 10,
# de no ser así, el LLM podría generar un gráfico con demasiados artistas, lo que sería difícil de interpretar. Por otro lado, una pregunta como "¿Cuál es la letra de mi canción 
# más escuchada?" fallaría si no tuviera la instrucción clara de que no se pueden usar datos externos, ya que el LLM podría intentar buscar la letra en internet, lo cual no está 
# permitido según las reglas establecidas en el enunciado.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    1. El usuario escribe una pregunta en la interfaz de Streamlit y la envía.
#    2. La aplicación muestra la pregunta en la interfaz como un mensaje del usuario.
#    3. La aplicación envía la pregunta junto con el system prompt a la API de OpenAI.
#    4. El LLM procesa la pregunta y genera una respuesta en formato JSON, que incluye el tipo de respuesta, el código Python para generar la visualización y una interpretación en lenguaje natural.
#    5. La aplicación recibe la respuesta y la parsea para convertirla en un diccionario de Python.
#    6. Si el tipo es "fuera_de_alcance", la aplicación muestra solo la interpretación al usuario.
#    7. Si el tipo es "grafico", la aplicación ejecuta el código generado en un entorno local, utilizando el DataFrame `df` y las librerías disponibles.
#    8. El código generado produce una figura de Plotly llamada `fig`, que se muestra en la interfaz de Streamlit.
#    9. La aplicación también muestra la interpretación en lenguaje natural y el código Python generado para transparencia y aprendizaje del usuario. 