import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from fpdf import FPDF

# --- CONFIGURACION ---
CSV_PATH = "data/processed/merged_df.csv"
DB_DIR = "chroma_db"
MODEL_NAME = "google/flan-t5-base"  # Modelo Hugging Face instruct-compatible

# --- CARGA DE DATOS ---
st.set_page_config(page_title="RAG para Preparadores Físicos", layout="wide")
st.title("⚽ RAG App para Análisis de Rendimiento y Recuperación")

st.sidebar.header("Navegación")
section = st.sidebar.radio("Ir a:", ["Explorador de Datos", "Consultas al LLM", "Recomendaciones"])

df = pd.read_csv(CSV_PATH)

# --- CREAR INDEX VECTORIAL SI NO EXISTE ---
def create_vector_index():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = []
        for _, row in df.iterrows():
            texto = f"El día {row['date']}, el valor de recuperación (emboss_baseline_score) fue {row.get('emboss_baseline_score', 'N/A')}. " \
                     f"Los biomarcadores (bio_baseline_composite) estaban en {row.get('bio_baseline_composite', 'N/A')}. " \
                     f"La distancia recorrida fue de {row.get('distance', 'N/A')} metros. " \
                     f"El número de aceleraciones fuertes fue {row.get('accel_decel_over_4_5', 'N/A')}."
            texts.append(texto)
        docs = text_splitter.create_documents(texts)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_DIR)

# --- CARGA DEL MODELO Y RAG ---
@st.cache_resource
def load_qa_chain():
    create_vector_index()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    from langchain.prompts import PromptTemplate

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Usa la siguiente información del jugador para responder la pregunta en español.
        Si no sabes la respuesta, responde \"No tengo suficiente información.\".

        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta:
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# --- SECCION: EXPLORADOR DE DATOS ---
if section == "Explorador de Datos":
    st.subheader("📊 Datos de Rendimiento y Recuperación")
    st.dataframe(df, use_container_width=True)

    if st.checkbox("Mostrar descripción de columnas"):
        st.write(df.describe(include='all'))

# --- SECCION: CONSULTAS AL LLM ---
elif section == "Consultas al LLM":
    st.subheader("💬 Haz una pregunta sobre el rendimiento o recuperación")
    qa = load_qa_chain()

    st.markdown("**Ejemplos de preguntas que puedes hacer:**")
    example_questions = [
        "¿Cómo estuvo la recuperación del jugador el 1 de octubre de 2023?",
        "¿Qué métricas fueron más altas el día después del partido contra Arsenal?",
        "¿Qué día tuvo el valor más bajo en emboss_baseline_score?",
        "¿Cuándo fue alta la aceleración/desaceleración?",
        "¿Qué días superó los 7000 metros con baja calidad muscular?",
        "¿Cuál fue la evolución del estado de recuperación en la última semana?"
    ]

    response = ""
    for q in example_questions:
        if st.button(q):
            user_query = q
            with st.spinner("Pensando..."):
                response = qa.run(user_query)
                st.success(response)

    user_query = st.text_input("O escribe tu propia pregunta:")
    if user_query:
        with st.spinner("Pensando..."):
            response = qa.run(user_query)
            st.success(response)

    if response:
        if st.button("📄 Descargar respuesta en PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=response)
            pdf_output = "respuesta_llm.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("Haz clic aquí para guardar el PDF", f, file_name=pdf_output)

# --- SECCION: RECOMENDACIONES ---
elif section == "Recomendaciones":
    st.subheader("📌 Recomendaciones del sistema")

    latest = df.sort_values("date", ascending=False).head(1).squeeze()
    recs = []

    if latest.get("emboss_baseline_score", 0) < -0.4:
        recs.append("🛌 El score de recuperación está bajo. Se recomienda una sesión de recuperación activa o descanso.")

    if latest.get("accel_decel_over_4_5", 0) > 50:
        recs.append("⚠️ Alta carga acelerativa detectada. Considerar monitoreo adicional para evitar fatiga.")

    if not recs:
        recs.append("✅ Todo parece dentro de los parámetros normales. Seguir con el plan habitual.")

    for r in recs:
        st.write(r)