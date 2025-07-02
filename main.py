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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- CONFIGURACION ---
CSV_PATH = "data/processed/merged_df.csv"
DB_DIR = "chroma_db"
MODEL_NAME = "google/flan-t5-base"  # Modelo Hugging Face instruct-compatible

col1,col2,col3=st.columns(3)
with col2:
    st.image("data/logo.png", width=200)

# --- CARGA DE DATOS ---
st.set_page_config(page_title="CoachLens", layout="wide")
st.title("Your personal Assistant for Elite Football Performance")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Data Explorer", "CoachLens", "Recommendations"])

df = pd.read_csv(CSV_PATH)

# --- CREAR INDEX VECTORIAL SI NO EXISTE ---
def create_vector_index():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = []
        for _, row in df.iterrows():
            texto = f"The day {row['date']}, the recovery value (emboss_baseline_score) was {row.get('emboss_baseline_score', 'N/A')}. " \
                     f"the bio_baseline_composite were in {row.get('bio_baseline_composite', 'N/A')}. " \
                     f"the distance travelled was {row.get('distance', 'N/A')} meters. " \
                     f"The number of strong accelerations was {row.get('accel_decel_over_4_5', 'N/A')}."
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
        Use the next information about the player to answer the question.
        If you do not know the answer, answer this instead \"Insufficient information\".

        Context:
        {context}

        Question:
        {question}

        Answer:
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
if section == "Data Explorer":
    st.subheader("📊 Performance and Recovery Data")
    st.dataframe(df, use_container_width=True)

    if st.checkbox("Show metadata"):
        st.write(df.describe(include='all'))

# --- SECCION: CONSULTAS AL LLM ---
elif section == "CoachLens":
    st.subheader("💬 Ask me anything!")
    qa = load_qa_chain()

    st.markdown("**Questions you can ask me:**")
    example_questions = [
        "How was the recovery of the player on the first of October of 2023?",
        "What were the highest metrics the day after the match against Arsenal?",
        "What day had the lowest emboss_baseline_score?",
        "On what days did the player exceed 7000 meters with poor muscle quality?",
        "How has the recovery progressed over the past week?"
    ]

    response = ""
    for q in example_questions:
        if st.button(q):
            user_query = q
            with st.spinner("Reasoning..."):
                response = qa.run(user_query)
                st.success(response)

    user_query = st.text_input("Or ask away:")
    if user_query:
        with st.spinner("Reasoning..."):
            response = qa.run(user_query)
            st.success(response)

    if response:
        if st.button("📄 Download your answer as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=response)
            pdf_output = "respuesta_llm.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("Click here to download the PDF", f, file_name=pdf_output)

# --- SECCION: RECOMENDACIONES ---
elif section == "Recommendations":
    st.subheader("📌 System Recommendations")

    latest = df.sort_values("date", ascending=False).head(1).squeeze()
    recs = []

    if latest.get("emboss_baseline_score", 0) < -0.4:
        recs.append("🛌 The recovery score is low. An active recovery or a good rest are recommended.")

    if latest.get("accel_decel_over_4_5", 0) > 50:
        recs.append("⚠️ High acceleration detected. Consider additional monitoring to avoid heavy fatigue.")

    if not recs:
        recs.append("✅ Everything seems inside the normal parameters. Keep up the good work!")

    for r in recs:
        st.write(r)