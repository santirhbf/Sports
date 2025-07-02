import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from fpdf import FPDF
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN ---
CSV_PATH = "data/processed/merged_df.csv"
MODEL_NAME = "google/flan-t5-base"

col1,col2,col3=st.columns(3)
with col2:
    st.image("data/logo.png", width=200)

# --- CARGA DE DATOS ---
st.set_page_config(page_title="CoachLens", layout="wide")
st.title("Your personal Assistant for Elite Football Performance")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Data Explorer", "CoachLens", "Recommendations"])

df = pd.read_csv(CSV_PATH)

# --- CREAR INDEX VECTORIAL EN MEMORIA ---
def create_vector_index():
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = []
    for _, row in df.iterrows():
        texto = f"The day {row['date']}, the recovery value (emboss_baseline_score) was {row.get('emboss_baseline_score', 'N/A')}. " \
                f"The bio_baseline_composite was {row.get('bio_baseline_composite', 'N/A')}. " \
                f"The travelled distance was {row.get('distance', 'N/A')} meters. " \
                f"The number of accelerations was {row.get('accel_decel_over_4_5', 'N/A')}."
        texts.append(texto)
    docs = text_splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# --- CARGA DEL MODELO Y RAG ---
@st.cache_resource
def load_qa_chain():
    vectordb = create_vector_index()
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

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the next information about the player to answer the question.
        If you do not know the answer, answer this instead: "Insufficient information."

        Context:
        {context}

        Question::
        {question}

        Answer::
        """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

# --- SECCIÓN: EXPLORADOR DE DATOS ---
if section == "Data Explorer":
    st.subheader("📊 Performance and Recovery Data")
    uploaded_file = st.file_uploader("Upload a file", type=["csv"])
    if uploaded_file is not None:
        df = uploaded_file
    st.dataframe(df, use_container_width=True)

    if st.checkbox("Show metadata"):
        st.write(df.describe(include='all'))

    
    

# --- SECCIÓN: CONSULTAS AL LLM ---
elif section == "CoachLens":
    st.subheader("💬 Ask me anything!")
    qa = load_qa_chain()

    st.markdown("**Question examples:**")
    example_questions = [
        "How was the player's recovery on the first of October of 2023?",
        "Which were the highest metrics the day after the match against Arsenal?",
        "What day was the lowest emboss_baseline_score?"
    ]

    response = ""
    for q in example_questions:
        if st.button(q):
            user_query = q
            with st.spinner("Reasoning..."):
                response = qa.run(user_query)
                st.success(response)

    user_query = st.text_input("Ask away:")
    if user_query:
        with st.spinner("Reasoning..."):
            response = qa.run(user_query)
            st.success(response)

    if response:
        if st.button("📄 Download the answer in PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=response)
            pdf_output = "CoachLens_Answer.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("Click here to download the PDF", f, file_name=pdf_output)

# --- SECCIÓN: RECOMENDACIONES ---
elif section == "Recommendations":
    st.subheader("📌 System Recommendations")

    latest = df.sort_values("date", ascending=False).head(1).squeeze()
    recs = []

    if latest.get("emboss_baseline_score", 0) < -0.4:
        recs.append("🛌 The recovery score is low.An active recovery or a good rest are recommended.")

    if latest.get("accel_decel_over_4_5", 0) > 50:
        recs.append("⚠️ High accelerative load detected. Consider additional monitoring to avoid heavy fatigue.")

    if not recs:
        recs.append("✅ Everything seems inside the normal parameters.Keep up the good work!")

    for r in recs:
        st.write(r)