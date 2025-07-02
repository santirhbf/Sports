# 🧠 RAG-powered Assistant for Elite Football Performance Data

*Using LLMs and RAG to Democratize Performance Insights*

## 📘 Overview

This project aims to bring football performance data closer to coaching staff by allowing **natural language querying of player performance data**. Instead of relying on data analysts, coaches and physical staff can interact with a Streamlit app powered by a **Large Language Model (LLM) and Retrieval-Augmented Generation (RAG)** architecture.

Developed as part of a Sports Analytics elective for the Master in Data Science program at IE University, the app leverages simulated data from the Chelsea FC Performance Insights Vizathon. It integrates player GPS data, recovery metrics, and physical capability scores into a single unified dataframe, which the app can query and interpret in response to user questions. Beyond querying, the application can also **generate training or recovery recommendations** based on context and workload history.

---

## 🎯 Project Objective

To develop a Streamlit application powered by RAG + LLM that enables non-technical users (e.g., coaches, physical trainers) to:

- Ask natural language questions about player performance data.
- Receive actionable insights and summaries from GPS, recovery, and physical capability datasets.
- Get context-based **recommendations** (e.g., whether a player should rest, train, or recover more).

---

## 📂 Datasets Used

The project uses simulated but realistic datasets provided publicly by Chelsea FC’s Performance Insights Team:

- `GPS Data.csv`: Daily movement and load metrics (e.g., total distance, high-speed running, acceleration counts).
- `Recovery Status Data.csv`: Aggregated scores across sleep, soreness, subjective recovery, and biomarkers.
- `Physical Capability Data.csv`: Strength, sprint, jump, and movement capacity metrics.

📌 **Data Source:**  
[Performance Insights Vizathon Homepage](https://chelsea-fc-performance-insights.github.io/Competition/#data)  
[GitHub Repository](https://github.com/Chelsea-Fc-Performance-Insights/Competition)

---

## 🛠️ Methodology

### 🔧 Data Processing

- All datasets are merged into a single dataframe using the date as the primary key.
- Time and categorical features are standardized and normalized where applicable.
- The merged dataframe serves as the **knowledge base** queried by the RAG system.

### 🤖 LLM & RAG Architecture

- Uses a Retrieval-Augmented Generation pipeline:
  - Vector index built from the merged dataframe
  - LLM generates answers and recommendations based on retrieved relevant data
- Enables context-aware responses and basic recommendation logic (e.g., rest vs. train)

🧠 *Note:* Technical components such as the LLM model, embedding model, and vector database used will be detailed in upcoming updates to this README.

---

## 🚀 Deliverables

- ✅ A merged, query-ready dataframe containing daily player performance and recovery metrics.
- ✅ A functional Streamlit app that allows users to:
  - Ask natural language questions about a player's workload, recovery, or physical trends.
  - Receive relevant summaries and training suggestions.
- 🔜 (Coming Soon) Support for CSV upload, chat history, and multi-player scaling.

---

## ⚠️ Limitations

- The dataset contains only one (synthetic) player, though designed to scale to full squads.
- The app runs on free-tier Streamlit Cloud, which requires public data and lacks scalability.
  - **Future deployment on AWS or another secure cloud provider** is advised for scalability and privacy, especially for sensitive health data.
- Initial recommendations are based on heuristics and workload patterns. A more advanced rules-based or fine-tuned model could enhance decision support.

---

## 👥 Team Members

- Santiago Botero  
- Sebastián de Wind  
- Thomas Arturo Renwick Morales  
- Santiago Ruiz  

---

## 📄 License & Acknowledgments

This project is based on data provided by the **Chelsea FC Performance Insights Vizathon** and is used strictly for academic and non-commercial purposes. All data remains the property of its original creators.
