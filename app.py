import streamlit as st
import pandas as pd
import tempfile
import pickle
import pyodbc
from rdflib import Graph
from fuzzywuzzy import fuzz
import nltk
from sqlalchemy import create_engine
import boto3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from nltk.corpus import stopwords


# Inicialización de chat_history si no está definido en st.session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Función para preprocesar texto, incluyendo normalización de acentos y manejo de saltos de línea
def preprocess_text(text):
    text = unidecode(text)  # Normalizar acentos y caracteres especiales
    text = text.replace('\n', ' ').replace('\r', ' ')  # Manejar saltos de línea
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def load_file(uploaded_file, suffix, loader):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        return loader(tmp_file_path)
    except Exception as e:
        st.error(f"Error al cargar archivo {uploaded_file.name}: {str(e)}.")
        return None

def load_ontology_file(uploaded_file):
    file_path = uploaded_file.name
    g = Graph()
    g.parse(data=uploaded_file.read(), format='turtle' if file_path.endswith('.ttl') else 'xml')
    return g

def load_pickle_file(uploaded_file):
    return pickle.load(uploaded_file)

def load_access_file(uploaded_file):
    conn_str = r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=" + uploaded_file.name + ";"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.accdb') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        conn = pyodbc.connect(conn_str.replace(uploaded_file.name, tmp_file_path))
        cursor = conn.cursor()
        tables = [table_info.table_name for table_info in cursor.tables(tableType='TABLE')]
        data = {table: pd.read_sql(f"SELECT * FROM [{table}]", conn) for table in tables}
        conn.close()
        return data
    except pyodbc.Error as e:
        st.error(f"Error de conexión o consulta con Access: {str(e)}")
        return None

def load_excel_file(uploaded_file):
    try:
        xl = pd.ExcelFile(uploaded_file)
        data = {}
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
            data[sheet_name] = df
        return data
    except Exception as e:
        st.error(f"Error al cargar archivo Excel {uploaded_file.name}: {str(e)}")
        return None

def tokenize_and_retrieve_info(user_query, ontology, db_data):
    if not user_query:
        st.warning("Por favor ingrese una consulta.")
        return None, None

    ontology_texts = []
    if ontology:
        for subj, pred, obj in ontology:
            text = f"{subj} {pred} {obj}".lower()
            ontology_texts.append(text)

    db_texts = []
    db_texts_map = {}
    if db_data:
        for table_name, df in db_data.items():
            if isinstance(df, pd.DataFrame):
                df['combined_text'] = df.apply(lambda row: ' '.join(map(str, row.values)).lower(), axis=1)
                texts = df['combined_text'].tolist()
                db_texts.extend(texts)
                db_texts_map[table_name] = df

    all_texts = ontology_texts + db_texts

    if not all_texts:
        st.warning("No se encontró información en la ontología o en la base de datos.")
        return None, None

    vectorizer = TfidfVectorizer()
    text_vecs = vectorizer.fit_transform(all_texts)
    user_query_vec = vectorizer.transform([user_query.lower()])

    ontology_info = []
    db_info = {}

    if ontology:
        for idx, (subj, pred, obj) in enumerate(ontology):
            similarity = cosine_similarity(user_query_vec, text_vecs[idx])[0][0]
            if similarity > 0.2:
                ontology_info.append((subj, pred, obj))

    if db_data:
        for table_name, df in db_texts_map.items():
            df['similarity'] = df['combined_text'].apply(lambda text: cosine_similarity(user_query_vec, vectorizer.transform([text]))[0][0])
            matching_rows = df[df['similarity'] > 0.2]
            if not matching_rows.empty:
                db_info[table_name] = matching_rows.drop(columns=['combined_text', 'similarity']).to_dict(orient='records')

    return ontology_info, db_info

def load_documents(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    try:
        for file in uploaded_files:
            if file.name.endswith('.pdf'):
                temp_filepath = os.path.join(temp_dir.name, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())
                try:
                    loader = PyPDFLoader(temp_filepath)
                    docs.extend(loader.load())
                except Exception as e:
                    st.error(f"Error al cargar el archivo PDF {file.name}: {str(e)}")
    finally:
        temp_dir.cleanup()
    return docs

def extract_tables_from_pdfs(pdf_files):
    tables = []
    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    extracted_tables = page.extract_tables()
                    for table in extracted_tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
                        tables.append(df)
        except Exception as e:
            st.error(f"Error al extraer tablas del archivo {pdf_file.name}: {e}")
    return tables

def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

class VectorStoreModel:
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def get_vectorstore(self, text_chunks):
        text_chunks = [chunk.page_content if hasattr(chunk, 'page_content') else chunk for chunk in text_chunks]
        test_embedding = self.embeddings.embed_query("test")
        embedding_dim = len(test_embedding)
        vector_store = faiss.IndexFlatL2(embedding_dim)
        for chunk in text_chunks:
            vector = self.embeddings.embed_query(chunk)
            vector = np.array(vector)
            vector = vector.reshape(1, -1)
            vector_store.add(vector)
        return vector_store

    def get_relevant_chunks(self, vector_store, query, text_chunks, top_k=10):
        if vector_store is None:
            st.error("El almacén de vectores no está disponible.")
            return []

        query_vector = self.embeddings.embed_query(query)
        query_vector = np.array(query_vector)
        query_vector = query_vector.reshape(1, -1)
        distances, indices = vector_store.search(query_vector, k=top_k)
        indices = indices.flatten()
        relevant_chunks = [text_chunks[i] for i in indices]
        return relevant_chunks

uploaded_files = st.file_uploader("Cargar archivos", type=["ttl", "pkl", "accdb", "xls", "xlsx", "pdf"], accept_multiple_files=True)

ontology = None
db_data = {}
pdf_files = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".ttl") or uploaded_file.name.endswith(".rdf"):
            ontology = load_file(uploaded_file, '.ttl', load_ontology_file)
        elif uploaded_file.name.endswith(".pkl"):
            db_data = load_file(uploaded_file, '.pkl', load_pickle_file)
        elif uploaded_file.name.endswith(".accdb"):
            db_data = load_file(uploaded_file, '.accdb', load_access_file)
        elif uploaded_file.name.endswith(".xls") or uploaded_file.name.endswith(".xlsx"):
            db_data = load_excel_file(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            pdf_files.append(uploaded_file)
            
    st.success("Archivos cargados exitosamente.")

    if pdf_files:
        extracted_tables = extract_tables_from_pdfs(pdf_files)
        for idx, table in enumerate(extracted_tables):
            st.write(f"Tabla extraída {idx + 1}")
            st.write(table)

st.write("Archivos cargados:", uploaded_files)
user_query = st.text_input("Ingrese su consulta")
if st.button("Buscar"):
    ontology_info, db_info = tokenize_and_retrieve_info(user_query, ontology, db_data)
    if ontology_info:
        st.write("Información encontrada en la ontología:")
        for subj, pred, obj in ontology_info:
            st.write(f"Sujeto: {subj}, Predicado: {pred}, Objeto: {obj}")
    else:
        st.write("No se encontró información relevante en la ontología.")
    
    if db_info:
        st.write("Información encontrada en la base de datos:")
        for table_name, rows in db_info.items():
            st.write(f"Tabla: {table_name}")
            st.write(rows)
    else:
        st.write("No se encontró información relevante en la base de datos.")
