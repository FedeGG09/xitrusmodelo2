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
from unidecode import unidecode  # Importar unidecode para normalizar caracteres

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
    return pd.read_excel(uploaded_file, sheet_name=None)

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
        
        relevant_chunks = []
        for idx in indices[0]:
            relevant_chunks.append(text_chunks[idx])

        return relevant_chunks

    def invoke_bedrock(self, prompt, aws_access_key_id, aws_secret_access_key, temperature=0.5, top_p=0.95):
        brt = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        modelId = 'meta.llama2-70b-chat-v1'
        
        body = json.dumps({
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p
        })
        
        response = brt.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
        return response['body'].read()

# Parámetros del modelo definidos directamente en el código
model_name = "nombre_del_modelo"  # Reemplaza con el nombre real del modelo
model_version = "version_del_modelo"  # Reemplaza con la versión real del modelo

# Instancia de VectorStoreModel
vector_store_model = VectorStoreModel(model_name, model_version)

# Function to generate the proposed string based on the style
def generate_style_string(style):
    if style == "neutral":
        return "it must reflect objectivity, factual accuracy, and a commitment to avoiding subjective influence, serving as a reliable source of information without introducing biases or personal opinions into the generated content."
    elif style == "formal":
        return "it must exhibit a high level of linguistic formality, presenting information in a structured and polished manner, suiting professional contexts and interactions where a formal tone is expected."
    elif style == "informal":
        return "it must be tailored to create a casual and conversational tone, allowing the AI's responses to engage users in a friendly and approachable manner, suitable for less formal interactions."
    elif style == "humorous":
        return "it must infuse responses with wit and humor, aiming to entertain and amuse users while still providing relevant information or assistance, creating an enjoyable and light-hearted interaction experience."
    else:
        return "it must reflect objectivity, factual accuracy, and a commitment to avoiding subjective influence, serving as a reliable source of information without introducing biases or personal opinions into the generated content."

# Streamlit interface
st.title("Carga y Consulta de Archivos en Ontología y Base de Datos")
st.write("Sube tus archivos para realizar consultas.")

uploaded_files = st.file_uploader("Subir archivos", accept_multiple_files=True)

# Checkbox para mostrar/ocultar opciones avanzadas
show_advanced_options = st.checkbox("Mostrar opciones avanzadas")

if show_advanced_options:
    advanced_expander = st.expander("Opciones avanzadas", expanded=False)
    with advanced_expander:
        # Selección de archivos por tipo
        st.write("Selecciona los archivos para cada tipo:")
        ontology_files = st.multiselect("Archivos de Ontología", uploaded_files, format_func=lambda x: x.name)
        pdf_files = st.multiselect("Archivos PDF", uploaded_files, format_func=lambda x: x.name)
        excel_files = st.multiselect("Archivos Excel", uploaded_files, format_func=lambda x: x.name)
        access_files = st.multiselect("Archivos Access", uploaded_files, format_func=lambda x: x.name)
        pickle_files = st.multiselect("Archivos Pickle", uploaded_files, format_func=lambda x: x.name)

        # Mostrar los archivos seleccionados en cada categoría
        st.write("Archivos seleccionados:")
        st.write("Ontología:", [file.name for file in ontology_files])
        st.write("PDF:", [file.name for file in pdf_files])
        st.write("Excel:", [file.name for file in excel_files])
        st.write("Access:", [file.name for file in access_files])
        st.write("Pickle:", [file.name for file in pickle_files])

# Manejo de archivos cargados y procesamiento
if uploaded_files:
    # Cargar archivos y procesar
    ontology_data = [load_ontology_file(file) for file in uploaded_files if file.name.endswith(('.ttl', '.rdf'))]
    pdf_data = load_documents([file for file in uploaded_files if file.name.endswith('.pdf')])
    excel_data = {file.name: load_excel_file(file) for file in uploaded_files if file.name.endswith('.xlsx')}
    access_data = {file.name: load_access_file(file) for file in uploaded_files if file.name.endswith('.accdb')}
    pickle_data = {file.name: load_pickle_file(file) for file in uploaded_files if file.name.endswith('.pkl')}

    st.success("Archivos cargados exitosamente.")

    # Proceso de extracción de tablas de archivos PDF
    if pdf_files:
        extracted_tables = extract_tables_from_pdfs([file for file in uploaded_files if file.name.endswith('.pdf')])
        for idx, table in enumerate(extracted_tables):
            st.write(f"Tabla extraída {idx + 1}")
            st.dataframe(table)

    # Proceso de tokenización y recuperación de información
    user_query = st.text_input("Ingrese su consulta:")

    if st.button("Buscar información"):
        ontology_info, db_info = tokenize_and_retrieve_info(user_query, ontology_data, excel_data)
        if ontology_info:
            st.write("Información relevante en la ontología:")
            for info in ontology_info:
                st.write(info)
        if db_info:
            st.write("Información relevante en la base de datos:")
            for table_name, records in db_info.items():
                st.write(f"Tabla: {table_name}")
                st.write(pd.DataFrame(records))

    # Procesamiento de chunks de texto
    text_chunks = split_text_into_chunks(pdf_data)

    # Almacén de vectores
    vector_store = vector_store_model.get_vectorstore(text_chunks)

    # Selección de estilo
    style = st.selectbox("Selecciona el estilo de respuesta", ["neutral", "formal", "informal", "humorous"])
    proposed_string = generate_style_string(style)

    # Guardar el historial de chat en st.session_state
    st.session_state.chat_history.append({
        'user_query': user_query,
        'ontology_info': ontology_info,
        'db_info': db_info,
        'style': style,
        'proposed_string': proposed_string
    })

    # Mostrar el historial de chat
    st.write("Historial de chat:")
    for idx, chat in enumerate(st.session_state.chat_history, 1):
        st.write(f"{idx}. Usuario: {chat['user_query']}")
        st.write(f"  - Ontología: {chat['ontology_info']}")
        st.write(f"  - Base de datos: {chat['db_info']}")
        st.write(f"  - Estilo: {chat['style']}")
        st.write(f"  - Propuesta: {chat['proposed_string']}")
else:
    st.info("Por favor, suba al menos un archivo para comenzar.")
