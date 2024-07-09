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
            dfs = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
            for df in dfs:
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
        return "it must be tailored to create a friendly and relaxed interaction, employing casual language and conversational elements, aiming to establish a connection with the audience in a manner that feels approachable and informal."
    elif style == "professional":
        return "it should embody formal language, clarity, and respect, presenting information in a polished and business-appropriate manner, well-suited for professional contexts where a high level of formality and authority is expected."
    else:
        return ""

# Function to generate the proposed string based on the source references
def generate_source_references_string(source_references):
    if source_references == "Mandatory":
        return "Explicitly include the name of the sources where different parts of the response come from."
    elif source_references == "On Demand":
        return "Only include source names upon request."
    else:
        return ""

# Extract parameters
def extract_parameters(actingUser, selectedRole):
    emotional_tone = actingUser["Role"]["EmotionalTone"]
    style = actingUser["Role"]["Style"]
    source_references = selectedRole["SourceReferences"]
    return {
        "emotional_tone": emotional_tone,
        "style": style,
        "source_references": source_references
    }
    
    return prompt_string

# Función para descargar el historial de chat
def download_chat_history(history):
    chat_content = "\n".join([f"{chat['role']}: {chat['content']}" for chat in history])
    file_name = "chat_history.txt"
    st.download_button(
        label="Descargar historial de chat",
        data=chat_content,
        file_name=file_name,
        mime="text/plain",
    )


def generate_response(relevant_chunks, emotional_tone, style):
    response_lines = []
    for chunk in relevant_chunks:
        response_lines.append(chunk)
    emotional_tone = emotional_tone.replace("_", " ")  # Reemplaza underscores con espacios para que Bedrock lo entienda
    prompt = f"Pregunta: {user_query}\n\nInformación relevante:\n{response_lines}\n\nTono emocional: {emotional_tone}\nEstilo: {style}"
    
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,

    response = vector_store_model.invoke_bedrock(
        prompt=prompt,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        temperature=0.5,
        top_p=0.95
    )
    return response
def download_chat_history(chat_history):
    # Crear contenido del historial de chat como texto
    chat_content = "\n".join([f"{chat.get('user', 'Usuario')}: {chat.get('query', '')}\nRespuesta del sistema: {chat.get('response', '')}\n\n" for chat in chat_history])

# Título de la aplicación
st.title("XITRUS - CHAT")

# Cargar archivos
uploaded_files = st.file_uploader("Carga tus archivos (TTL, PKL, ACCDB, XLSX, PDF)", type=['ttl', 'pkl', 'accdb', 'xlsx', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    ontology = None
    db_data = {}

    # Cargar ontologías
    ontology_files = [f for f in uploaded_files if f.name.endswith('.ttl') or f.name.endswith('.owl')]
    if ontology_files:
        for f in ontology_files:
            ontology = load_ontology_file(f)

    # Cargar archivos PKL
    pkl_files = [f for f in uploaded_files if f.name.endswith('.pkl')]
    if pkl_files:
        for f in pkl_files:
            db_data.update(load_pickle_file(f))

    # Cargar archivos ACCDB
    accdb_files = [f for f in uploaded_files if f.name.endswith('.accdb')]
    if accdb_files:
        for f in accdb_files:
            db_data.update(load_access_file(f))

    # Cargar archivos XLSX
    xlsx_files = [f for f in uploaded_files if f.name.endswith('.xlsx')]
    if xlsx_files:
        for f in xlsx_files:
            db_data.update(load_excel_file(f))

    # Cargar y procesar archivos PDF
    pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
    pdf_documents = load_documents(pdf_files)  # Cargar documentos PDF
    pdf_tables = extract_tables_from_pdfs(pdf_files)  # Extraer tablas de archivos PDF

    # Entrada de consulta del usuario
    user_query = st.text_input("Ingrese su consulta:")

    # Seleccionar tono emocional
    emotional_tone = st.selectbox("Seleccionar tono emocional", ["positivo", "neutral", "empático", "humorístico", "instructivo", "alentador"])

    # Seleccionar estilo
    style = st.selectbox("Seleccionar estilo", ["neutral", "formal", "informal", "profesional"])

    # Botón de búsqueda
    if st.button("Buscar"):
        if user_query:
            ontology_info, db_info = tokenize_and_retrieve_info(user_query, ontology, db_data)

            relevant_chunks = []
            if pdf_documents:
                pdf_chunks = split_text_into_chunks(pdf_documents)  # Dividir documentos en fragmentos
                vector_store = vector_store_model.get_vectorstore(pdf_chunks)
                relevant_chunks = vector_store_model.get_relevant_chunks(vector_store, user_query, pdf_chunks)

            st.write("Fragmentos relevantes:")
            for chunk in relevant_chunks:
                st.write(chunk)

            # Generar el prompt para Bedrock
            prompt = f"Pregunta: {user_query}\n\nInformación relevante:\n" + "\n".join(
                [chunk.page_content if hasattr(chunk, 'page_content') else chunk for chunk in relevant_chunks]
            ) + "\n\nRespuesta:"

            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,

            response = vector_store_model.invoke_bedrock(prompt, aws_access_key_id, aws_secret_access_key)
            st.subheader("Respuesta generada:")
            st.write(response)

            st.write("Información de la ontología encontrada:")
            if ontology_info:
                for subj, pred, obj in ontology_info:
                    st.write(f"Sujeto: {subj}, Predicado: {pred}, Objeto: {obj}")
            else:
                st.write("No se encontró información en la ontología.")

            st.write("Información de la base de datos encontrada:")
            if db_info:
                for table_name, records in db_info.items():
                    st.write(f"Tabla: {table_name}")
                    st.write(pd.DataFrame(records))
            else:
                st.write("No se encontró información en la base de datos.")

            st.write("Tablas extraídas de PDFs:")
            for table in pdf_tables:
                st.write(table)

            st.session_state.chat_history.append({
                "user": user_query,
                "ontology_info": ontology_info,
                "db_info": db_info,
                "relevant_chunks": relevant_chunks,
                "response": response
            })
        else:
            st.warning("Por favor ingrese una consulta.")
    
    # Mostrar historial de chat
    st.header("Historial de Chat")
    for entry in st.session_state.chat_history:
        st.subheader("Consulta del Usuario:")
        st.write(entry.get("user", ""))
        st.subheader("Información de la Ontología:")
        if entry.get("ontology_info"):
            for subj, pred, obj in entry["ontology_info"]:
                st.write(f"Sujeto: {subj}, Predicado: {pred}, Objeto: {obj}")
        else:
            st.write("No se encontró información en la ontología.")

        st.subheader("Información de la Base de Datos:")
        if entry.get("db_info"):
            for table_name, records in entry["db_info"].items():
                st.write(f"Tabla: {table_name}")
                st.write(pd.DataFrame(records))
        else:
            st.write("No se encontró información en la base de datos.")

        st.subheader("Fragmentos Relevantes de PDFs:")
        if entry.get("relevant_chunks"):
            for chunk in entry["relevant_chunks"]:
                st.write(chunk)
        else:
            st.write("No se encontraron fragmentos relevantes en los PDFs.")

        st.subheader("Respuesta del Sistema:")
        st.write(entry.get("response", ""))

    # Mostrar botón para descargar historial de chat
    download_chat_history(st.session_state.chat_history)
else:
    st.warning("Por favor, cargue al menos un archivo.")
