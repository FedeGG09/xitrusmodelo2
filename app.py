import openai
import streamlit as st
import tempfile
import os
import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import json
import pandas as pd 

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Inicialización de chat_history si no está definido en st.session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Función para cargar documentos PDF
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

# Función para extraer tablas de archivos PDF
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

# Función para dividir textos en fragmentos
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Clase para el modelo de almacenamiento de vectores
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

    def invoke_openai(self, prompt, openai_api_key):
        openai.api_key = openai_api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=450,
                n=1,
                stop=None,
                temperature=0.7,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0]['message']['content'].strip()
            else:
                return "Lo siento, no pude generar una respuesta para tu pregunta."
        except openai.error.OpenAIError as e:
            return f"Ocurrió un error al procesar la respuesta de OpenAI: {str(e)}"
        except Exception as e:
            return f"Ocurrió un error inesperado: {str(e)}"

# Parámetros del modelo definidos directamente en el código
model_name = "nombre_del_modelo"  # Reemplaza con el nombre real del modelo
model_version = "version_del_modelo"  # Reemplaza con la versión real del modelo

# Instancia de VectorStoreModel
vector_store_model = VectorStoreModel(model_name, model_version)

# Título de la aplicación
st.title("XITRUS - CHAT")

# Cargar archivos PDF
uploaded_files = st.file_uploader("Carga tus archivos PDF", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    # Cargar y procesar archivos PDF
    pdf_documents = load_documents(uploaded_files)  # Cargar documentos PDF
    pdf_tables = extract_tables_from_pdfs(uploaded_files)  # Extraer tablas de archivos PDF

    # Entrada de consulta del usuario
    user_query = st.text_input("Ingrese su consulta:")

    # Seleccionar tono emocional
    emotional_tone = st.selectbox("Seleccionar tono emocional", ["positivo", "neutral", "empático", "humorístico", "instructivo", "alentador"])

    # Seleccionar estilo
    style = st.selectbox("Seleccionar estilo", ["neutral", "formal", "informal", "profesional"])

    # Botón de búsqueda
    if st.button("Buscar"):
        if user_query:
            relevant_chunks = []
            if pdf_documents:
                pdf_chunks = split_text_into_chunks(pdf_documents)  # Dividir documentos en fragmentos
                vector_store = vector_store_model.get_vectorstore(pdf_chunks)
                relevant_chunks = vector_store_model.get_relevant_chunks(vector_store, user_query, pdf_chunks)

            st.write("Fragmentos relevantes:")
            for chunk in relevant_chunks:
                st.write(chunk)

            # Generar el prompt para OpenAI
            prompt = f"Pregunta: {user_query}\n\nInformación relevante:\n" + "\n".join(
                [chunk.page_content if hasattr(chunk, 'page_content') else chunk for chunk in relevant_chunks]
            ) + "\n\nRespuesta:"

            response = vector_store_model.invoke_openai(prompt, openai_api_key)
            st.subheader("Respuesta generada:")
            st.write(response)
