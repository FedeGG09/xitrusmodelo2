import streamlit as st
import pandas as pd
import pyodbc
from sqlalchemy import create_engine
from rdflib import Graph, URIRef, RDF, RDFS, OWL, XSD, Literal, Namespace
from sklearn.feature_extraction.text import CountVectorizer
import tempfile
import pickle
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Funciones para cargar archivos
def load_ontology_file(uploaded_file):
    try:
        ontology = Graph()
        format = 'xml' if uploaded_file.name.endswith('.owl') else 'turtle'
        ontology.parse(file=uploaded_file, format=format)
        st.success(f"Archivo de ontología {uploaded_file.name} cargado exitosamente.")
        return ontology
    except Exception as e:
        st.error(f"Error al cargar el archivo de ontología: {str(e)}")
        return None

def load_excel_file(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        data = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
        st.success(f"Archivo Excel {uploaded_file.name} cargado exitosamente.")
        return data
    except Exception as e:
        st.error(f"Error al cargar el archivo Excel: {str(e)}")
        return None

def load_access_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.accdb') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.close()
            conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={tmp_file.name};"
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
            data = {table: pd.read_sql(f'SELECT * FROM [{table}]', conn) for table in tables}
            st.success(f"Archivo Access {uploaded_file.name} cargado exitosamente.")
            return data
    except Exception as e:
        st.error(f"Error al cargar el archivo Access: {str(e)}")
        return None

def load_bak_file(uploaded_file):
    # Esta función requiere la implementación del procedimiento adecuado para restaurar BAK en SQL Server.
    st.warning("Restauración de archivos BAK no implementada. Retorna una lista vacía de tablas.")
    return {}

def get_tables_from_db(server_ip, server_login, server_password, database_name):
    try:
        connection_string = f"mssql+pyodbc://{server_login}:{server_password}@{server_ip}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            tables = pd.read_sql("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", connection)
            table_names = tables['TABLE_NAME'].tolist()
            return table_names, connection_string
    except Exception as e:
        st.error(f"Error al conectar con SQL Server: {str(e)}")
        return [], ""

# Funciones de ontologización, vectorización y tokenización
def get_xsd_type(dtype):
    dtype = dtype.lower()
    if "int" in dtype:
        return XSD.integer
    elif "float" in dtype or "double" in dtype:
        return XSD.float
    elif "datetime" in dtype or "date" in dtype:
        return XSD.dateTime
    elif "bool" in dtype:
        return XSD.boolean
    else:
        return XSD.string

def sanitize_for_uri(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "").replace("-", "_").replace("/", "_")

def ontologize_database(data):
    ontology = Graph()
    example_ns = Namespace("http://example.org/")
    ontology.bind("ex", example_ns)
    ontology.bind("xsd", XSD)
    ontology.bind("owl", OWL)
    ontology.bind("rdfs", RDFS)

    for source, dfs in data.items():
        if isinstance(dfs, Graph):
            continue  # Omite los objetos de tipo Graph
        for table_name, df in dfs.items():
            if not isinstance(df, pd.DataFrame):
                st.error(f"La tabla {table_name} no es un DataFrame.")
                continue
            sanitized_table_name = sanitize_for_uri(table_name)
            table_class = URIRef(f"http://example.org/{sanitized_table_name}")
            ontology.add((table_class, RDF.type, OWL.Class))

            for column in df.columns:
                sanitized_column = sanitize_for_uri(column)
                column_property = URIRef(f"http://example.org/{sanitized_table_name}/{sanitized_column}")
                dtype = df[column].dtype
                xsd_type = get_xsd_type(str(dtype))

                # Añadir propiedades de anotación
                ontology.add((column_property, RDFS.label, Literal(column)))

                if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_datetime64_any_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
                    # Añadir DatatypeProperty
                    ontology.add((column_property, RDF.type, OWL.DatatypeProperty))
                    ontology.add((column_property, RDFS.range, xsd_type))
                else:
                    # Añadir ObjectProperty
                    related_class = URIRef(f"http://example.org/{sanitize_for_uri(column).replace('_id', '')}")
                    ontology.add((related_class, RDF.type, OWL.Class))
                    object_property = URIRef(f"http://example.org/has_{sanitize_for_uri(column).replace('_id', '')}")
                    ontology.add((object_property, RDF.type, OWL.ObjectProperty))
                    ontology.add((object_property, RDFS.domain, table_class))
                    ontology.add((object_property, RDFS.range, related_class))
                    ontology.add((object_property, RDFS.label, Literal(f"has {column.replace('_id', '')}")))

                ontology.add((column_property, RDFS.domain, table_class))
                    
    return ontology

def vectorize_database(data):
    texts = []
    for source, dfs in data.items():
        if isinstance(dfs, Graph):
            continue  # Omite los objetos de tipo Graph
        for table_name, df in dfs.items():
            if not isinstance(df, pd.DataFrame):
                continue
            texts.append(table_name)
            texts.extend(df.columns)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        pickle.dump(vectorizer, tmp_file)
        vectorizer_file_path = tmp_file.name

    return vectorizer, vectorizer_file_path

def tokenize_vectorizer(vectorizer_file_path):
    with open(vectorizer_file_path, 'rb') as file:
        vectorizer = pickle.load(file)

    tokens = []
    for word in vectorizer.get_feature_names_out():
        tokens.extend(word_tokenize(word))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        pickle.dump(tokens, tmp_file)
        tokenized_file_path = tmp_file.name

    return tokenized_file_path

# Configuración de la página
st.set_page_config(page_title="Generador de Ontologías y Vista Previa", layout="wide")
st.title("Generador de Ontologías y Vista Previa de Bases de Datos")

# Sidebar para cargar archivos
st.sidebar.header("Cargar Archivos")
uploaded_owl = st.sidebar.file_uploader("Cargar OWL", type=["owl"], accept_multiple_files=True)
uploaded_ttl = st.sidebar.file_uploader("Cargar TTL", type=["ttl"], accept_multiple_files=True)
uploaded_excel = st.sidebar.file_uploader("Cargar Excel", type=["xlsx"], accept_multiple_files=True)
uploaded_access = st.sidebar.file_uploader("Cargar Access", type=["accdb"], accept_multiple_files=True)
uploaded_bak = st.sidebar.file_uploader("Cargar BAK", type=["bak"], accept_multiple_files=True)

# Variables para almacenar los datos
ontologies = []
data = {}

# Cargar archivos y mostrar vista previa
for uploaded_file in uploaded_owl:
    ontology = load_ontology_file(uploaded_file)
    if ontology:
        ontologies.append(ontology)

for uploaded_file in uploaded_ttl:
    ontology = load_ontology_file(uploaded_file)
    if ontology:
        ontologies.append(ontology)

for uploaded_file in uploaded_excel:
    excel_data = load_excel_file(uploaded_file)
    if excel_data:
        data[f'excel_{uploaded_file.name}'] = excel_data

for uploaded_file in uploaded_access:
    access_data = load_access_file(uploaded_file)
    if access_data:
        data[f'access_{uploaded_file.name}'] = access_data

for uploaded_file in uploaded_bak:
    bak_data = load_bak_file(uploaded_file)
    if bak_data:
        data[f'bak_{uploaded_file.name}'] = bak_data

# Mostrar vista previa de las ontologías
if ontologies:
    st.header("Vista Previa de las Ontologías")
    for ontology in ontologies:
        turtle_data = ontology.serialize(format='turtle')
        st.code(turtle_data, language='turtle')

# Mostrar vista previa de las bases de datos
if data:
    st.header("Vista Previa de los Datos")
    for source, dfs in data.items():
        st.subheader(f"Fuente: {source}")
        for table_name, df in dfs.items():
            st.subheader(f"Tabla: {table_name}")
            st.dataframe(df.head())

# Botones para ontologizar, vectorizar y tokenizar
if data:
    st.header("Opciones de Procesamiento")
    if st.button("Ontologizar Estructura"):
        ontology = ontologize_database(data)
        st.success("Ontología generada exitosamente.")
        turtle_data = ontology.serialize(format='turtle')
        st.code(turtle_data, language='turtle')

        # Botón para descargar la ontología en formato Turtle
        st.download_button(
            label="Descargar Ontología Generada",
            data=turtle_data,
            file_name="ontologia_generada.ttl",
            mime="text/turtle"
        )

    if st.button("Vectorizar Estructura"):
        vectorizer, vectorizer_file_path = vectorize_database(data)
        st.success(f"Vectorizador generado exitosamente. Guardado en: {vectorizer_file_path}")

        # Leer el contenido del archivo pickle para descarga
        with open(vectorizer_file_path, 'rb') as file:
            vectorizer_data = file.read()

        # Botón para descargar la estructura vectorizada
        st.download_button(
            label="Descargar Vectorizador",
            data=vectorizer_data,
            file_name="vectorizador.pkl",
            mime="application/octet-stream"
        )

    if st.button("Tokenizar Estructura"):
        vectorizer_file_path = vectorize_database(data)[1]  # Obtener el archivo vectorizado
        tokenized_file_path = tokenize_vectorizer(vectorizer_file_path)
        st.success(f"Estructura tokenizada generada exitosamente. Guardada en: {tokenized_file_path}")

        # Leer el contenido del archivo pickle para descarga
        with open(tokenized_file_path, 'rb') as file:
            tokenized_data = file.read()

        # Botón para descargar la estructura tokenizada
        st.download_button(
            label="Descargar Tokenizador",
            data=tokenized_data,
            file_name="tokenizador.pkl",
            mime="application/octet-stream"
        )

# Conexión a SQL Server y carga de tablas
st.sidebar.header("Conectar a SQL Server")
server_ip = st.sidebar.text_input("IP del Servidor")
server_login = st.sidebar.text_input("Usuario", value="", type="default")
server_password = st.sidebar.text_input("Contraseña", value="", type="password")
database_name = st.sidebar.text_input("Nombre de la Base de Datos")

if st.sidebar.button("Conectar y Cargar Tablas"):
    table_names, connection_string = get_tables_from_db(server_ip, server_login, server_password, database_name)
    if table_names:
        try:
            conn = pyodbc.connect(connection_string)
            data['sqlserver'] = {table: pd.read_sql(f"SELECT * FROM {table}", conn) for table in table_names}
            st.sidebar.success("Tablas cargadas exitosamente.")
        except Exception as e:
            st.sidebar.error(f"Error al cargar las tablas: {str(e)}")
    else:
        st.sidebar.error("No se encontraron tablas o hubo un error en la conexión.")

# Mostrar vista previa de las tablas SQL Server
if 'sqlserver' in data:
    st.header("Vista Previa de SQL Server")
    for table, df in data['sqlserver'].items():
        st.subheader(f"{table}")
        st.dataframe(df.head())