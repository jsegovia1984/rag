import os
import textwrap
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Colors:
    """Clase para definir colores de texto en la terminal."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --- CONFIGURACIÓN ---
# Nombre exacto de tu carpeta de documentos
FOLDER_PATH = "./Fallos y dictamenes"
# Directorio donde se guardará la base de datos de Chroma
CHROMA_PERSIST_DIR = "chroma_db"

# Verificar que la carpeta de documentos existe
if not os.path.exists(FOLDER_PATH):
    raise FileNotFoundError(f"No encuentro la carpeta de documentos: {FOLDER_PATH}")

# --- EMBEDDINGS Y ALMACENAMIENTO ---
# Definimos la función de embeddings que vamos a usar
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# Comprobamos si la base de datos ya existe
if os.path.exists(CHROMA_PERSIST_DIR):
    # Si existe, la cargamos
    print(f"Cargando base de datos de vectores desde '{CHROMA_PERSIST_DIR}'...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function
    )
    print("¡Base de datos cargada con éxito!")
    print(f"\n{Colors.YELLOW}INFO: Para regenerar la base de datos (por ejemplo, si agregaste archivos), borra la carpeta '{CHROMA_PERSIST_DIR}'.{Colors.ENDC}")

else:
    # Si no existe, la creamos desde cero
    print(f"No se encontró una base de datos. Creando una nueva desde '{FOLDER_PATH}'...")

    # 1. CARGAR DATOS
    print("Cargando PDFs... esto puede tardar un poco.")
    loader = DirectoryLoader(
        FOLDER_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    print(f"Se cargaron {len(docs)} páginas de documentos.")

    # 2. DIVIDIR (CHUNKING)
    print("Dividiendo los documentos en fragmentos (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Se han creado {len(splits)} fragmentos de texto.")

    # 3. CREAR Y GUARDAR LA BASE DE DATOS
    print("Creando y guardando la base de datos de vectores (embeddings)... esto puede tardar bastante.")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR  # ¡Clave para guardar la BD!
    )
    print(f"¡Base de datos creada y guardada en '{CHROMA_PERSIST_DIR}' con éxito!")

# --- CONFIGURACIÓN DEL RETRIEVER ---
retriever = vectorstore.as_retriever()

# 4. DEFINIR EL LLM
llm = ChatOllama(model="llama3")

# 5. EL PROMPT
template = """Eres un asistente legal experto. Responde basándote SOLO en los siguientes dictámenes y sentencias proporcionados.
Si la información no está en el contexto, di que no lo sabes.

Contexto:
{context}

Pregunta: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. LA CADENA
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. INTERFAZ DE CHAT MEJORADA
print(f"\n{Colors.BOLD}--- Sistema de Asistencia Legal ---{Colors.ENDC}")
print("Escribe tu consulta sobre los documentos. Escribe 'salir' para terminar.")

while True:
    try:
        # Usamos un prompt más limpio y con color
        print(f"\n{'-'*80}")
        pregunta = input(f"{Colors.BLUE}{Colors.BOLD}Tu pregunta > {Colors.ENDC}")
        
        if pregunta.lower().strip() in ["salir", "exit"]:
            print(f"\n{Colors.YELLOW}Gracias por usar el asistente. ¡Hasta luego!{Colors.ENDC}")
            break
        
        if not pregunta.strip():
            continue

        # Indicador de que está trabajando
        print(f"\n{Colors.YELLOW}Analizando documentos y generando respuesta...{Colors.ENDC}")
        
        response = rag_chain.invoke(pregunta)
        
        # Formateo de la respuesta para que sea más legible
        wrapped_response = textwrap.fill(response, width=80, initial_indent="  ", subsequent_indent="  ")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}Asistente Legal:{Colors.ENDC}\n{wrapped_response}")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\n{Colors.YELLOW}Interrupción detectada. Saliendo del programa...{Colors.ENDC}")
        break