import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from pydantic import BaseModel, Field
import gradio as gr
import tempfile
import os

# =====================================================
# 1. CONFIGURACIÓN DE GROQ (con variable de entorno)
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY no configurada. Configúrala en Render como variable de entorno.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# =====================================================
# 2. BASE DE CONOCIMIENTO RAG (directo en el código)
# =====================================================
contenido_conocimiento = """
Ingeniería de Sistemas (UNJBG): Duración de 5 años (10 ciclos académicos). Pertenece a la Facultad de Ingeniería.
Medicina Humana (UNJBG): Duración de 6 años (incluye internado médico). Pertenece a la Facultad de Ciencias de la Salud.
Ingeniería Comercial (UNJBG): Duración de 5 años (10 ciclos académicos). Pertenece a la Facultad de Ingeniería.
Examen de Admisión UNJBG: Consta de 20 preguntas de opción múltiple divididas equilibradamente en razonamiento matemático, verbal y conocimientos generales.
"""

# Crear el archivo temporal
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(contenido_conocimiento)
    temp_file_path = f.name

# Cargar y procesar
loader = TextLoader(temp_file_path, encoding='utf-8')
documentos = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = text_splitter.split_documents(documentos)
embeddings = FastEmbedEmbeddings()
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever_unjbg = vector_store.as_retriever(search_kwargs={"k": 2})

# Limpiar archivo temporal
os.unlink(temp_file_path)

# =====================================================
# 3. ESQUEMA PYDANTIC Y FEW-SHOT PROMPTING
# =====================================================
class PerfilPostulante(BaseModel):
    nombre: Optional[str] = Field(default="Anónimo", description="Nombre extraído del postulante")
    carrera_interes: Optional[str] = Field(default="Por definir", description="Carrera de interés")
    nivel_certeza: str = Field(description="Nivel de certeza de su decisión (ej. 'Seguro', 'Indeciso')")
    analisis_emocional: str = Field(description="Estado emocional detectado (ej. 'Estresado', 'Tranquilo')")

json_parser = JsonOutputParser(pydantic_object=PerfilPostulante)

ejemplos_postulantes = [
    {
        "input": "No sé qué estudiar... dudo entre seguir Comercial o irme a letras. Estoy muy confundido",
        "output": '{"nombre": "Anónimo", "carrera_interes": "Por definir", "nivel_certeza": "Indeciso", "analisis_emocional": "Confundido"}'
    },
    {
        "input": "Hola, soy Sofía. Quiero Medicina, estoy segura pero algo nerviosa por el examen",
        "output": '{"nombre": "Sofía", "carrera_interes": "Medicina", "nivel_certeza": "Seguro", "analisis_emocional": "Nerviosa"}'
    }
]

prompt_ejemplos = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=ejemplos_postulantes,
    example_prompt=prompt_ejemplos
)

json_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un analizador de datos rígido de la oficina de admisión de la UNJBG. Tu tarea es extraer información estructurada de los postulantes en formato JSON."),
    few_shot_prompt,
    ("human", "Analiza el siguiente mensaje de un postulante:\n{message}")
])

json_partial_template = json_template.partial(format_instructions=json_parser.get_format_instructions())
json_chain = json_partial_template | llm | json_parser

# =====================================================
# 4. ORQUESTADOR PARALELO (el cerebro)
# =====================================================
def procesar_consulta_estudiante(x):
    mensaje = x["message"]
    historial = x.get("chat_history", [])
    
    docs = retriever_unjbg.invoke(mensaje)
    contexto = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = SystemMessage(content=f"""
Eres un orientador vocacional empático, amable y profesional de la UNJBG (Universidad Nacional Jorge Basadre Grohmann) de Tacna, Perú.
Tu objetivo es guiar a los postulantes, calmar sus nervios y resolver sus dudas sobre carreras y el proceso de admisión.

INFORMACIÓN OFICIAL RECUPERADA DE LA BASE DE CONOCIMIENTOS (RAG):
{contexto}

**INSTRUCCIÓN CRÍTICA:** Debes responder estrictamente basándote en la información del RAG. Si el RAG dice que una carrera dura 6 años, debes decir que dura 6 años. NO uses tu conocimiento externo. Responde solo con el contexto provisto.
""")
    
    mensajes = [system_prompt] + historial + [HumanMessage(content=mensaje)]
    respuesta = llm.invoke(mensajes)
    return respuesta.content

preparar_json = RunnableLambda(lambda x: {"message": x["message"]})
rama_json = preparar_json | json_chain
rama_agente = RunnableLambda(procesar_consulta_estudiante)

orquestador = RunnableParallel({
    "respuesta_estudiante": rama_agente,
    "datos_estructurados": rama_json
})

# =====================================================
# 5. CLASE PARA GRADIO CON MEMORIA POR SESIÓN
# =====================================================
class AgenteConsejeroUNJBG:
    def __init__(self):
        self.historial = []
    
    def generar_respuesta(self, pregunta_usuario):
        if not pregunta_usuario or not pregunta_usuario.strip():
            return "Por favor, escribe una consulta válida.", {"status": "Entrada vacía"}
        
        try:
            resultado = orquestador.invoke({
                "message": pregunta_usuario,
                "chat_history": self.historial
            })
            
            respuesta = resultado.get('respuesta_estudiante', '')
            metadatos = resultado.get('datos_estructurados', {})
            
            self.historial.append(HumanMessage(content=pregunta_usuario))
            self.historial.append(AIMessage(content=respuesta))
            
            return respuesta, metadatos
        except Exception as e:
            return f"❌ Error: {str(e)}", {"status": "Error", "detalle": str(e)}
    
    def limpiar_memoria(self):
        self.historial = []
        return "🔄 Conversación reiniciada", {"status": "Reiniciado"}

bot = AgenteConsejeroUNJBG()

# =====================================================
# 6. INTERFAZ GRADIO (Adaptada para Render)
# =====================================================
with gr.Blocks(theme=gr.themes.Soft(), title="UNJBG - Consejero Vocacional") as demo:
    gr.Markdown("""
    # 🎓 Agente Consejero de Admisión UNJBG (Tacna, Perú)
    ### Sistema Inteligente con RAG Vectorial y Extracción Estructurada JSON
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="✏️ Consulta del Postulante",
                placeholder="Ej: Hola, me llamo Juan. Estoy estresado y quiero saber cuánto dura Ingeniería de Sistemas",
                lines=3
            )
            with gr.Row():
                btn_enviar = gr.Button("🚀 Enviar Consulta", variant="primary")
                btn_limpiar = gr.Button("🔄 Reiniciar Conversación", variant="secondary")
            
            output_box = gr.Textbox(
                label="💬 Respuesta del Consejero Vocacional",
                interactive=False,
                lines=10
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Extracción de Metadatos (JSON)")
            json_box = gr.JSON(
                label="Perfil del Postulante",
                value={"status": "Esperando consulta..."}
            )
    
    # Eventos
    btn_enviar.click(
        fn=bot.generar_respuesta,
        inputs=input_box,
        outputs=[output_box, json_box]
    )
    btn_limpiar.click(
        fn=bot.limpiar_memoria,
        inputs=None,
        outputs=[output_box, json_box]
    )
    input_box.submit(
        fn=bot.generar_respuesta,
        inputs=input_box,
        outputs=[output_box, json_box]
    )

# =====================================================
# 7. LANZAMIENTO PARA RENDER (PUERTO 10000)
# =====================================================
if __name__ == "__main__":
    # Render asigna el puerto 10000 por defecto
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)