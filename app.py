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
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
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
# 5. CLASE MEJORADA PARA GRADIO (con historial)
# =====================================================
class AgenteConsejeroUNJBG:
    def __init__(self):
        self.historial_messages = []  # para LangChain
        self.historial_gradio = []    # para el componente Chatbot (lista de tuplas)

    def generar_respuesta(self, pregunta_usuario):
        if not pregunta_usuario or not pregunta_usuario.strip():
            return self.historial_gradio, {"status": "Entrada vacía"}
        
        try:
            resultado = orquestador.invoke({
                "message": pregunta_usuario,
                "chat_history": self.historial_messages
            })
            
            respuesta = resultado.get('respuesta_estudiante', '')
            metadatos = resultado.get('datos_estructurados', {})
            
            # Actualizar historial de LangChain
            self.historial_messages.append(HumanMessage(content=pregunta_usuario))
            self.historial_messages.append(AIMessage(content=respuesta))
            
            # Actualizar historial de Gradio (tupla: (usuario, asistente))
            self.historial_gradio.append((pregunta_usuario, respuesta))
            
            return self.historial_gradio, metadatos
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.historial_gradio.append((pregunta_usuario, error_msg))
            return self.historial_gradio, {"status": "Error", "detalle": str(e)}
    
    def limpiar_memoria(self):
        self.historial_messages = []
        self.historial_gradio = []
        return self.historial_gradio, {"status": "Reiniciado"}

bot = AgenteConsejeroUNJBG()

# =====================================================
# 6. INTERFAZ GRADIO CON DISEÑO INSTITUCIONAL (sin emojis excesivos)
# =====================================================

# Tema personalizado: colores sobrios
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f5f7fa",
    block_title_text_color="#1a2a4a",
    block_label_text_color="#1a2a4a",
    button_primary_background_fill="#1a3a6a",
    button_primary_background_fill_hover="#0e2a4a",
    button_primary_text_color="white",
    button_secondary_background_fill="#e8edf4",
    button_secondary_background_fill_hover="#d0dae8",
    input_background_fill="white",
    input_border_color="#c0c8d8",
    shadow_spread="4px",
)

with gr.Blocks(theme=custom_theme, title="UNJBG - Consejero Vocacional", css="""
    .header-container {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 6px solid #1a3a6a;
    }
    .header-title {
        color: #1a2a4a;
        font-weight: 600;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #4a5a7a;
        font-weight: 400;
        font-size: 1.1rem;
        margin: 0.2rem 0 0 0;
    }
    .header-badge {
        background: #e8edf4;
        color: #1a3a6a;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .footer-note {
        text-align: center;
        color: #7a8a9a;
        font-size: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e6ee;
        padding-top: 1rem;
    }
""") as demo:
    
    # Encabezado personalizado (sin emojis)
    gr.HTML("""
    <div class="header-container">
        <h1 class="header-title">Universidad Nacional Jorge Basadre Grohmann</h1>
        <p class="header-subtitle">Oficina de Admisión · Consejero Vocacional</p>
        <span class="header-badge">Sistema Inteligente con RAG</span>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversación",
                height=480,
                bubble_full_width=False,
                avatar_images=(None, "🏛️"),  # opcional: ícono institucional
                show_copy_button=True,
                render_markdown=True,
                elem_id="chatbot"
            )
            with gr.Row():
                msg = gr.Textbox(
                    label="Escribe tu consulta",
                    placeholder="Ejemplo: Hola, me llamo Juan. Estoy nervioso por el examen...",
                    scale=4,
                    lines=2,
                    elem_id="input-msg"
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("Reiniciar conversación", variant="secondary", size="sm")
                info_btn = gr.Button("Ayuda", variant="secondary", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### Datos del postulante (JSON)")
            json_box = gr.JSON(
                label="Perfil extraído",
                value={"status": "Esperando consulta..."},
                elem_id="json-box"
            )
    
    # Estado: usamos el historial de la clase directamente
    # Pero necesitamos una función para actualizar el chatbot y json
    def respond(message, state_history):
        # state_history no se usa realmente, porque la clase guarda el estado
        # pero lo mantenemos por compatibilidad
        historial, metadatos = bot.generar_respuesta(message)
        return historial, metadatos
    
    def clear_conversation():
        historial, metadatos = bot.limpiar_memoria()
        return historial, metadatos
    
    def show_help():
        return """
        **Uso del consejero**
        
        - Preséntate con tu nombre y cómo te sientes.
        - Pregunta sobre duración de carreras, plan de estudios o proceso de admisión.
        - Expresa tus dudas vocacionales para recibir orientación.
        
        El sistema extraerá automáticamente tus datos en formato JSON.
        """
    
    # Eventos
    send_btn.click(
        fn=respond,
        inputs=[msg],
        outputs=[chatbot, json_box]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg],
        outputs=[chatbot, json_box]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=None,
        outputs=[chatbot, json_box]
    )
    
    info_btn.click(
        fn=show_help,
        inputs=None,
        outputs=[msg]
    )
    
    # Pie de página
    gr.HTML("""
    <div class="footer-note">
        © 2026 UNJBG · Tacna, Perú · Desarrollado con LangChain, Groq y Gradio
    </div>
    """)

# =====================================================
# 7. LANZAMIENTO
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)