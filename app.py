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

# =====================================================
# 1. CONFIGURACIÓN DE GROQ
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
# 2. BASE DE CONOCIMIENTO RAG
# =====================================================
contenido_conocimiento = """
Ingeniería de Sistemas (UNJBG): Duración de 5 años (10 ciclos académicos). Pertenece a la Facultad de Ingeniería.
Medicina Humana (UNJBG): Duración de 6 años (incluye internado médico). Pertenece a la Facultad de Ciencias de la Salud.
Ingeniería Comercial (UNJBG): Duración de 5 años (10 ciclos académicos). Pertenece a la Facultad de Ingeniería.
Examen de Admisión UNJBG: Consta de 20 preguntas de opción múltiple divididas equilibradamente en razonamiento matemático, verbal y conocimientos generales.
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(contenido_conocimiento)
    temp_file_path = f.name

loader = TextLoader(temp_file_path, encoding='utf-8')
documentos = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = text_splitter.split_documents(documentos)
embeddings = FastEmbedEmbeddings()
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever_unjbg = vector_store.as_retriever(search_kwargs={"k": 2})
os.unlink(temp_file_path)

# =====================================================
# 3. ESQUEMA PYDANTIC Y FEW-SHOT
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
# 4. ORQUESTADOR PARALELO
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
# 5. CLASE DEL AGENTE (CON DOS HISTORIALES)
# =====================================================
class AgenteConsejeroUNJBG:
    def __init__(self):
        self.historial = []  # Para LangChain (mensajes)
        self.chat_history = []  # Para Gradio (tuplas)
    
    def generar_respuesta(self, mensaje_usuario):
        if not mensaje_usuario or not mensaje_usuario.strip():
            return self.chat_history, {"status": "Entrada vacía"}, self.historial
        
        try:
            resultado = orquestador.invoke({
                "message": mensaje_usuario,
                "chat_history": self.historial
            })
            
            respuesta_agente = resultado.get('respuesta_estudiante', '')
            metadatos = resultado.get('datos_estructurados', {})
            
            # Actualizar ambos historiales
            self.chat_history.append((mensaje_usuario, respuesta_agente))
            self.historial.append(HumanMessage(content=mensaje_usuario))
            self.historial.append(AIMessage(content=respuesta_agente))
            
            return self.chat_history, metadatos, self.historial
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.chat_history.append((mensaje_usuario, error_msg))
            return self.chat_history, {"status": "Error"}, self.historial
    
    def limpiar_memoria(self):
        self.historial = []
        self.chat_history = []
        return [], {"status": "Reiniciado"}, []

bot = AgenteConsejeroUNJBG()

# =====================================================
# 6. INTERFAZ GRADIO CON ESTILOS UNJBG
# =====================================================

# --- Tema CSS personalizado para la UNJBG ---
css = """
/* Variables de color UNJBG */
:root {
    --unjbg-bg-main: #0A2240;
    --unjbg-green: #005035;
    --unjbg-header-bg: #FFFFFF;
    --unjbg-text-dark: #1E293B;
    --unjbg-json-bg: #111827;
    --unjbg-border-active: #005035;
    --unjbg-bg-input: #0F172A;
}

/* Estilo del encabezado */
.unjbg-header {
    background-color: var(--unjbg-header-bg) !important;
    border-bottom: 4px solid var(--unjbg-green) !important;
    padding: 1rem 1.5rem !important;
    border-radius: 0px !important;
}
.unjbg-header h1 {
    color: var(--unjbg-text-dark) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
.unjbg-header p {
    color: #64748B !important;
    font-size: 1rem !important;
    margin: 0 !important;
}

/* Fondo principal y bloques */
.gradio-container {
    background-color: var(--unjbg-bg-main) !important;
}
.block {
    background-color: var(--unjbg-bg-main) !important;
}
.main-container {
    background-color: var(--unjbg-bg-main) !important;
}

/* Botón "Enviar" verde institucional */
.btn-enviar {
    background-color: var(--unjbg-green) !important;
    color: white !important;
    border: none !important;
    transition: background-color 0.2s !important;
}
.btn-enviar:hover {
    background-color: #003B26 !important;
}

/* Panel JSON */
.json-container {
    background-color: var(--unjbg-json-bg) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    border: 1px solid #334155 !important;
}

/* Burbujas de chat */
.message {
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    margin: 0.25rem 0 !important;
}
.message.user {
    background-color: var(--unjbg-green) !important;
    color: white !important;
}
.message.assistant {
    background-color: #1E293B !important;
    color: #E2E8F0 !important;
}

/* Input de texto */
.input-area textarea {
    background-color: var(--unjbg-bg-input) !important;
    color: #CBD5E1 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.input-area textarea:focus {
    border-color: var(--unjbg-green) !important;
    box-shadow: 0 0 0 2px rgba(0, 80, 53, 0.3) !important;
}
"""

# --- Creación de la interfaz ---
with gr.Blocks(theme=gr.themes.Soft(), css=css, title="UNJBG - Consejero de Admisión") as demo:
    # Encabezado institucional
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            gr.Image(
                value="logo_unjbg.png",  
                show_label=False,
                container=False,
                height=80,
                interactive=False
            )
        with gr.Column(scale=4):
            gr.Markdown("""
            # Universidad Nacional Jorge Basadre Grohmann
            ### <span style="color:#005035">Agente Consejero de Admisión</span>
            Sistema Inteligente con RAG, LangChain y Groq
            """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Componente de chat (con estilos para burbujas)
            chatbot = gr.Chatbot(
                label="Conversación",
                height=480,
                bubble_full_width=False,
                avatar_images=(None, "🧑‍🏫"),
                show_copy_button=True,
                render_markdown=True,
                elem_classes="message",
            )
            
            with gr.Row(elem_classes="input-area"):
                msg = gr.Textbox(
                    label="Escribe tu consulta",
                    placeholder="Ej: Hola, me llamo Juan. Estoy nervioso por el examen...",
                    scale=4,
                    lines=2,
                )
                send_btn = gr.Button("Enviar", variant="primary", elem_classes="btn-enviar", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Reiniciar conversación", variant="secondary", size="sm")
                info_btn = gr.Button("Ayuda", variant="secondary", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### Perfil del Postulante (JSON)")
            json_box = gr.JSON(
                label="Datos estructurados",
                value={"status": "Esperando tu primera consulta..."},
                elem_classes="json-container",
            )
    
    # =============================================
    # FUNCIONES DE RESPUESTA
    # =============================================
    def respond(message, state):
        return bot.generar_respuesta(message)
    
    def clear():
        bot.limpiar_memoria()
        return [], {"status": "Reiniciado"}, []
    
    # =============================================
    # EVENTOS
    # =============================================
    send_btn.click(
        fn=respond,
        inputs=[msg, gr.State(bot.historial)],
        outputs=[chatbot, json_box, gr.State(bot.historial)]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, gr.State(bot.historial)],
        outputs=[chatbot, json_box, gr.State(bot.historial)]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )
    
    clear_btn.click(
        fn=clear,
        inputs=None,
        outputs=[chatbot, json_box, gr.State(bot.historial)]
    )
    
    def show_help():
        return """
        **📋 ¿Cómo usar el consejero?**
        
        1. **Preséntate**: Di tu nombre y cómo te sientes.
        2. **Pregunta sobre carreras**: Ej. "¿Cuánto dura Ingeniería de Sistemas?"
        3. **Consulta sobre el examen**: Ej. "¿Cómo es el examen de admisión?"
        4. **Expresa dudas**: Ej. "No sé qué carrera elegir"
        
        El sistema extraerá automáticamente tus datos en formato JSON.
        """
    
    info_btn.click(
        fn=show_help,
        inputs=None,
        outputs=[msg]
    )

# =====================================================
# 7. LANZAMIENTO PARA RENDER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)