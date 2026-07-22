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
# 6. INTERFAZ GRADIO PERSONALIZADA (CON HISTORIAL)
# =====================================================

# Tema personalizado con colores de la UNJBG
unjbg_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Poppins"),
).set(
    block_title_text_color="#003366",      # Azul oscuro institucional
    block_label_text_color="#003366",
    button_primary_background_fill="#003366",
    button_primary_background_fill_hover="#001a33",
    button_primary_text_color="white",
    input_background_fill="#f0f4f8",
    input_border_color="#003366",
    shadow_spread="2px",
)

with gr.Blocks(theme=unjbg_theme, title="UNJBG - Consejero Vocacional") as demo:
    # Encabezado con logo
    gr.Markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #003366, #004c99); border-radius: 12px; margin-bottom: 2rem;">
        <img src="logo_unjbg.png" style="height: 60px; margin-bottom: 10px;">
        <h1 style="...">UNJBG - Tacna</h1>
        <h2 style="color: #ffd700; margin: 0; font-weight: 300; font-size: 1.4rem;">Agente Consejero de Admisión</h2>
        <p style="color: #e0e8f0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Sistema Inteligente con RAG, LangChain y Groq</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Componente de chat con historial
            chatbot = gr.Chatbot(
                label="💬 Conversación",
                height=450,
                bubble_full_width=False,
                avatar_images=(None, "🧑‍🏫"),
                show_copy_button=True,
                render_markdown=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="✏️ Escribe tu consulta",
                    placeholder="Ej: Hola, me llamo Juan. Estoy nervioso por el examen...",
                    scale=4,
                    lines=2,
                )
                send_btn = gr.Button("🚀 Enviar", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🔄 Reiniciar Conversación", variant="secondary", size="sm")
                info_btn = gr.Button("ℹ️ Ayuda", variant="secondary", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Perfil del Postulante (JSON)")
            json_box = gr.JSON(
                label="Datos Estructurados",
                value={"status": "Esperando tu primera consulta..."},
            )
    
    # Estado para el historial (se mantiene en la sesión)
    state = gr.State([])
    
    # =============================================
    # FUNCIONES DE RESPUESTA (con historial)
    # =============================================
    
    def respond(message, chat_history, json_state):
        if not message or not message.strip():
            return chat_history, {"status": "Entrada vacía"}, chat_history
        
        try:
            # Llamar al orquestador (como antes)
            resultado = orquestador.invoke({
                "message": message,
                "chat_history": chat_history
            })
            
            respuesta_agente = resultado.get('respuesta_estudiante', '')
            metadatos = resultado.get('datos_estructurados', {})
            
            # Construir el historial para Gradio (lista de tuplas)
            new_history = chat_history + [(message, respuesta_agente)]
            
            # Actualizar el estado interno (para memoria)
            chat_history.append(HumanMessage(content=message))
            chat_history.append(AIMessage(content=respuesta_agente))
            
            return new_history, metadatos, chat_history
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return chat_history + [(message, error_msg)], {"status": "Error"}, chat_history
    
    def clear_conversation():
        return [], {"status": "Conversación reiniciada"}, []
    
    # =============================================
    # EVENTOS
    # =============================================
    
    send_btn.click(
        fn=respond,
        inputs=[msg, state, state],
        outputs=[chatbot, json_box, state]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, state, state],
        outputs=[chatbot, json_box, state]
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=[msg]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=None,
        outputs=[chatbot, json_box, state]
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
# 7. LANZAMIENTO PARA RENDER (PUERTO 10000)
# =====================================================
if __name__ == "__main__":
    # Render asigna el puerto 10000 por defecto
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)