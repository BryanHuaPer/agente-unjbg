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
# 2. BASE DE CONOCIMIENTO RAG (CONTENIDO COMPLETO)
# =====================================================
contenido_conocimiento = """
# CANAL 1: CIENCIAS DE LA SALUD
# MEDICINA HUMANA
Duración: 6 años (12 semestres).
Grado: Bachiller en Medicina. Título: Médico Cirujano.
Campo laboral: Hospitales, clínicas, consultorios, salud pública, docencia e investigación.
Perfil del postulante: Vocación de servicio, capacidad de análisis, sensibilidad social, manejo de estrés.
Recomendación de egresado: "La medicina es una vocación de servicio. Cada paciente es una oportunidad para aprender y crecer."

# ENFERMERÍA
Duración: 5 años (10 semestres).
Grado: Bachiller en Enfermería. Título: Licenciado en Enfermería.
Campo laboral: Hospitales, clínicas, centros de salud, docencia, gestión en salud.
Perfil del postulante: Vocación de servicio, empatía, responsabilidad, capacidad de trabajo en equipo.
Recomendación de egresado: "La enfermería es el arte de cuidar. Cada día es una oportunidad para marcar la diferencia."

# ODONTOLOGÍA
Duración: 5 años (10 semestres).
Grado: Bachiller en Odontología. Título: Cirujano Dentista.
Campo laboral: Consultorios, clínicas, salud pública, docencia, investigación.
Perfil del postulante: Destreza manual, precisión, vocación de servicio, capacidad de análisis.
Recomendación de egresado: "La odontología combina ciencia y arte. La sonrisa de un paciente es la mejor recompensa."

# FARMACIA Y BIOQUÍMICA
Duración: 5 años (10 semestres).
Grado: Bachiller en Farmacia y Bioquímica. Título: Químico Farmacéutico.
Campo laboral: Farmacias, laboratorios, industria farmacéutica, docencia, investigación.
Perfil del postulante: Capacidad de análisis, precisión, responsabilidad, interés por la química y la biología.
Recomendación de egresado: "La farmacia es la ciencia de los medicamentos. Cada fórmula es una oportunidad para salvar vidas."

# CIENCIAS DE LA NUTRICIÓN
Duración: 5 años (10 semestres).
Grado: Bachiller en Nutrición. Título: Licenciado en Nutrición.
Campo laboral: Hospitales, clínicas, consultorios, industria alimentaria, docencia, investigación.
Perfil del postulante: Interés por la alimentación y la salud, capacidad de análisis, vocación de servicio.
Recomendación de egresado: "La nutrición es la base de la salud. Cada paciente es una oportunidad para educar y transformar."

# CANAL 2: INGENIERÍA
# INGENIERÍA DE SISTEMAS
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería de Sistemas. Título: Ingeniero de Sistemas.
Campo laboral: Desarrollo de software, consultoría tecnológica, gestión de TI, docencia, investigación.
Perfil del postulante: Razonamiento lógico, capacidad de abstracción, interés por la tecnología, trabajo en equipo.
Recomendación de egresado: "La tecnología es el futuro. Cada línea de código es una oportunidad para innovar."

# INGENIERÍA CIVIL
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería Civil. Título: Ingeniero Civil.
Campo laboral: Construcción, diseño de infraestructura, consultoría, docencia, investigación.
Perfil del postulante: Razonamiento espacial, capacidad de análisis, interés por la construcción, responsabilidad.
Recomendación de egresado: "La ingeniería civil construye el futuro. Cada obra es un legado para las generaciones."

# INGENIERÍA DE MINAS
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería de Minas. Título: Ingeniero de Minas.
Campo laboral: Minería, exploración, consultoría, docencia, investigación.
Perfil del postulante: Razonamiento lógico, capacidad de análisis, interés por la geología, responsabilidad.
Recomendación de egresado: "La minería es la columna vertebral del desarrollo. Cada proyecto es una oportunidad para crecer."

# INGENIERÍA INDUSTRIAL
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería Industrial. Título: Ingeniero Industrial.
Campo laboral: Industria, consultoría, gestión de operaciones, docencia, investigación.
Perfil del postulante: Razonamiento lógico, capacidad de análisis, liderazgo, interés por la optimización.
Recomendación de egresado: "La ingeniería industrial optimiza procesos. Cada mejora es una oportunidad para ser más eficientes."

# INGENIERÍA MECÁNICA
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería Mecánica. Título: Ingeniero Mecánico.
Campo laboral: Industria, diseño de máquinas, consultoría, docencia, investigación.
Perfil del postulante: Razonamiento espacial, capacidad de análisis, interés por la mecánica, trabajo en equipo.
Recomendación de egresado: "La ingeniería mecánica mueve el mundo. Cada máquina es una oportunidad para innovar."

# INGENIERÍA AGROINDUSTRIAL
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería Agroindustrial. Título: Ingeniero Agroindustrial.
Campo laboral: Agroindustria, consultoría, docencia, investigación.
Perfil del postulante: Interés por la agricultura y la industria, capacidad de análisis, trabajo en equipo.
Recomendación de egresado: "La agroindustria transforma el campo. Cada producto es una oportunidad para crecer."

# CANAL 3: CIENCIAS SOCIALES Y EDUCACIÓN
# EDUCACIÓN: IDIOMA EXTRANJERO
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la Educación. Título: Licenciado en educación, especialidad en Idioma Extranjero.
Campo laboral: Traducción de documentos comerciales. Docencia en EBR y Educación Superior. Formulación de programas de enseñanza del inglés. Mediación intercultural.
Perfil del postulante: Conocimiento pre-intermedio de Inglés. Competencias en comprensión y producción de textos. Tolerancia a la diversidad laboral y cultural.
Recomendación de egresado: "Mi formación me ha permitido adquirir herramientas pedagógicas efectivas, preparándome para inspirar y guiar en el crecimiento académico y personal de mis estudiantes."

# EDUCACIÓN: CIENCIAS DE LA NATURALEZA Y PROMOCIÓN EDUCATIVA AMBIENTAL
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la Educación. Título: Licenciado en Educación, Especialidad en Ciencias de la Naturaleza y Promoción Educativa Ambiental.
Campo laboral: Docencia en secundaria pública y privada. Docencia universitaria y en institutos superiores. Gestión y dirección de instituciones educativas.
Perfil del postulante: Vocación docente. Estilos de vida saludable. Iniciativa para indagar el mundo natural con conocimientos científicos. Conocimientos básicos de TICs.
Recomendación de egresado: "Estoy convencido que ser docente de las ciencias de la naturaleza y promoción educativa ambiental es la carrera más hermosa que puedes experimentar."

# ARTES
Duración: 5 años (10 semestres).
Grado: Bachiller en Artes Plásticas. Título: Licenciado en Artes Plásticas.
Campo laboral: Talleres artísticos públicos y privados. Centros de producción artística. Instituciones de investigación y difusión artística. Gestión, promoción cultural y curaduría. Docencia.
Perfil del postulante: Conocimiento de la historia del arte. Habilidad en TICs. Imaginación creativa y sensibilidad estética. Habilidad para gestionar proyectos culturales.
Recomendación de egresado: "Emprender la carrera de Artes en la UNJBG es un acto desafiante. Si te apasiona y te entregas, encontrarás tu identidad artística."

# CIENCIAS DE LA COMUNICACIÓN
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la Comunicación Social. Título: Licenciado en Comunicación Social en Periodismo y Relaciones Públicas.
Campo laboral: Medios de comunicación (radio, prensa, TV, internet). Relaciones públicas en instituciones estatales y privadas. Empresas de publicidad y marketing. Consultorías empresariales.
Perfil del postulante: Alto desempeño en expresión oral, escrita y audiovisual. Sensibilidad social y cultural. Interés por la lectura, el cine y la multimedia. Actitud investigativa.
Recomendación de egresado: "Comunicar es imaginar futuros posibles, transformar realidades y narrar con sentido. Los invito a sostener sus convicciones y no dejar de soñar."

# ARQUITECTURA
Duración: 5 años (10 semestres).
Grado: Bachiller en Arquitectura. Título: Arquitecto.
Campo laboral: Proyectista de obras nuevas y restauración. Constructor y ejecutor de obras. Asesor en gestión de proyectos inmobiliarios. Planificación urbana. Docencia e investigación.
Perfil del postulante: Creativo, innovador y líder. Conocimiento de cultura y realidad social. Sensibilidad artística. Ética y moral profesional.
Recomendación de egresado: "Lograr el primer puesto de mi promoción me permitió obtener una beca y titularme en el Magister de Administración de la Construcción en Chile."

# DERECHO Y CIENCIAS POLÍTICAS
Duración: 6 años (12 semestres).
Grado: Bachiller en Derecho y Ciencias Políticas. Título: Abogado.
Campo laboral: Poder Judicial y Ministerio Público. Ejercicio libre de la abogacía. Asesoría en el sector público y privado. Notaría y Registro. Asesoría política.
Perfil del postulante: Precisión para la lectura. Habilidades de análisis. Capacidad de comunicación escrita y oral. Sentido de justicia, responsabilidad y lealtad. Inclinación a la investigación.
Recomendación de egresado: "Confíen en sus capacidades y atrévanse a nuevos retos. No olviden que la competencia es con uno mismo, para ser mejor cada día."

# EDUCACIÓN INICIAL
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la educación. Título: Licenciado en Educación Inicial.
Campo laboral: Guarderías, nidos, centros de educación inicial. Programas gubernamentales (Cuna Más, PRONOEI, Wawa Wasi). Asesoramiento en estimulación temprana e inclusión educativa.
Perfil del postulante: Respeto y tolerancia. Aprecio por el arte para expresar el mundo personal. Gestión de proyectos de emprendimiento. Indagación del mundo físico. Destreza manual.
Recomendación de egresado: "La Dirección de Admisión invita a los estudiantes a participar en los procesos. ¡Sean basadrinos de corazón!"

# EDUCACIÓN PRIMARIA
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la educación. Título: Licenciado en Educación Primaria.
Campo laboral: Colegios públicos y privados. Diseño curricular y pedagogía. Capacitación y formación docente. Gestión educativa y dirección de instituciones.
Perfil del postulante: Habilidades comunicativas, matemáticas y gráficas. Capacidad de observación y análisis. Trabajo autónomo y en equipo. Aplicación de la ciencia y tecnología. Destreza manual.
Recomendación de egresado: "La Dirección de Admisión invita a los estudiantes a participar en los procesos. ¡Sean basadrinos de corazón!"

# PSICOLOGÍA
Duración: 6 años (12 semestres).
Grado: Bachiller en Psicología. Título: Licenciado en Psicología.
Campo laboral: Psicología social (ONGs, ministerios). Educación (colegios, universidades, orientación vocacional). Recursos humanos (empresas e instituciones). Psicología clínica (evaluación e intervención).
Perfil del postulante: Capacidad para la interacción social. Tolerancia. Conocimientos de cultura general. Comprensión de la conducta humana. Conocimiento informático. Capacidad analítica.
Recomendación de egresado: "Estudiar Psicología es iniciar un viaje hacia el corazón del ser humano. Es acompañar al otro en sus luchas internas y contribuir al bienestar colectivo."

# CANAL 4: CIENCIAS ACTUARIALES Y EMPRESARIALES
# CIENCIAS ADMINISTRATIVAS
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias Administrativas. Título: Licenciado en Administración.
Campo laboral: Organizaciones nacionales e internacionales (público o privado). Dirección y gestión estratégica, Finanzas, Marketing, Recursos Humanos, Producción y Logística. Emprendimientos innovadores.
Perfil del postulante: Comprensión lectora. Equilibrio emocional, respeto y disciplina. Capacidad de liderazgo. Conocimiento de economía y ofimática (Excel, Word).
Recomendación de egresado: "Lo aprendido en la carrera de Administración me ha permitido alcanzar mis metas personales y colaborar en el desarrollo del país."

# CIENCIAS CONTABLES Y FINANCIERAS
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias Contables y Financieras. Título: Contador Público.
Campo laboral: Asesoría contable a personas y empresas. Análisis de estados financieros. Auditoría de sistemas públicos y privados (NIA). Gestión de procesos contables. Evaluación de información económica y financiera.
Perfil del postulante: Comprensión lectora. Razonamiento matemático y lógico. Capacidad de liderazgo y trabajo en equipo. Conocimiento de economía y ofimática (Excel, Word).
Recomendación de egresado: "En las aulas de la UNJBG pude afianzar mi vocación. Me forjaron con la pasión de que, siendo contador, podemos cambiar al mundo."

# INGENIERÍA COMERCIAL
Duración: 5 años (10 semestres).
Grado: Bachiller en Ingeniería Comercial. Título: Ingeniero Comercial.
Campo laboral: Gestión, marketing, economía, finanzas, administración. Asesor en proyectos de inversión. Consultor en diversificación empresarial. Planeamiento estratégico. Impulso de PYMES.
Perfil del postulante: Aplicación de TICs. Comunicación verbal y no verbal. Conocimientos de gestión empresarial y negocios. Habilidades blandas para la formación integral.
Recomendación de egresado: "Recomiendo a todos los estudiantes aprovechar las oportunidades que ofrece la universidad y siempre dejar en alto a Tacna."

# ECONOMÍA AGRARIA
Duración: 5 años (10 semestres).
Grado: Bachiller en Ciencias con mención en Economía Agraria. Título: Ingeniero en Economía Agraria.
Campo laboral: Liderazgo en agronegocios y comercio exterior. Funciones administrativas en empresas. Entidades del Estado (planificación, presupuesto, logística). Análisis financiero en entidades bancarias.
Perfil del postulante: Comprensión de problemas socioeconómicos. Comunicación de ideas coherentes. Habilidades blandas. Compromiso con el estudio.
Recomendación de egresado: "Gracias a la formación recibida, he podido contribuir en espacios técnicos del estado vinculados a la gestión de la inversión pública. Lo que hoy aprenden será su mayor herramienta."

# INFORMACIÓN GENERAL DEL PROCESO DE ADMISIÓN
# CENTRO PREUNIVERSITARIO (CEPU - UNJBG)
El CEPU (Centro Preuniversitario) es un programa académico oficial de preparación que ofrece la UNJBG. Está diseñado para brindar a los postulantes una formación sólida y especializada que les permita afrontar con éxito el examen de admisión.
¿A quién está dirigido?
- A estudiantes de 3°, 4° y 5° de secundaria que desean iniciar su preparación con anticipación.
- A egresados de colegio que buscan reforzar sus conocimientos antes de postular.
¿Para qué sirve?
- Nivelar y reforzar los conocimientos en las áreas de matemática, comunicación, ciencias y humanidades.
- Familiarizar al estudiante con la estructura y el estilo del examen de admisión de la UNJBG.
- Aumentar significativamente las probabilidades de ingreso a la carrera deseada.

# EXAMEN DE ADMISIÓN ORDINARIO
El Examen de Admisión Ordinario es la prueba oficial y principal que la UNJBG aplica para seleccionar a sus nuevos ingresantes.
¿Para qué sirve?
- Evalúa los conocimientos, habilidades y aptitudes académicas de los postulantes.
- Es la vía estándar para cubrir la mayoría de las vacantes de todas las carreras profesionales.
¿Quiénes pueden postular?
- Egresados de Educación Básica Regular (EBR) y Educación Básica Alternativa (EBA) del Perú.
- Egresados con estudios en el extranjero que estén convalidados o sean equivalentes a la educación secundaria peruana.

# ESTRUCTURA DEL EXAMEN
El examen consta de 60 preguntas de selección múltiple.
- Duración: 2 horas (de 10:00 a.m. a 12:00 p.m.)
- Sistema de puntuación:
  - Pregunta bien contestada: 10 puntos.
  - Pregunta no contestada (en blanco): 1 punto.
  - Pregunta mal contestada: -0.25 puntos (resta un cuarto de punto).

# REQUISITOS PARA LA INSCRIPCIÓN
1. Documento Nacional de Identidad (DNI) vigente.
2. Certificado de Estudios Secundarios (original y oficial).
3. Declaración Jurada (firmada y escaneada).
4. Comprobante de pago por derecho de inscripción.

# BENEFICIOS Y EXONERACIONES
Simulacro de Examen de Admisión:
- La UNJBG organiza periódicamente simulacros oficiales (previos a los exámenes reales).
- El simulacro incluye un test vocacional gratuito (Test de Valoración de Perfil de Ingreso - VPI).
Exoneración de pago (Premio al Mérito):
- Los 5 primeros puestos de cada canal obtienen la EXONERACIÓN TOTAL del pago por derecho de examen en la siguiente fase.

# CANALES OFICIALES
- Página web del CEPU: https://cepuweb.unjbg.edu.pe/
- Página web de Admisión: https://admision.unjbg.edu.pe/carreras
- Contacto directo (WhatsApp): 913 549 200 / Teléfono: (052) 583000 Anexo 2352.
"""

# Crear archivo temporal para el RAG
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(contenido_conocimiento.strip())
    temp_file_path = f.name

# Cargar y procesar documentos
loader = TextLoader(temp_file_path, encoding='utf-8')
documentos = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documentos)
embeddings = FastEmbedEmbeddings()
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever_unjbg = vector_store.as_retriever(search_kwargs={"k": 8})
os.unlink(temp_file_path)

print("✅ Sistema RAG configurado exitosamente.")

# =====================================================
# 3. ESQUEMA PYDANTIC Y FEW-SHOT (VERSIÓN COMPLETA)
# =====================================================
class PerfilPostulante(BaseModel):
    nombre: Optional[str] = Field(default="Anónimo", description="Nombre extraído del postulante o 'Anónimo'/'N/A' si no se menciona.")
    carrera_interes: Optional[str] = Field(default="Por definir", description="Carrera de interés mencionada, 'Por definir' o 'No especificada'.")
    nivel_certeza: str = Field(description="Nivel de certeza de su decisión (ej. 'Seguro', 'Dudoso', 'Incierto', 'Muy seguro').")
    analisis_emocional: str = Field(description="Estado emocional detectado (ej. 'Estresado', 'Nervioso', 'Entusiasmado', 'Tranquilo', 'Ansioso').")

json_parser = JsonOutputParser(pydantic_object=PerfilPostulante)

# Ejemplos de entrenamiento Few-Shot (6 ejemplos)
ejemplos_postulantes = [
    # Caso 1: Indecisión y estrés
    {
        "input": "No sé qué estudiar... dudo entre seguir Comercial o irme a letras. Estoy muy estresado.",
        "output": '{"nombre": "Anónimo", "carrera_interes": "Por definir", "nivel_certeza": "Dudoso", "analisis_emocional": "Estresado"}'
    },
    # Caso 2: Decisión firme y nerviosismo
    {
        "input": "Hola, soy Sofía. Quiero Medicina, estoy segura pero algo nerviosa por el examen.",
        "output": '{"nombre": "Sofía", "carrera_interes": "Medicina", "nivel_certeza": "Seguro", "analisis_emocional": "Nervioso"}'
    },
    # Caso 3: Entusiasmo y seguridad total
    {
        "input": "¡Hola! Me llamo Carlos y estoy emocionado porque quiero estudiar Ingeniería Civil. Es mi sueño desde niño.",
        "output": '{"nombre": "Carlos", "carrera_interes": "Ingeniería Civil", "nivel_certeza": "Muy seguro", "analisis_emocional": "Entusiasmado"}'
    },
    # Caso 4: Ansiedad y temor a no ingresar
    {
        "input": "Buenas tardes, soy Valeria. Quiero entrar a Enfermería pero tengo mucha ansiedad, me da miedo no ingresar.",
        "output": '{"nombre": "Valeria", "carrera_interes": "Enfermería", "nivel_certeza": "Seguro", "analisis_emocional": "Ansioso"}'
    },
    # Caso 5: Postulante que no menciona carrera ni nombre
    {
        "input": "Hola, estoy confundido y no sé qué hacer. ¿Me puedes ayudar?",
        "output": '{"nombre": "Anónimo", "carrera_interes": "Por definir", "nivel_certeza": "Incierto", "analisis_emocional": "Confundido"}'
    },
    # Caso 6: Postulante tranquilo que ya tiene una carrera en mente
    {
        "input": "Soy Diego, estoy interesado en la carrera de Arquitectura. Estoy tranquilo y preparándome.",
        "output": '{"nombre": "Diego", "carrera_interes": "Arquitectura", "nivel_certeza": "Seguro", "analisis_emocional": "Tranquilo"}'
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
    ("system", """Eres un analizador de datos rígido de la oficina de admisión de la UNJBG. 
Tu única tarea es extraer información del mensaje de un estudiante y formatearlo estrictamente según las instrucciones. 
Siempre debes responder con un objeto JSON válido según el esquema especificado.
{format_instructions}"""),
    few_shot_prompt,
    ("human", "Analiza el siguiente mensaje de un postulante:\n{message}")
])

json_partial_template = json_template.partial(format_instructions=json_parser.get_format_instructions())
json_chain = json_partial_template | llm | json_parser

print("✅ Cadena JSON con Pydantic y Few-Shot lista.")

# =====================================================
# 4. ORQUESTADOR PARALELO (RAG + JSON)
# =====================================================
def procesar_consulta_estudiante(x):
    mensaje_usuario = x["message"]
    historial = x.get("chat_history", []).copy()
    
    # 1. Búsqueda semántica RAG
    docs_relevantes = retriever_unjbg.invoke(mensaje_usuario)
    contexto_rag = "\n\n".join([doc.page_content for doc in docs_relevantes])
    
    # 2. Prompt del sistema con el contexto RAG inyectado
    system_prompt = SystemMessage(content=f"""Eres un orientador vocacional empático, amable y experto de la Universidad Nacional Jorge Basadre Grohmann (UNJBG) en Tacna.
Tu objetivo es guiar a los postulantes, calmar sus nervios y resolver sus dudas sobre carreras y el examen de admisión.

INFORMACIÓN OFICIAL RECUPERADA DE LA BASE VECTORIAL (RAG):
{contexto_rag}

**INSTRUCCIONES OBLIGATORIAS:**
1. **Prioriza el RAG:** Usa la información del RAG para responder con precisión. Si el RAG tiene un dato exacto (como la duración de una carrera), respeta ese dato sin cambiarlo.
2. **Plan B (Contexto vacío):** Si el RAG no encontró información específica para la pregunta, PUEDES usar tu conocimiento general y experiencia como orientador vocacional para ayudarlo. Siempre aclara que estás dando una respuesta general y recomienda visitar la web oficial para más detalles.
3. **Redirigir temas externos:** Si el postulante te pregunta sobre temas que NO están relacionados con la UNJBG, las carreras, los procesos de admisión, el CEPU, los requisitos o la logística del examen, debes responder cortésmente de la siguiente manera:
*"Lo siento, soy un asistente especializado únicamente en la orientación vocacional y el proceso de admisión de la UNJBG. No tengo información sobre ese tema. ¿Puedo ayudarte con alguna duda sobre las carreras o el examen de admisión?"*
4. **Manejo de carreras inexistentes:** Si te preguntan por una carrera que no está en la lista de la UNJBG, responde: *"Esa carrera no forma parte de la oferta académica de la UNJBG. Para conocer todas las carreras que ofrecemos, visita nuestro portal oficial."*
""")
    
    # 3. Ensamblar mensajes con historial
    mensajes_totales = [system_prompt] + historial + [HumanMessage(content=mensaje_usuario)]
    
    # 4. Invocar al LLM
    respuesta_llm = llm.invoke(mensajes_totales)
    return respuesta_llm.content

preparar_json = RunnableLambda(lambda x: {"message": x["message"]})
rama_json = preparar_json | json_chain
rama_agente = RunnableLambda(procesar_consulta_estudiante)

orquestador_paralelo = RunnableParallel({
    "respuesta_estudiante": rama_agente,
    "datos_estructurados": rama_json
})

print("✅ Orquestador Paralelo LCEL ensamblado con éxito.")

# =====================================================
# 5. CLASE DEL AGENTE CON MEMORIA
# =====================================================
class AgenteConsejeroUNJBG:
    def __init__(self):
        self.historial = []  # Para LangChain (mensajes)
        self.chat_history = []  # Para Gradio (tuplas)
    
    def generar_respuesta(self, mensaje_usuario):
        if not mensaje_usuario or not mensaje_usuario.strip():
            return self.chat_history, {"status": "Entrada vacía"}, self.historial
        
        try:
            resultado = orquestador_paralelo.invoke({
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
            error_msg = f"⚠️ Error: {str(e)}"
            self.chat_history.append((mensaje_usuario, error_msg))
            return self.chat_history, {"status": "Error"}, self.historial
    
    def limpiar_memoria(self):
        self.historial = []
        self.chat_history = []
        return [], {"status": "🔄 Conversación reiniciada"}, []

bot = AgenteConsejeroUNJBG()

# =====================================================
# 6. INTERFAZ GRADIO CON ESTILOS UNJBG
# =====================================================

css = """
/* Variables de color UNJBG */
:root {
    --unjbg-bg-main: #0A2240;
    --unjbg-green: #005035;
    --unjbg-gold: #C9A84C;
    --unjbg-header-bg: #FFFFFF;
    --unjbg-text-dark: #1E293B;
    --unjbg-json-bg: #111827;
    --unjbg-border-active: #005035;
    --unjbg-bg-input: #0F172A;
}

/* Fondo general */
.gradio-container {
    background: linear-gradient(135deg, #0A2240 0%, #1a3a5c 100%) !important;
}

/* Encabezado */
.unjbg-header {
    background: linear-gradient(90deg, #FFFFFF 0%, #F8FAFC 100%) !important;
    border-bottom: 4px solid var(--unjbg-green) !important;
    padding: 1.2rem 2rem !important;
    border-radius: 12px 12px 0 0 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
}
.unjbg-header h1 {
    color: var(--unjbg-text-dark) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.5px !important;
}
.unjbg-header .subtitle {
    color: #64748B !important;
    font-size: 1rem !important;
    margin: 0 !important;
}
.unjbg-header .badge {
    background-color: var(--unjbg-green) !important;
    color: white !important;
    padding: 0.2rem 1rem !important;
    border-radius: 20px !important;
    font-size: 0.75rem !important;
    display: inline-block !important;
}

/* Contenedor principal */
.main-container {
    background-color: var(--unjbg-bg-main) !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1rem !important;
}

/* Botón enviar */
.btn-enviar {
    background: linear-gradient(135deg, #005035 0%, #003B26 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}
.btn-enviar:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 80, 53, 0.4) !important;
}

/* Panel JSON */
.json-container {
    background-color: var(--unjbg-json-bg) !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    border: 1px solid #334155 !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3) !important;
}
.json-container .json-header {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.5rem !important;
    border-bottom: 1px solid #1E293B !important;
    padding-bottom: 0.5rem !important;
}

/* Burbujas de chat */
.message {
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    margin: 0.25rem 0 !important;
    max-width: 85% !important;
}
.message.user {
    background: linear-gradient(135deg, #005035 0%, #007A4D 100%) !important;
    color: white !important;
    border-top-right-radius: 0 !important;
    margin-left: auto !important;
}
.message.assistant {
    background: linear-gradient(135deg, #1E293B 0%, #2D3B4F 100%) !important;
    color: #E2E8F0 !important;
    border-top-left-radius: 0 !important;
    margin-right: auto !important;
    border-left: 3px solid var(--unjbg-gold) !important;
}

/* Input de texto */
.input-area textarea {
    background-color: var(--unjbg-bg-input) !important;
    color: #CBD5E1 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.input-area textarea:focus {
    border-color: var(--unjbg-green) !important;
    box-shadow: 0 0 0 3px rgba(0, 80, 53, 0.2) !important;
}
.input-area textarea::placeholder {
    color: #64748B !important;
}

/* Botones secundarios */
.btn-secondary {
    background-color: #1E293B !important;
    color: #94A3B8 !important;
    border: 1px solid #334155 !important;
    transition: all 0.3s !important;
}
.btn-secondary:hover {
    background-color: #2D3B4F !important;
    color: white !important;
}

/* Título de sección */
.section-title {
    color: #94A3B8 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
"""

# Creación de la interfaz
with gr.Blocks(theme=gr.themes.Soft(), css=css, title="UNJBG - Consejero de Admisión") as demo:
    # Encabezado institucional
    with gr.Row(elem_classes="unjbg-header"):
        with gr.Column(scale=1, min_width=80):
            gr.HTML("""
            <div style="background-color:#005035; color:white; width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:28px; font-weight:bold;">
                U
            </div>
            """)
        with gr.Column(scale=5):
            gr.Markdown("""
            # Universidad Nacional Jorge Basadre Grohmann
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <span class="subtitle">🧑‍🏫 Agente Consejero de Admisión</span>
                <span class="badge">RAG + LangChain + Groq</span>
                <span class="badge" style="background-color:#C9A84C !important;">24/7</span>
            </div>
            """)
    
    with gr.Row(elem_classes="main-container"):
        with gr.Column(scale=3):
            # Componente de chat
            chatbot = gr.Chatbot(
                label="💬 Conversación",
                height=500,
                bubble_full_width=False,
                avatar_images=(None, "🎓"),
                show_copy_button=True,
                render_markdown=True,
                elem_classes="message",
            )
            
            with gr.Row(elem_classes="input-area"):
                msg = gr.Textbox(
                    label="Escribe tu consulta",
                    placeholder="Ej: Hola, me llamo Juan. Estoy nervioso por el examen de admisión...",
                    scale=4,
                    lines=2,
                    container=False,
                )
                send_btn = gr.Button("📤 Enviar", variant="primary", elem_classes="btn-enviar", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🔄 Reiniciar", variant="secondary", elem_classes="btn-secondary", size="sm")
                info_btn = gr.Button("ℹ️ Ayuda", variant="secondary", elem_classes="btn-secondary", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Perfil del Postulante")
            gr.Markdown("""<div style="color:#94A3B8; font-size:0.8rem; margin-bottom:0.5rem;">Datos extraídos automáticamente en JSON</div>""")
            json_box = gr.JSON(
                label="",
                value={"status": "⏳ Esperando tu primera consulta..."},
                elem_classes="json-container",
            )
            gr.Markdown("""
            <div style="color:#64748B; font-size:0.75rem; margin-top:0.75rem; padding:0.5rem; background:#0F172A; border-radius:8px; border:1px solid #1E293B;">
                🔒 Los datos se procesan en tiempo real con IA
            </div>
            """)

    # =============================================
    # FUNCIONES DE RESPUESTA
    # =============================================
    def respond(message, state):
        # state es el historial de LangChain, lo pasamos al bot
        return bot.generar_respuesta(message)
    
    def clear():
        return bot.limpiar_memoria()
    
    def show_help():
        return """
        **📋 ¿Cómo usar el consejero?**
        
        1. **Preséntate**: Di tu nombre y cómo te sientes.
        2. **Pregunta sobre carreras**: Ej. "¿Cuánto dura Ingeniería de Sistemas?"
        3. **Consulta sobre el examen**: Ej. "¿Cómo es el examen de admisión?"
        4. **Expresa dudas**: Ej. "No sé qué carrera elegir"
        5. **Pregunta por requisitos**: Ej. "¿Qué necesito para inscribirme?"
        
        El sistema extraerá automáticamente tus datos en formato JSON.
        """
    
    # =============================================
    # EVENTOS
    # =============================================
    def respond_and_clear(message, state):
        chat, json_data, new_state = bot.generar_respuesta(message)
        return chat, json_data, new_state, ""
    
    send_btn.click(
        fn=respond_and_clear,
        inputs=[msg, gr.State(bot.historial)],
        outputs=[chatbot, json_box, gr.State(bot.historial), msg]
    )
    
    msg.submit(
        fn=respond_and_clear,
        inputs=[msg, gr.State(bot.historial)],
        outputs=[chatbot, json_box, gr.State(bot.historial), msg]
    )
    
    clear_btn.click(
        fn=clear,
        inputs=None,
        outputs=[chatbot, json_box, gr.State(bot.historial)]
    )
    
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
    print(f"🚀 Agente UNJBG ejecutándose en el puerto {port}")