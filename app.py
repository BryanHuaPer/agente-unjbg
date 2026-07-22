import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import gradio as gr
import tempfile

# =====================================================
# 1. CONFIGURACION DE GROQ
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY no configurada. Configurala en Render como variable de entorno.")

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
Duracion: 6 años (12 semestres).
Grado: Bachiller en Medicina. Titulo: Medico Cirujano.
Campo laboral: Hospitales, clinicas, consultorios, salud publica, docencia e investigacion.
Perfil del postulante: Vocacion de servicio, capacidad de analisis, sensibilidad social, manejo de estres.
Recomendacion de egresado: "La medicina es una vocacion de servicio. Cada paciente es una oportunidad para aprender y crecer."

# ENFERMERIA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Enfermeria. Titulo: Licenciado en Enfermeria.
Campo laboral: Hospitales, clinicas, centros de salud, docencia, gestion en salud.
Perfil del postulante: Vocacion de servicio, empatia, responsabilidad, capacidad de trabajo en equipo.
Recomendacion de egresado: "La enfermeria es el arte de cuidar. Cada dia es una oportunidad para marcar la diferencia."

# ODONTOLOGIA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Odontologia. Titulo: Cirujano Dentista.
Campo laboral: Consultorios, clinicas, salud publica, docencia, investigacion.
Perfil del postulante: Destreza manual, precision, vocacion de servicio, capacidad de analisis.
Recomendacion de egresado: "La odontologia combina ciencia y arte. La sonrisa de un paciente es la mejor recompensa."

# FARMACIA Y BIOQUIMICA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Farmacia y Bioquimica. Titulo: Quimico Farmaceutico.
Campo laboral: Farmacias, laboratorios, industria farmaceutica, docencia, investigacion.
Perfil del postulante: Capacidad de analisis, precision, responsabilidad, interes por la quimica y la biologia.
Recomendacion de egresado: "La farmacia es la ciencia de los medicamentos. Cada formula es una oportunidad para salvar vidas."

# CIENCIAS DE LA NUTRICION
Duracion: 5 años (10 semestres).
Grado: Bachiller en Nutricion. Titulo: Licenciado en Nutricion.
Campo laboral: Hospitales, clinicas, consultorios, industria alimentaria, docencia, investigacion.
Perfil del postulante: Interes por la alimentacion y la salud, capacidad de analisis, vocacion de servicio.
Recomendacion de egresado: "La nutricion es la base de la salud. Cada paciente es una oportunidad para educar y transformar."

# CANAL 2: INGENIERIA
# INGENIERIA DE SISTEMAS
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria de Sistemas. Titulo: Ingeniero de Sistemas.
Campo laboral: Desarrollo de software, consultoria tecnologica, gestion de TI, docencia, investigacion.
Perfil del postulante: Razonamiento logico, capacidad de abstraccion, interes por la tecnologia, trabajo en equipo.
Recomendacion de egresado: "La tecnologia es el futuro. Cada linea de codigo es una oportunidad para innovar."

# INGENIERIA CIVIL
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria Civil. Titulo: Ingeniero Civil.
Campo laboral: Construccion, diseño de infraestructura, consultoria, docencia, investigacion.
Perfil del postulante: Razonamiento espacial, capacidad de analisis, interes por la construccion, responsabilidad.
Recomendacion de egresado: "La ingenieria civil construye el futuro. Cada obra es un legado para las generaciones."

# INGENIERIA DE MINAS
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria de Minas. Titulo: Ingeniero de Minas.
Campo laboral: Mineria, exploracion, consultoria, docencia, investigacion.
Perfil del postulante: Razonamiento logico, capacidad de analisis, interes por la geologia, responsabilidad.
Recomendacion de egresado: "La mineria es la columna vertebral del desarrollo. Cada proyecto es una oportunidad para crecer."

# INGENIERIA INDUSTRIAL
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria Industrial. Titulo: Ingeniero Industrial.
Campo laboral: Industria, consultoria, gestion de operaciones, docencia, investigacion.
Perfil del postulante: Razonamiento logico, capacidad de analisis, liderazgo, interes por la optimizacion.
Recomendacion de egresado: "La ingenieria industrial optimiza procesos. Cada mejora es una oportunidad para ser mas eficientes."

# INGENIERIA MECANICA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria Mecanica. Titulo: Ingeniero Mecanico.
Campo laboral: Industria, diseño de maquinas, consultoria, docencia, investigacion.
Perfil del postulante: Razonamiento espacial, capacidad de analisis, interes por la mecanica, trabajo en equipo.
Recomendacion de egresado: "La ingenieria mecanica mueve el mundo. Cada maquina es una oportunidad para innovar."

# INGENIERIA AGROINDUSTRIAL
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria Agroindustrial. Titulo: Ingeniero Agroindustrial.
Campo laboral: Agroindustria, consultoria, docencia, investigacion.
Perfil del postulante: Interes por la agricultura y la industria, capacidad de analisis, trabajo en equipo.
Recomendacion de egresado: "La agroindustria transforma el campo. Cada producto es una oportunidad para crecer."

# CANAL 3: CIENCIAS SOCIALES Y EDUCACION
# EDUCACION: IDIOMA EXTRANJERO
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la Educacion. Titulo: Licenciado en educacion, especialidad en Idioma Extranjero.
Campo laboral: Traduccion de documentos comerciales. Docencia en EBR y Educacion Superior. Formulacion de programas de enseñanza del ingles. Mediacion intercultural.
Perfil del postulante: Conocimiento pre-intermedio de Ingles. Competencias en comprension y produccion de textos. Tolerancia a la diversidad laboral y cultural.
Recomendacion de egresado: "Mi formacion me ha permitido adquirir herramientas pedagogicas efectivas, preparandome para inspirar y guiar en el crecimiento academico y personal de mis estudiantes."

# EDUCACION: CIENCIAS DE LA NATURALEZA Y PROMOCION EDUCATIVA AMBIENTAL
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la Educacion. Titulo: Licenciado en Educacion, Especialidad en Ciencias de la Naturaleza y Promocion Educativa Ambiental.
Campo laboral: Docencia en secundaria publica y privada. Docencia universitaria y en institutos superiores. Gestion y direccion de instituciones educativas.
Perfil del postulante: Vocacion docente. Estilos de vida saludable. Iniciativa para indagar el mundo natural con conocimientos cientificos. Conocimientos basicos de TICs.
Recomendacion de egresado: "Estoy convencido que ser docente de las ciencias de la naturaleza y promocion educativa ambiental es la carrera mas hermosa que puedes experimentar."

# ARTES
Duracion: 5 años (10 semestres).
Grado: Bachiller en Artes Plasticas. Titulo: Licenciado en Artes Plasticas.
Campo laboral: Talleres artisticos publicos y privados. Centros de produccion artistica. Instituciones de investigacion y difusion artistica. Gestion, promocion cultural y curaduria. Docencia.
Perfil del postulante: Conocimiento de la historia del arte. Habilidad en TICs. Imaginacion creativa y sensibilidad estetica. Habilidad para gestionar proyectos culturales.
Recomendacion de egresado: "Emprender la carrera de Artes en la UNJBG es un acto desafiante. Si te apasiona y te entregas, encontraras tu identidad artistica."

# CIENCIAS DE LA COMUNICACION
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la Comunicacion Social. Titulo: Licenciado en Comunicacion Social en Periodismo y Relaciones Publicas.
Campo laboral: Medios de comunicacion (radio, prensa, TV, internet). Relaciones publicas en instituciones estatales y privadas. Empresas de publicidad y marketing. Consultorias empresariales.
Perfil del postulante: Alto desempeño en expresion oral, escrita y audiovisual. Sensibilidad social y cultural. Interes por la lectura, el cine y la multimedia. Actitud investigativa.
Recomendacion de egresado: "Comunicar es imaginar futuros posibles, transformar realidades y narrar con sentido. Los invito a sostener sus convicciones y no dejar de soñar."

# ARQUITECTURA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Arquitectura. Titulo: Arquitecto.
Campo laboral: Proyectista de obras nuevas y restauracion. Constructor y ejecutor de obras. Asesor en gestion de proyectos inmobiliarios. Planificacion urbana. Docencia e investigacion.
Perfil del postulante: Creativo, innovador y lider. Conocimiento de cultura y realidad social. Sensibilidad artistica. Etica y moral profesional.
Recomendacion de egresado: "Lograr el primer puesto de mi promocion me permitio obtener una beca y titularme en el Magister de Administracion de la Construccion en Chile."

# DERECHO Y CIENCIAS POLITICAS
Duracion: 6 años (12 semestres).
Grado: Bachiller en Derecho y Ciencias Politicas. Titulo: Abogado.
Campo laboral: Poder Judicial y Ministerio Publico. Ejercicio libre de la abogacia. Asesoria en el sector publico y privado. Notaria y Registro. Asesoria politica.
Perfil del postulante: Precision para la lectura. Habilidades de analisis. Capacidad de comunicacion escrita y oral. Sentido de justicia, responsabilidad y lealtad. Inclinacion a la investigacion.
Recomendacion de egresado: "Confien en sus capacidades y atrevansen a nuevos retos. No olviden que la competencia es con uno mismo, para ser mejor cada dia."

# EDUCACION INICIAL
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la educacion. Titulo: Licenciado en Educacion Inicial.
Campo laboral: Guarderias, nidos, centros de educacion inicial. Programas gubernamentales (Cuna Mas, PRONOEI, Wawa Wasi). Asesoramiento en estimulacion temprana e inclusion educativa.
Perfil del postulante: Respeto y tolerancia. Aprecio por el arte para expresar el mundo personal. Gestion de proyectos de emprendimiento. Indagacion del mundo fisico. Destreza manual.
Recomendacion de egresado: "La Direccion de Admision invita a los estudiantes a participar en los procesos. ¡Sean basadrinos de corazon!"

# EDUCACION PRIMARIA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias de la educacion. Titulo: Licenciado en Educacion Primaria.
Campo laboral: Colegios publicos y privados. Diseño curricular y pedagogia. Capacitacion y formacion docente. Gestion educativa y direccion de instituciones.
Perfil del postulante: Habilidades comunicativas, matematicas y graficas. Capacidad de observacion y analisis. Trabajo autonomo y en equipo. Aplicacion de la ciencia y tecnologia. Destreza manual.
Recomendacion de egresado: "La Direccion de Admision invita a los estudiantes a participar en los procesos. ¡Sean basadrinos de corazon!"

# PSICOLOGIA
Duracion: 6 años (12 semestres).
Grado: Bachiller en Psicologia. Titulo: Licenciado en Psicologia.
Campo laboral: Psicologia social (ONGs, ministerios). Educacion (colegios, universidades, orientacion vocacional). Recursos humanos (empresas e instituciones). Psicologia clinica (evaluacion e intervencion).
Perfil del postulante: Capacidad para la interaccion social. Tolerancia. Conocimientos de cultura general. Comprension de la conducta humana. Conocimiento informatico. Capacidad analitica.
Recomendacion de egresado: "Estudiar Psicologia es iniciar un viaje hacia el corazon del ser humano. Es acompanar al otro en sus luchas internas y contribuir al bienestar colectivo."

# CANAL 4: CIENCIAS ACTUARIALES Y EMPRESARIALES
# CIENCIAS ADMINISTRATIVAS
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias Administrativas. Titulo: Licenciado en Administracion.
Campo laboral: Organizaciones nacionales e internacionales (publico o privado). Direccion y gestion estrategica, Finanzas, Marketing, Recursos Humanos, Produccion y Logistica. Emprendimientos innovadores.
Perfil del postulante: Comprension lectora. Equilibrio emocional, respeto y disciplina. Capacidad de liderazgo. Conocimiento de economia y ofimatica (Excel, Word).
Recomendacion de egresado: "Lo aprendido en la carrera de Administracion me ha permitido alcanzar mis metas personales y colaborar en el desarrollo del pais."

# CIENCIAS CONTABLES Y FINANCIERAS
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias Contables y Financieras. Titulo: Contador Publico.
Campo laboral: Asesoria contable a personas y empresas. Analisis de estados financieros. Auditoria de sistemas publicos y privados (NIA). Gestion de procesos contables. Evaluacion de informacion economica y financiera.
Perfil del postulante: Comprension lectora. Razonamiento matematico y logico. Capacidad de liderazgo y trabajo en equipo. Conocimiento de economia y ofimatica (Excel, Word).
Recomendacion de egresado: "En las aulas de la UNJBG pude afianzar mi vocacion. Me forjaron con la pasion de que, siendo contador, podemos cambiar al mundo."

# INGENIERIA COMERCIAL
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ingenieria Comercial. Titulo: Ingeniero Comercial.
Campo laboral: Gestion, marketing, economia, finanzas, administracion. Asesor en proyectos de inversion. Consultor en diversificacion empresarial. Planeamiento estrategico. Impulso de PYMES.
Perfil del postulante: Aplicacion de TICs. Comunicacion verbal y no verbal. Conocimientos de gestion empresarial y negocios. Habilidades blandas para la formacion integral.
Recomendacion de egresado: "Recomiendo a todos los estudiantes aprovechar las oportunidades que ofrece la universidad y siempre dejar en alto a Tacna."

# ECONOMIA AGRARIA
Duracion: 5 años (10 semestres).
Grado: Bachiller en Ciencias con mencion en Economia Agraria. Titulo: Ingeniero en Economia Agraria.
Campo laboral: Liderazgo en agronegocios y comercio exterior. Funciones administrativas en empresas. Entidades del Estado (planificacion, presupuesto, logistica). Analisis financiero en entidades bancarias.
Perfil del postulante: Comprension de problemas socioeconomicos. Comunicacion de ideas coherentes. Habilidades blandas. Compromiso con el estudio.
Recomendacion de egresado: "Gracias a la formacion recibida, he podido contribuir en espacios tecnicos del estado vinculados a la gestion de la inversion publica. Lo que hoy aprenden sera su mayor herramienta."

# INFORMACION GENERAL DEL PROCESO DE ADMISION
# CENTRO PREUNIVERSITARIO (CEPU - UNJBG)
El CEPU (Centro Preuniversitario) es un programa academico oficial de preparacion que ofrece la UNJBG. Esta diseñado para brindar a los postulantes una formacion solida y especializada que les permita afrontar con exito el examen de admision.
A quien esta dirigido:
- A estudiantes de 3°, 4° y 5° de secundaria que desean iniciar su preparacion con anticipacion.
- A egresados de colegio que buscan reforzar sus conocimientos antes de postular.
Para que sirve:
- Nivelar y reforzar los conocimientos en las areas de matematica, comunicacion, ciencias y humanidades.
- Familiarizar al estudiante con la estructura y el estilo del examen de admision de la UNJBG.
- Aumentar significativamente las probabilidades de ingreso a la carrera deseada.

# EXAMEN DE ADMISION ORDINARIO
El Examen de Admision Ordinario es la prueba oficial y principal que la UNJBG aplica para seleccionar a sus nuevos ingresantes.
Para que sirve:
- Evalua los conocimientos, habilidades y aptitudes academicas de los postulantes.
- Es la via estandar para cubrir la mayoria de las vacantes de todas las carreras profesionales.
Quienes pueden postular:
- Egresados de Educacion Basica Regular (EBR) y Educacion Basica Alternativa (EBA) del Peru.
- Egresados con estudios en el extranjero que esten convalidados o sean equivalentes a la educacion secundaria peruana.

# ESTRUCTURA DEL EXAMEN
El examen consta de 60 preguntas de seleccion multiple.
- Duracion: 2 horas (de 10:00 a.m. a 12:00 p.m.)
- Sistema de puntuacion:
  - Pregunta bien contestada: 10 puntos.
  - Pregunta no contestada (en blanco): 1 punto.
  - Pregunta mal contestada: -0.25 puntos (resta un cuarto de punto).

# REQUISITOS PARA LA INSCRIPCION
1. Documento Nacional de Identidad (DNI) vigente.
2. Certificado de Estudios Secundarios (original y oficial).
3. Declaracion Jurada (firmada y escaneada).
4. Comprobante de pago por derecho de inscripcion.

# BENEFICIOS Y EXONERACIONES
Simulacro de Examen de Admision:
- La UNJBG organiza periodicamente simulacros oficiales (previos a los examenes reales).
- El simulacro incluye un test vocacional gratuito (Test de Valoracion de Perfil de Ingreso - VPI).
Exoneracion de pago (Premio al Merito):
- Los 5 primeros puestos de cada canal obtienen la EXONERACION TOTAL del pago por derecho de examen en la siguiente fase.

# CANALES OFICIALES
- Pagina web del CEPU: https://cepuweb.unjbg.edu.pe/
- Pagina web de Admision: https://admision.unjbg.edu.pe/carreras
- Contacto directo (WhatsApp): 913 549 200 / Telefono: (052) 583000 Anexo 2352.
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

print("Sistema RAG configurado exitosamente.")

# =====================================================
# 3. ORQUESTADOR PRINCIPAL (SOLO CONVERSACION)
# =====================================================
def procesar_consulta_estudiante(x):
    mensaje_usuario = x["message"]
    historial = x.get("chat_history", []).copy()
    
    # 1. Busqueda semantica RAG
    docs_relevantes = retriever_unjbg.invoke(mensaje_usuario)
    contexto_rag = "\n\n".join([doc.page_content for doc in docs_relevantes])
    
    # 2. Prompt del sistema con el contexto RAG inyectado
    system_prompt = SystemMessage(content=f"""Eres un orientador vocacional empatico, amable y experto de la Universidad Nacional Jorge Basadre Grohmann (UNJBG) en Tacna.
Tu objetivo es guiar a los postulantes, calmar sus nervios y resolver sus dudas sobre carreras y el examen de admision.

INFORMACION OFICIAL RECUPERADA DE LA BASE VECTORIAL (RAG):
{contexto_rag}

**INSTRUCCIONES OBLIGATORIAS:**
1. **Prioriza el RAG:** Usa la informacion del RAG para responder con precision. Si el RAG tiene un dato exacto (como la duracion de una carrera), respeta ese dato sin cambiarlo.
2. **Plan B (Contexto vacio):** Si el RAG no encontro informacion especifica para la pregunta, PUEDES usar tu conocimiento general y experiencia como orientador vocacional para ayudarlo. Siempre aclara que estas dando una respuesta general y recomienda visitar la web oficial para mas detalles.
3. **Redirigir temas externos:** Si el postulante te pregunta sobre temas que NO estan relacionados con la UNJBG, las carreras, los procesos de admision, el CEPU, los requisitos o la logistica del examen, debes responder cortesmente de la siguiente manera:
"Lo siento, soy un asistente especializado unicamente en la orientacion vocacional y el proceso de admision de la UNJBG. No tengo informacion sobre ese tema. ¿Puedo ayudarte con alguna duda sobre las carreras o el examen de admision?"
4. **Manejo de carreras inexistentes:** Si te preguntan por una carrera que no esta en la lista de la UNJBG, responde: "Esa carrera no forma parte de la oferta academica de la UNJBG. Para conocer todas las carreras que ofrecemos, visita nuestro portal oficial."
""")
    
    # 3. Ensamblar mensajes con historial
    mensajes_totales = [system_prompt] + historial + [HumanMessage(content=mensaje_usuario)]
    
    # 4. Invocar al LLM
    respuesta_llm = llm.invoke(mensajes_totales)
    return respuesta_llm.content

rama_agente = RunnableLambda(procesar_consulta_estudiante)

# =====================================================
# 4. CLASE DEL AGENTE CON MEMORIA
# =====================================================
class AgenteConsejeroUNJBG:
    def __init__(self):
        self.historial = []  # Para LangChain (mensajes)
        self.chat_history = []  # Para Gradio (tuplas)
    
    def generar_respuesta(self, mensaje_usuario):
        if not mensaje_usuario or not mensaje_usuario.strip():
            return self.chat_history, self.historial
        
        try:
            resultado = rama_agente.invoke({
                "message": mensaje_usuario,
                "chat_history": self.historial
            })
            
            respuesta_agente = resultado
            
            # Actualizar ambos historiales
            self.chat_history.append((mensaje_usuario, respuesta_agente))
            self.historial.append(HumanMessage(content=mensaje_usuario))
            self.historial.append(AIMessage(content=respuesta_agente))
            
            return self.chat_history, self.historial
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.chat_history.append((mensaje_usuario, error_msg))
            return self.chat_history, self.historial
    
    def limpiar_memoria(self):
        self.historial = []
        self.chat_history = []
        return [], []

bot = AgenteConsejeroUNJBG()

# =====================================================
# 5. INTERFAZ GRADIO CON ESTILOS UNJBG
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

/* Logo */
.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #005035;
    border-radius: 50%;
    width: 70px;
    height: 70px;
    flex-shrink: 0;
}
.logo-container img {
    width: 50px;
    height: 50px;
    object-fit: contain;
    filter: brightness(0) invert(1);
}

/* Contenedor principal */
.main-container {
    background-color: var(--unjbg-bg-main) !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1rem !important;
}

/* Boton enviar */
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

/* Estilos para el logo UNJBG en el chat */
.chat-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #005035;
    color: white;
    font-weight: bold;
    font-size: 16px;
}
"""

# Creacion de la interfaz
with gr.Blocks(theme=gr.themes.Soft(), css=css, title="UNJBG - Consejero de Admision") as demo:
    # Encabezado institucional
    with gr.Row(elem_classes="unjbg-header"):
        with gr.Column(scale=1, min_width=80):
            # Logo de la UNJBG (usando un div estilizado)
            gr.HTML("""
            <div class="logo-container">
                <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <rect width="100" height="100" rx="10" fill="#005035"/>
                    <text x="50" y="68" font-size="50" text-anchor="middle" fill="white" font-weight="bold" font-family="Arial">U</text>
                </svg>
            </div>
            """)
        with gr.Column(scale=5):
            gr.Markdown("""
            # Universidad Nacional Jorge Basadre Grohmann
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <span class="subtitle">Agente Consejero de Admision</span>
                <span class="badge">RAG + LangChain + Groq</span>
                <span class="badge" style="background-color:#C9A84C !important;">24/7</span>
            </div>
            """)
    
    with gr.Row(elem_classes="main-container"):
        with gr.Column(scale=1):
            # Componente de chat
            chatbot = gr.Chatbot(
                label="Conversacion",
                height=550,
                bubble_full_width=False,
                avatar_images=(None, "logo_unjbg.png"),
                show_copy_button=True,
                render_markdown=True,
                elem_classes="message",
            )
            
            with gr.Row(elem_classes="input-area"):
                msg = gr.Textbox(
                    label="Escribe tu consulta",
                    placeholder="Ej: Hola, me llamo Juan. Estoy nervioso por el examen de admision...",
                    scale=4,
                    lines=2,
                    container=False,
                )
                send_btn = gr.Button("Enviar", variant="primary", elem_classes="btn-enviar", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Reiniciar", variant="secondary", elem_classes="btn-secondary", size="sm")
                info_btn = gr.Button("Ayuda", variant="secondary", elem_classes="btn-secondary", size="sm")

    # =============================================
    # FUNCIONES DE RESPUESTA
    # =============================================
    def respond_and_clear(message, state):
        chat, new_state = bot.generar_respuesta(message)
        return chat, new_state, ""
    
    def clear():
        return bot.limpiar_memoria()
    
    def show_help():
        return """
        Como usar el consejero:
        
        1. Presentate: Di tu nombre y como te sientes.
        2. Pregunta sobre carreras: Ej. "Cuanto dura Ingenieria de Sistemas?"
        3. Consulta sobre el examen: Ej. "Como es el examen de admision?"
        4. Expresa dudas: Ej. "No se que carrera elegir"
        5. Pregunta por requisitos: Ej. "Que necesito para inscribirme?"
        """
    
    # =============================================
    # EVENTOS
    # =============================================
    send_btn.click(
        fn=respond_and_clear,
        inputs=[msg, gr.State(bot.historial)],
        outputs=[chatbot, gr.State(bot.historial), msg]
    )
    
    msg.submit(
        fn=respond_and_clear,
        inputs=[msg, gr.State(bot.historial)],
        outputs=[chatbot, gr.State(bot.historial), msg]
    )
    
    clear_btn.click(
        fn=clear,
        inputs=None,
        outputs=[chatbot, gr.State(bot.historial)]
    )
    
    info_btn.click(
        fn=show_help,
        inputs=None,
        outputs=[msg]
    )

# =====================================================
# 6. LANZAMIENTO PARA RENDER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
    print(f"Agente UNJBG ejecutandose en el puerto {port}")