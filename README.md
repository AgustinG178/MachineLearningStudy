
# 🧠 Chatbot Médico Explicativo para Interpretación de Análisis Clínicos

Este proyecto forma parte de mi desarrollo académico como estudiante de **Bioingeniería en la UNER**, y busca aplicar conocimientos de **inteligencia artificial, machine learning, programación y sistemas de bases de datos** al ámbito de la salud, con un fuerte compromiso por la **accesibilidad, la digitalización abierta** y la empatía comunicacional.

## 📌 Objetivo general

Diseñar e implementar un **chatbot médico explicativo** que sea capaz de interpretar resultados clínicos de forma automatizada y comunicarlos en lenguaje claro, no técnico, a personas sin formación médica. El sistema también funcionará como una **plataforma de expediente clínico unificado**, donde se almacenen los datos de cada paciente de forma estructurada y segura.

## 🚀 Motivación

Hoy en día, muchos pacientes reciben informes que no comprenden. La medicina está fragmentada en silos digitales que dificultan el acceso a la información propia. Este proyecto busca **ayudar a los médicos, no reemplazarlos**, construyendo una interfaz de comunicación puente entre el conocimiento clínico y el entendimiento cotidiano.

## 🧩 Componentes principales

- **Chatbot explicativo con procesamiento de lenguaje natural (NLP)** para traducir valores de análisis a explicaciones claras.
- **Base de datos estructurada** para almacenar historiales médicos individuales.
- **Modelo de análisis automático** con detección de alertas y patrones relevantes.
- **Frontend accesible** para que tanto pacientes como médicos accedan a los resultados.

## 🛠️ Tecnologías propuestas

- Python, FastAPI o Flask (API)
- HuggingFace Transformers, scikit-learn, spaCy (NLP/ML)
- PostgreSQL o SQLite (DB)
- Streamlit o React (Frontend)
- Docker, JWT, cifrado por campo (seguridad y despliegue)

## 🔐 Enfoque ético y legal

Todo el diseño considera el respeto por la privacidad de los datos médicos, la autonomía del paciente y el cumplimiento de normativas vigentes en Argentina (como la Ley 26.5

## 📂Orden de carpetas

chatbot-medico/
│
├── app/                        ← Backend: FastAPI/Flask
│   ├── main.py                 ← Punto de entrada API
│   ├── models.py               ← Modelos de SQLAlchemy / Pydantic
│   ├── database.py             ← Conexión y esquema DB
│   ├── routes/                 ← Endpoints REST
│   ├── services/               ← Lógica de negocio (e.g. intérprete clínico)
│   └── utils/                  ← Funciones auxiliares
│
├── chatbot/                    ← Módulo NLP
│   ├── interpreter.py          ← Reglas y explicaciones médicas
│   ├── prompts/                ← Prompts en lenguaje natural
│   └── models/                 ← Modelos fine-tuned o descargados
│
├── data/                       ← Datos anonimizados / CSV de pruebas
│   ├── ejemplos_pacientes.csv
│   ├── normal_ranges.json      ← Valores normales por estudio
│   └── glosario_clinico.json   ← Diccionario técnico → coloquial
│
├── frontend/                   ← UI para pacientes y médicos
│   ├── streamlit_app.py        ← Chat visual y carga de datos
│   └── components/             ← Gráficos, input, dashboards
│
├── tests/                      ← Testeo unitario y de integración
│   ├── test_api.py
│   └── test_chatbot.py
│
├── notebooks/                  ← Experimentación de ML o NLP
│   ├── analisis_exploratorio.ipynb
│   └── clustering_pacientes.ipynb
│
├── requirements.txt            ← Dependencias del proyecto
├── README.md                   ← Documentación general
└── .env                        ← Variables sensibles (.gitignored)

## 👨‍💻 Sobre mí

Me llamo **Agustín**, soy estudiante de Bioingeniería y un apasionado de la **física, la matemática y la programación**, con particular interés en el campo de la **IA aplicada a la salud**. Este proyecto representa una convergencia natural de mis intereses y una propuesta concreta para aportar valor en un área donde la tecnología aún tiene mucho por mejorar.

---

> 📣 *Este proyecto está en etapa de diseño y prototipado inicial. Toda colaboración, feedback técnico o médico, será más que bienvenida.*
>
