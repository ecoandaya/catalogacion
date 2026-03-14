# OpenAI File Processor

Script para **Linux / consola** que permite procesar automáticamente todos los archivos de una carpeta usando la API de OpenAI.

El programa:

- recorre todos los archivos de una carpeta
- los sube **uno por uno** a la API
- ejecuta una **consulta automática para cada archivo**
- guarda la respuesta en un **log estructurado**
- registra errores sin detener el proceso

Está pensado para **procesar grandes lotes de documentos**.

---

# Características

✔ Procesamiento **archivo por archivo**  
✔ Compatible con **carpetas grandes**  
✔ Registro de resultados en **JSONL**  
✔ Registro de **errores con traceback**  
✔ Procesamiento **recursivo opcional**  
✔ Eliminación automática de archivos subidos  
✔ Control de pausa entre peticiones  
✔ Prompt configurable  

# Requisitos

- Linux / MacOS
- Python **3.9 o superior**
- Cuenta de OpenAI
- API Key de OpenAI
# Instalación
Clonar el repositorio:
git clone [https://github.com/TUUSUARIO/openai-file-processor.git](https://github.com/TUUSUARIO/openai-file-processor.git)
cd openai-file-processor
Crear entorno virtual:
python3 -m venv .venv
Activarlo:
source .venv/bin/activate
Instalar dependencias:
pip install openai
# Configuración
Definir la API key:
export OPENAI_API_KEY="tu_api_key"
Para hacerlo permanente:
echo 'export OPENAI_API_KEY="tu_api_key"' >> ~/.bashrc
source ~/.bashrc
# Uso básico
Procesar una carpeta:
python procesar_carpeta_openai.py /ruta/a/los/archivos
# Opciones disponibles

| Opción | Descripción |
|------|-------------|
| `--model` | Modelo de OpenAI a utilizar |
| `--prompt` | Prompt que se ejecutará para cada archivo |
| `--instructions` | Instrucciones adicionales |
| `--log` | Archivo de log |
| `--recursive` | Procesar subdirectorios |
| `--delete-remote` | Eliminar archivos subidos |
| `--sleep` | Pausa entre archivos |

---

# Ejemplo de uso avanzado

```

python procesar_carpeta_openai.py documentos 
--recursive 
--model gpt-4.1 
--prompt "Lee el archivo '{filename}' y genera:

1. un resumen
2. 10 etiquetas temáticas
3. una valoración de utilidad." 
   --instructions "Responde siempre en español." 
   --log resultados.jsonl 
   --delete-remote 
   --sleep 1

```

---

# Formato del log

Los resultados se guardan en **JSONL**  
(una línea JSON por archivo).

Ejemplo de resultado correcto:

```

{
"status": "ok",
"timestamp": "2026-03-14 11:20:33",
"file_name": "informe.pdf",
"file_path": "/home/user/docs/informe.pdf",
"file_size_bytes": 183920,
"remote_file_id": "file_abc123",
"model": "gpt-4.1",
"response_text": "Resumen del documento...",
"elapsed_seconds": 4.81
}

```

Ejemplo de error:

```

{
"status": "error",
"file_name": "archivo.pdf",
"error_type": "Exception",
"error": "mensaje de error",
"traceback": "detalle del error"
}

```

---

# Ejemplos de prompts útiles

### Resumen automático

```

Analiza el archivo '{filename}' y produce un resumen claro en español.

```

### Extracción de etiquetas

```

Extrae 15 etiquetas temáticas representativas del contenido del archivo '{filename}'.

```

### Catalogación de documentos

```

Analiza el documento '{filename}' y devuelve:

* tema principal
* subtemas
* tipo de documento
* resumen

```

---

# Estructura del proyecto

```

openai-file-processor
│
├─ procesar_carpeta_openai.py
├─ README.md
└─ logs/

```

---

# Límites de la API

Según la documentación de OpenAI:

- tamaño máximo por archivo: **512 MB**
- almacenamiento total por proyecto: **2.5 TB**

---

# Buenas prácticas

Para procesar grandes cantidades de archivos:

- usar `--sleep` para evitar límites de tasa
- usar `--delete-remote` para no acumular archivos
- guardar logs periódicamente

---

# Posibles mejoras

- exportar resultados a CSV
- procesamiento paralelo
- barra de progreso
- reintentos automáticos
- análisis de imágenes
- indexación de documentos

---

# Licencia

MIT License

---

# Autor

Script creado para automatizar el análisis masivo de archivos mediante la API de OpenAI.
```

