# Monris AI (PySide6 + Groq)

Interfaz de chat en escritorio construida con PySide6 y Groq. Incluye streaming de respuestas, selector de modelo, temperatura, perfiles, modo compacto, editor de system prompt por sesion, y metricas basicas por respuesta.

## Funcionalidades principales
- Streaming real de respuestas (token por token).
- Selector de modelo y temperatura por sesion.
- Perfiles rapidos (QA, Dev, Support) que ajustan prompt y temperatura.
- Editor de system prompt por sesion.
- Modo compacto para ocultar mensajes antiguos.
- Metricas por respuesta (latencia, tokens y costo si se configura pricing).
- UI con modal de configuracion y animaciones suaves.

## Requisitos
- Python 3.10+
- Dependencias:
  - PySide6
  - groq

## Instalacion
1) Crea y activa un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Instala dependencias:

```powershell
pip install PySide6 groq
```

## Configurar el API Key (PowerShell)
El sistema usa la variable de entorno `GROQ_API_KEY`.

Temporal (solo para la sesion actual de PowerShell):

```powershell
$env:GROQ_API_KEY="TU_API_KEY_AQUI"
```

Persistente para tu usuario (se mantiene en nuevas sesiones):

```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "TU_API_KEY_AQUI", "User")
```

Si usas la forma persistente, abre una nueva terminal para que tome el cambio.

## Ejecutar

```powershell
python test.py
```

## Uso rapido
- Presiona "Configuracion" para abrir el modal con modelos, temperatura, perfiles y prompt.
- Usa "Aplicar prompt" para guardar el prompt de esa sesion.
- "Modo compacto" oculta mensajes anteriores para chats largos.
- El indicador de estado muestra "Thinking..." mientras llega la respuesta.

## Notas
- El costo aparece como "N/A" hasta que se configure `PRICING_USD_PER_MILLION` en `test.py`.
- La lista de modelos se controla en `test.py` con `AVAILABLE_MODELS` y `EXCLUDED_MODELS`.
