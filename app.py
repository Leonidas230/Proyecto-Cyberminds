import os
import re
import math
import pickle
from collections import Counter
from flask import Flask, render_template_string, request, jsonify

import tiktoken
from openai import OpenAI

try:
    from duckduckgo_search import DDGS
    USAR_DDG = True
except ImportError:
    USAR_DDG = False

# 1. MÓDULO RAG (Índice de la Constitución)

def _extraer_texto_pdf(ruta: str) -> str:
    from pypdf import PdfReader
    lector = PdfReader(ruta)
    return "\n".join(pagina.extract_text() or "" for pagina in lector.pages)

def _cargar_texto(ruta: str) -> str:
    if ruta.lower().endswith(".pdf"):
        return _extraer_texto_pdf(ruta)
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()

_PATRON_ARTICULO = re.compile(r'(Art(?:í|i)culo\s+\d+\.?-?|Art\.\s*\d+\.?-?)', re.IGNORECASE)

def _dividir_en_articulos(texto: str):
    partes = _PATRON_ARTICULO.split(texto)
    articulos = []
    for i in range(1, len(partes) - 1, 2):
        encabezado = partes[i].strip()
        cuerpo = " ".join(partes[i + 1].split())
        numero = re.search(r'\d+', encabezado)
        if not cuerpo:
            continue
        articulos.append({
            "numero": numero.group() if numero else "?",
            "texto": cuerpo[:2000],
        })
    return articulos

_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "del", "un", "una", "que",
    "por", "con", "su", "sus", "se", "es", "para", "al", "o", "como", "e",
    "lo", "sin", "sobre", "entre", "les", "le",
}

def _tokenizar(texto: str):
    crudos = re.findall(r"[a-záéíóúñü]+", texto.lower())
    return [t for t in crudos if t not in _STOPWORDS and len(t) > 2]

class IndiceConstitucion:
    def __init__(self, articulos):
        self.articulos = articulos
        tokens_por_doc = [_tokenizar(a["texto"]) for a in articulos]
        n = len(tokens_por_doc)
        df = Counter()
        for tokens in tokens_por_doc:
            df.update(set(tokens))
        self._idf = {p: math.log((1 + n) / (1 + c)) + 1 for p, c in df.items()}
        self._vectores = [self._vectorizar(t) for t in tokens_por_doc]

    def _vectorizar(self, tokens):
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {p: (c / total) * self._idf.get(p, 0.0) for p, c in tf.items()}

    @staticmethod
    def _coseno(v1, v2):
        comunes = set(v1) & set(v2)
        if not comunes:
            return 0.0
        num = sum(v1[k] * v2[k] for k in comunes)
        n1 = math.sqrt(sum(v * v for v in v1.values())) or 1e-9
        n2 = math.sqrt(sum(v * v for v in v2.values())) or 1e-9
        return num / (n1 * n2)

    def buscar(self, pregunta: str, top_k: int = 5):
        vq = self._vectorizar(_tokenizar(pregunta))
        puntuados = sorted(
            ((self._coseno(vq, v), i) for i, v in enumerate(self._vectores)),
            key=lambda x: x[0], reverse=True,
        )
        return [self.articulos[i] for score, i in puntuados[:top_k] if score > 0]

def construir_indice(ruta_documento: str, ruta_cache: str = None) -> IndiceConstitucion:
    ruta_cache = ruta_cache or (ruta_documento + ".indice.pkl")
    huella = (os.path.getmtime(ruta_documento), os.path.getsize(ruta_documento))

    if os.path.exists(ruta_cache):
        try:
            with open(ruta_cache, "rb") as f:
                datos = pickle.load(f)
            if datos.get("huella") == huella:
                return IndiceConstitucion(datos["articulos"])
        except Exception:
            pass

    print(f"[INFO] Procesando '{ruta_documento}' y construyendo el índice...")
    texto = _cargar_texto(ruta_documento)
    articulos = _dividir_en_articulos(texto)
    with open(ruta_cache, "wb") as f:
        pickle.dump({"huella": huella, "articulos": articulos}, f)
    return IndiceConstitucion(articulos)


# 2. AGENTE CON REGLAS DE TEXTO

class AgenteConstitucionalHibrido:
    def __init__(self, client: OpenAI, ruta_constitucion: str = None, modelo="gemini-3.6-flash"):
        self.client = client
        self.modelo = modelo
        self.indice = construir_indice(ruta_constitucion) if ruta_constitucion and os.path.exists(ruta_constitucion) else None
        
        self.system_prompt = (
            "Eres un asistente jurídico experto en derecho constitucional ecuatoriano. "
            "REGLAS ESTRICTAS DE FORMATO: "
            "1. NO utilices símbolos de almohadilla (#) bajo ninguna circunstancia. "
            "2. NO uses asteriscos (*) ni markdown complejo para viñetas o títulos. Escribe texto plano estructurado con guiones comunes (-) o números simples. "
            "3. Sé conciso y directo para evitar que la respuesta se corte. "
            "Distingue claramente entre lo que establece textualmente la Constitución y la información externa."
        )

    def _buscar_en_web(self, query: str) -> str:
        if not USAR_DDG:
            return "[Búsqueda web desactivada]"
        try:
            results = DDGS().text(query, max_results=3)
            if not results:
                return "No se encontraron resultados relevantes en la web."
            return "\n".join([f"- Título: {r['title']}\n  Snippet: {r['body']}" for r in results])
        except Exception as e:
            return f"[Error en búsqueda web: {e}]"

    def consultar(self, pregunta_usuario: str, usar_investigacion: bool = False) -> str:
        contexto_constitucion = ""
        if self.indice:
            arts = self.indice.buscar(pregunta_usuario, top_k=4)
            if arts:
                bloque = "\n\n".join(f"Art. {a['numero']}: {a['texto']}" for a in arts)
                contexto_constitucion = f"Fragmentos constitucionales:\n{bloque}"

        contexto_web = ""
        if usar_investigacion:
            contexto_web = f"\n\nInvestigación web:\n{self._buscar_en_web(pregunta_usuario)}"

        prompt_final = (
            f"{contexto_constitucion}\n"
            f"{contexto_web}\n\n"
            f"Pregunta del usuario: {pregunta_usuario}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_final}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.modelo,
                messages=messages,
                temperature=0.2,
                max_tokens=2500  # Aumentado para evitar respuestas cortadas
            )
            return response.choices[0].message.content or "Sin respuesta."
        except Exception as e:
            return f"Error al comunicarse con la API: {e}"


# 3. INTERFAZ WEB LOCAL (FLASK)

app = Flask(__name__)

API_KEY = "AQ.Ab8RN6KhHgh9A3O_i8wER-MxyPMaf_nN10LFSh1p6zik7gq4bQ"
RUTA_PDF = r"C:\Users\Leoni\Downloads\Constitucion-de-la-Republica-del-Ecuador_act_ene-2021.pdf"

client = OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
agente = AgenteConstitucionalHibrido(client=client, ruta_constitucion=RUTA_PDF)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Agente Jurídico Constitucional Ecuatoriano</title>
    <!-- Librería opcional para procesar texto limpio de forma elegante si se desea -->
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; }
        .chat-container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        h2 { color: #2c3e50; text-align: center; }
        .chat-box { height: 450px; border: 1px solid #ccc; overflow-y: scroll; padding: 15px; margin-bottom: 15px; background: #fafafa; border-radius: 4px; }
        .message { margin-bottom: 15px; padding: 12px; border-radius: 5px; line-height: 1.5; white-space: pre-wrap; font-size: 14px; }
        .user { background: #d1e7dd; text-align: right; }
        .assistant { background: #cfe2ff; text-align: left; color: #111; }
        .controls { display: flex; gap: 10px; align-items: center; }
        textarea { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; height: 50px; resize: none; font-family: Arial, sans-serif; }
        button { padding: 10px 20px; background: #0d6efd; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background: #0b5ed7; }
        .checkbox-container { display: flex; align-items: center; gap: 5px; font-size: 14px; color: #333; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2> Agente Constitucional Híbrido (Ecuador)</h2>
        <div class="chat-box" id="chatBox">
            <div class="message assistant">¡Hola! Soy tu asistente constitucional. Pregúntame tus dudas o marca la casilla si deseas investigación web complementaria.</div>
        </div>
        <div class="controls">
            <textarea id="userInput" placeholder="Escribe tu consulta jurídica aquí..." onkeydown="if(event.key === 'Enter' && !event.shiftKey){event.preventDefault(); enviarMensaje();}"></textarea>
            <button onclick="enviarMensaje()">Enviar</button>
        </div>
        <div style="margin-top: 10px;">
            <label class="checkbox-container">
                <input type="checkbox" id="investigarWeb"> Investigar también por cuenta propia en la web (Fuera de la Constitución)
            </label>
        </div>
    </div>

    <script>
        async function enviarMensaje() {
            const input = document.getElementById('userInput');
            const chatBox = document.getElementById('chatBox');
            const investigarWeb = document.getElementById('investigarWeb').checked;
            const texto = input.value.trim();
            
            if (!texto) return;

            chatBox.innerHTML += `<div class="message user"><b>Tú:</b> ${texto}</div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "loading_" + Date.now();
            chatBox.innerHTML += `<div class="message assistant" id="${loadingId}">Pensando y consultando fuentes...</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pregunta: texto, investigar: investigarWeb })
                });
                const data = await response.json();
                
                document.getElementById(loadingId).remove();
                chatBox.innerHTML += `<div class="message assistant"><b>Agente:</b><br>${data.respuesta}</div>`;
            } catch (error) {
                document.getElementById(loadingId).remove();
                chatBox.innerHTML += `<div class="message assistant"><b>Error:</b> No se pudo procesar la solicitud.</div>`;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    pregunta = data.get("pregunta", "")
    investigar = data.get("investigar", False)
    
    respuesta = agente.consultar(pregunta, usar_investigacion=investigar)
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)