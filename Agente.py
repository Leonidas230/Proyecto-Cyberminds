"""
AgenteConstitucional — asistente jurídico en derecho constitucional ecuatoriano,
con recuperación de información (RAG) sobre el texto real de la Constitución.

Requisitos:
    pip install openai tiktoken pypdf

Uso:
    1. Descarga el PDF (o .txt) de la Constitución de la República del Ecuador
       VIGENTE (con reformas incluidas) desde una fuente oficial y colócalo
       junto a este script, por ejemplo como "constitucion_ecuador.pdf".
    2. Define la variable de entorno GEMINI_API_KEY.
    3. Ejecuta: python Agente.py
"""

import os
import re
import math
import pickle
from collections import Counter

import tiktoken
from openai import OpenAI


# 1. Ingesta y recuperación de la Constitución (RAG)

def _extraer_texto_pdf(ruta: str) -> str:
    from pypdf import PdfReader
    lector = PdfReader(ruta)
    return "\n".join(pagina.extract_text() or "" for pagina in lector.pages)


def _cargar_texto(ruta: str) -> str:
    if ruta.lower().endswith(".pdf"):
        return _extraer_texto_pdf(ruta)
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()


# Detecta encabezados de artículo tipo "Art. 66.-" o "Artículo 66.-"
_PATRON_ARTICULO = re.compile(r'(Art(?:í|i)culo\s+\d+\.?-?|Art\.\s*\d+\.?-?)', re.IGNORECASE)


def _dividir_en_articulos(texto: str):
    partes = _PATRON_ARTICULO.split(texto)
    articulos = []
    # partes[0] es el preámbulo/índice antes del primer "Art."; se descarta.
    for i in range(1, len(partes) - 1, 2):
        encabezado = partes[i].strip()
        cuerpo = " ".join(partes[i + 1].split())  # colapsa saltos de línea/espacios
        numero = re.search(r'\d+', encabezado)
        if not cuerpo:
            continue
        articulos.append({
            "numero": numero.group() if numero else "?",
            "texto": cuerpo[:2000],  # límite defensivo por artículo
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
    """Índice TF-IDF minimalista (sin dependencias externas de ML) sobre los
    artículos de la Constitución. Permite recuperar, para cada pregunta, los
    artículos más relevantes ANTES de llamar al modelo (RAG), en vez de
    depender del conocimiento paramétrico del LLM."""

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
    """Construye el índice de artículos, reutilizando una caché en disco
    (pickle) mientras el archivo fuente no cambie de tamaño ni fecha."""
    ruta_cache = ruta_cache or (ruta_documento + ".indice.pkl")
    huella = (os.path.getmtime(ruta_documento), os.path.getsize(ruta_documento))

    if os.path.exists(ruta_cache):
        try:
            with open(ruta_cache, "rb") as f:
                datos = pickle.load(f)
            if datos.get("huella") == huella:
                return IndiceConstitucion(datos["articulos"])
        except Exception:
            pass  # caché corrupta o incompatible: se reconstruye desde cero

    print(f"[INFO] Procesando '{ruta_documento}' y construyendo el índice de artículos...")
    texto = _cargar_texto(ruta_documento)
    articulos = _dividir_en_articulos(texto)
    if not articulos:
        raise ValueError(
            "No se detectaron artículos en el documento. Verifica que sea el texto "
            "completo de la Constitución y que use el formato 'Art. N.-' o 'Artículo N.-'."
        )
    print(f"[INFO] Se detectaron {len(articulos)} artículos.")

    with open(ruta_cache, "wb") as f:
        pickle.dump({"huella": huella, "articulos": articulos}, f)

    return IndiceConstitucion(articulos)


# ---------------------------------------------------------------------------
# 2. Agente conversacional
# ---------------------------------------------------------------------------

class AgenteConstitucional:
    def __init__(self, client: OpenAI, ruta_constitucion: str = None,
                 modelo="gemini-3.6-flash", max_turnos_historial=6, top_k_articulos=5):
        self.client = client
        self.modelo = modelo
        self.max_turnos_historial = max_turnos_historial
        self.top_k_articulos = top_k_articulos

        # NOTA: tiktoken (cl100k_base) es el tokenizador de OpenAI, no el de
        # Gemini; se usa solo como ESTIMACIÓN aproximada cuando la API no
        # devuelve 'usage'. Si falla la descarga del encoding, se cae a un
        # conteo por palabras para no romper el programa.
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

        self.indice = None
        if ruta_constitucion:
            self.indice = construir_indice(ruta_constitucion)
        else:
            print("[AVISO] No se proporcionó el texto de la Constitución: el agente "
                  "responderá solo con el conocimiento general del modelo, sin "
                  "fragmentos verificados (mayor riesgo de referencias incorrectas).")

        self.system_prompt = {
            "role": "system",
            "content": (
                "Eres un asistente jurídico experto en derecho constitucional ecuatoriano. "
                "En cada consulta recibirás, cuando estén disponibles, fragmentos reales de "
                "la Constitución de la República del Ecuador recuperados automáticamente de "
                "un índice local. Responde basándote ÚNICAMENTE en esos fragmentos: cita el "
                "número de artículo tal como aparece en ellos y no completes con conocimiento "
                "propio los detalles que no figuren en el fragmento. Si los fragmentos "
                "proporcionados no contienen la respuesta, dilo explícitamente en vez de "
                "inventar un artículo, norma o sentencia. Sé claro, formal y preciso, y "
                "distingue siempre entre texto constitucional citado, interpretación jurídica "
                "y explicación propia."
            )
        }
        self.historial = [self.system_prompt]

    def _contar_tokens_local(self, texto: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(texto))
        return len(texto.split())

    def _recortar_historial(self):
        max_mensajes = 1 + self.max_turnos_historial * 2
        if len(self.historial) > max_mensajes:
            self.historial = [self.system_prompt] + self.historial[-(max_mensajes - 1):]

    def _construir_mensaje_usuario(self, pregunta_usuario: str) -> str:
        """Recupera los artículos más relevantes (RAG) y arma el mensaje que
        realmente se envía al modelo, sin alterar la pregunta original que
        se guarda en el historial visible."""
        if not self.indice:
            return pregunta_usuario

        articulos = self.indice.buscar(pregunta_usuario, top_k=self.top_k_articulos)
        if not articulos:
            return (
                f"{pregunta_usuario}\n\n"
                "[No se encontraron artículos con similitud suficiente en el índice local. "
                "Indica explícitamente que no puedes confirmar una referencia exacta.]"
            )

        bloque = "\n\n".join(f"Art. {a['numero']}: {a['texto']}" for a in articulos)
        return (
            "Fragmentos recuperados de la Constitución (usa solo estos para responder):\n"
            f"{bloque}\n\n"
            f"Pregunta del usuario: {pregunta_usuario}"
        )

    def consultar(self, pregunta_usuario, temperatura=0.2, max_tokens=1200):
        mensaje_para_modelo = self._construir_mensaje_usuario(pregunta_usuario)
        # Se trabaja sobre una copia tentativa: el historial real solo se
        # actualiza si la llamada a la API tiene éxito (evita turnos "user"
        # huérfanos que romperían la alternancia de roles en la API).
        historial_tentativo = self.historial + [{"role": "user", "content": mensaje_para_modelo}]

        print(f"\n[INFO] Enviando consulta al modelo '{self.modelo}'...")

        try:
            response = self.client.chat.completions.create(
                model=self.modelo,
                messages=historial_tentativo,
                temperature=temperatura,
                max_tokens=max_tokens
            )

            respuesta_ia = response.choices[0].message.content
            if respuesta_ia is None:
                respuesta_ia = "[El modelo no devolvió contenido de texto en la respuesta]"

            # Se guarda la pregunta ORIGINAL (sin los fragmentos) en el
            # historial visible, para no inflar el contexto en turnos
            # futuros; los fragmentos relevantes se vuelven a recuperar en
            # cada consulta según la pregunta de ese turno.
            self.historial.append({"role": "user", "content": pregunta_usuario})
            self.historial.append({"role": "assistant", "content": respuesta_ia})
            self._recortar_historial()

            usage = response.usage
            print("\n--- REPORTE DE TOKENS DE ESTA CONSULTA ---")
            if usage:
                print(f"• Tokens en el prompt (Entrada): {usage.prompt_tokens}")
                print(f"• Tokens en la respuesta (Salida): {usage.completion_tokens}")
                print(f"• Tokens Totales Consumidos: {usage.total_tokens}")
            else:
                tokens_prompt_est = self._contar_tokens_local(mensaje_para_modelo)
                tokens_respuesta_est = self._contar_tokens_local(respuesta_ia)
                print("[AVISO] La API no devolvió 'usage'. Estimación local (aproximada):")
                print(f"• Tokens prompt (estimado): {tokens_prompt_est}")
                print(f"• Tokens respuesta (estimado): {tokens_respuesta_est}")
            print("--------------------------------------------\n")

            return respuesta_ia

        except Exception as e:
            print(f"[ERROR] Falló la llamada a la API: {e}")
            return f"Ocurrió un error al comunicarse con la API: {e}"


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Define la variable de entorno GEMINI_API_KEY antes de ejecutar.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    RUTA_CONSTITUCION = r"C:\Users\Leoni\Downloads\Constitucion-de-la-Republica-del-Ecuador_act_ene-2021.pdf"

    agente = AgenteConstitucional(
        client=client,
        ruta_constitucion=RUTA_CONSTITUCION if os.path.exists(RUTA_CONSTITUCION) else None,
        modelo="gemini-3.6-flash",
    )

    print("--- AGENTE JURÍDICO CONSTITUCIONAL INICIADO ---")

    pregunta = "¿Cuáles son mis derechos como Ciudadano Ecuatoriano?"
    print(f"Usuario: {pregunta}")

    respuesta = agente.consultar(pregunta, temperatura=0.1)
    print(f"Agente:\n{respuesta}")