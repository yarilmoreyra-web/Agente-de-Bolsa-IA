# 📈 US Day Trading Pre-Market Agent

Agente automatizado de análisis bursátil para day trading en acciones de Estados Unidos. Diseñado para recopilar datos, detectar oportunidades, evaluarlas mediante Python e interpretar contexto con Inteligencia Artificial (Google Gemini), enviando un informe diario automatizado a Telegram.

> ⚠️ **Aviso legal**: Esta herramienta tiene fines puramente informativos y educativos. **No constituye asesoramiento financiero ni recomendación de inversión**. El day trading conlleva un riesgo elevado de pérdida de capital.

---

## 🕒 Cronograma Diario (Hora de Nueva York - ET)

- **08:35** — Arranque del workflow automatizado.
- **08:35 – 08:45** — Carga del universo, calendario bursátil y descarga de datos históricos y barras de 5 minutos (base del RVOL).
- **08:45** — **Snapshot pre-market** definitivo.
- **08:45 – 08:57** — Filtrado cuantitativo, consulta de noticias, análisis con Gemini y validación estricta de niveles/riesgos.
- **09:00** — Envío del informe operativo vía **Telegram**.
- **09:30** — Apertura del mercado (el agente no ejecuta órdenes automáticas).

---

## 🛠️ Estructura del Proyecto