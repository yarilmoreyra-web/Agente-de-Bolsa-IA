# Agente de análisis pre-market (day trading de acciones US)

Agente automático que, cada día bursátil, analiza tu lista de acciones y te envía
por Telegram un informe **30 minutos antes de la apertura** (09:00 hora de Nueva York).
Selecciona **hasta 3** candidatas para una operación intradía (entrada en la
apertura, salida el mismo día) o dice **NO OPERAR** si ninguna es lo bastante buena.

> ⚠️ **Herramienta informativa, no es asesoramiento financiero.** El agente
> **NO ejecuta operaciones**. Los datos gratuitos pueden ser incompletos o llegar
> con retraso. Operar en bolsa implica riesgo de pérdida. Si operas con una
> cuenta pequeña en EE. UU., infórmate de la regla PDT (pattern day trader) de
> tu bróker.

---

## 1. Qué hace, paso a paso

Hora de Nueva York (ET). Todo el cronograma se puede cambiar en `config.py`.

| Hora (ET) | Qué ocurre |
|---|---|
| ~08:35 | Arranca el workflow de GitHub Actions (puede retrasarse). |
| 08:35–08:45 | Lee `lista_tickers.xlsx`, comprueba el calendario bursátil, descarga histórico diario, barras de 5 min históricas (base del RVOL) y contexto de mercado. |
| **08:45** | **Snapshot pre-market**: toma los datos lo más tarde posible. |
| 08:45–08:57 | Calcula indicadores, filtra a 6-10 candidatas, busca noticias, consulta a Gemini, valida su respuesta y arma el informe. |
| **09:00** | **Envía el informe por Telegram.** Si llega tarde, lo marca «⏱ Informe tardío». |
| 09:30 | Apertura. El agente no opera. |

**Los 7 pasos del pipeline** (los verás numerados en la consola y en el log):

1. Descarga de datos diarios.
2. Barras de 5 minutos históricas (base del RVOL).
3. Contexto de mercado (SPY, QQQ, DIA, VIX, sectores, futuros, bono a 10 años, calendario macro).
4. Snapshot pre-market.
5. Indicadores, RVOL, niveles y fuerza relativa (**calcula Python**, no Gemini).
6. Noticias y catalizadores de las tickers que pasan el pre-filtro.
7. Selección de candidatas, análisis con Gemini, validación y informe.

**Reparto de trabajo:** Python descarga y calcula todo. Gemini solo *interpreta* los
datos que Python le entrega (catalizadores, momentum, riesgo de gap-and-fade,
riesgo/beneficio) y elige hasta 3. Gemini no puede inventar precios, noticias ni
niveles; si su respuesta contradice los datos de Python, **manda Python**.

---

## 2. Estructura del proyecto

```
trading-agent/
├── main.py                 punto de entrada (CLI, calendario, guarda horaria)
├── config.py               TODOS los horarios, umbrales, pesos y rutas
├── market_data.py          descarga (yfinance) y snapshot pre-market
├── technical_analysis.py   RSI, SMA, EMA, ATR, niveles, RVOL, fuerza relativa
├── news.py                 noticias y clasificación de catalizadores (modular)
├── market_context.py       SPY/QQQ/DIA/VIX/sectores/macro
├── candidate_filter.py     filtro y puntuación de candidatas
├── gemini_analyzer.py      llamada a Gemini y validación de su respuesta
├── report.py               informe en Markdown (y versión HTML para Telegram)
├── update_agent.py         actualización post-apertura: flujo completo (main.py --update)
├── post_open.py            métricas y estado de cada selección a las 10:00 (Python)
├── update_analyzer.py      Gemini compara «mañana vs ahora» + validación
├── update_report.py        texto del Telegram de la actualización
├── telegram.py             envío a Telegram (reintentos, mensajes largos)
├── history.py              histórico JSON y CSV
├── backtest.py             comparación posterior con el comportamiento real
├── utils.py                logging, horas, reintentos, lectura del Excel, calendario
├── lista_tickers.xlsx      tu universo de tickers (columna "Ticker")
├── requirements.txt / requirements-dev.txt
├── .env.example            plantilla de variables de entorno (sin valores)
├── tests/                  pruebas automáticas (pytest, sin internet)
├── data/
│   ├── history/            un JSON por día (AAAA-MM-DD.json)
│   ├── history.csv         una fila por candidata y día
│   ├── intraday/           barras de 5 min guardadas para el backtest
│   ├── reports/            informe de cada día en Markdown (+ AAAA-MM-DD_actualizacion.md)
│   ├── updates/            actualización de las 10:00 de cada día (AAAA-MM-DD.json)
│   ├── backtests/          resultados de la validación posterior
│   └── macro_events.json   eventos macro/Fed (lo mantienes tú)
├── logs/agent.log          registro de ejecución (no se sube a GitHub)
└── .github/workflows/      daily_agent.yml (09:00) y update_agent.yml (10:00)
```

Si tu proyecto tiene archivos adicionales (por ejemplo, un módulo que orquesta
el pipeline), consérvalos: este README describe el flujo, no obliga a una
estructura exacta.

---

## 3. Instalación local

Necesitas **Python 3.11 o superior** y una terminal abierta **dentro de la
carpeta del proyecto**.

```bash
# 1) Crear el entorno virtual (una sola vez)
py -m venv .venv                       # Windows   (en Mac/Linux: python3 -m venv .venv)

# 2) Activarlo (cada vez que abras una terminal nueva)
.venv\Scripts\Activate.ps1             # Windows PowerShell
source .venv/bin/activate              # Mac/Linux
# Debe aparecer (.venv) al inicio de la línea.
# Si Windows bloquea el script:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3) Instalar dependencias
pip install -r requirements.txt -r requirements-dev.txt

# 4) Preparar las claves
copy .env.example .env                 # Windows   (Mac/Linux: cp .env.example .env)
# Abre .env y rellena tus valores. Este archivo NUNCA se sube a GitHub.
```

**Tu lista de tickers:** `lista_tickers.xlsx` debe estar en la raíz del proyecto,
con la palabra `Ticker` en la primera celda de la columna y un ticker por celda.
El agente elimina vacíos y duplicados, pasa a mayúsculas y muestra
«Lista leída del archivo: N tickers». Nunca usa listas recordadas de otras sesiones.

---

## 4. Ejecución local

```bash
python -m pytest                                        # comprobar que todo funciona
python main.py --dry-run --no-gemini --no-telegram --force   # prueba completa sin enviar nada
python main.py --data-check 5 --date 2026-09-18         # ver datos reales de 5 tickers
python main.py                                          # ejecución normal (solo dentro de la ventana horaria)
```

| Opción | Efecto |
|---|---|
| `--date AAAA-MM-DD` | Analiza esa sesión (por defecto, hoy en Nueva York). |
| `--dry-run` | Genera el informe pero no envía a Telegram ni guarda histórico. |
| `--no-gemini` | Solo análisis de Python (sin IA). |
| `--no-telegram` | No envía nada a Telegram. |
| `--no-wait` | No espera hasta las 09:00 ET para enviar. |
| `--force` | Ignora calendario, ventana horaria y «ya enviado». Para pruebas. |
| `--universe RUTA` | Usa otro Excel de tickers. |
| `--data-check N` | Descarga y muestra indicadores de los N primeros tickers. |
| `--backtest-date AAAA-MM-DD` | Evalúa a posteriori el informe de esa fecha. |
| `--update` | Actualización post-apertura (10:00 ET); ver sección 4 bis. Acepta `--date`, `--dry-run`, `--no-gemini`, `--no-telegram`, `--no-wait` y `--force`. |

**Sin `--force`** el agente solo se ejecuta si hoy hay sesión bursátil, si la
hora en Nueva York está en la ventana permitida (08:00–09:25 por defecto) y si
el informe de hoy no se envió ya. Si no se cumple, termina con un mensaje claro
y sin error.

---

## 4 bis. Actualización post-apertura (10:00 ET)

Segunda pasada del día, **30 minutos después de la apertura**: el agente
reanaliza las candidatas del informe de las 09:00, las compara con lo que
dijo entonces y envía **otro Telegram** independiente.

| Hora (Nueva York) | Qué ocurre |
|---|---|
| ~09:45 | Arranca el workflow `update_agent.yml` y espera. |
| 10:00 | Toma los datos: barras de 5 min de la sesión regular (09:30–10:00). |
| 10:00–10:03 | Calcula, revisa noticias nuevas y consulta a Gemini (una llamada). |
| ~10:03 | Envía el Telegram y guarda `data/updates/AAAA-MM-DD.json`. |

En hora de Lima (mientras EE. UU. está en horario de verano) el mensaje llega
sobre las 09:03; desde el cambio de hora de noviembre, sobre las 10:03.

**Qué trae el mensaje**

* Mercado a las 08:45 vs ahora (SPY, QQQ, VIX).
* Por cada selección de la mañana: `Informe 09:00: BUY (HIGH) → Ahora: WAIT`,
  precio de apertura y actual, VWAP, volumen y RVOL de los primeros 30 min,
  estado, R/R con el precio actual, si se cumplió la regla de apertura, qué
  cambió y qué hacer, y noticias publicadas después del informe.
* Un resumen de cambios (confirmadas, debilitadas, invalidadas...) y las demás
  candidatas de la mañana, con hasta 2 para vigilar.

**Estados que decide Python (Gemini no puede contradecirlos)**

| Estado | Significa | Resultado |
|---|---|---|
| `STOP_TOCADO` | El precio llegó al stop del informe. | Invalidada · NO_TRADE |
| `OBJETIVO_1_ALCANZADO` | Llegó al objetivo 1 antes que al stop. | Objetivo alcanzado · NO_TRADE (no perseguir) |
| `EXTENDIDA` | Pasó el máximo de entrada válido. | Extendida · WAIT |
| `EN_CONTRA` | Bajo la zona de entrada, sin tocar el stop. | Debilitada · WAIT |
| `EN_ZONA` | Sigue en la zona. | Gemini valora; BUY solo si el R/R actual ≥ mínimo |

Si en una misma barra de 5 min se tocan el stop y el objetivo 1, se cuenta el
stop primero (lectura conservadora).

**Qué NO hace:** no inventa niveles nuevos (usa los del informe de las 09:00 y
solo recalcula el R/R), no busca tickers fuera de las candidatas de la mañana
y no modifica el histórico de la mañana. Si no hay informe de la mañana, o el
de hoy fue «NO OPERAR», lo dice y no genera señales nuevas.

Ajustes en `config.py` → `UPDATE`: hora de los datos (`snapshot_time`), ventana
válida (`run_window_start/end`), nº de candidatas (`max_tickers`), tamaño de la
lista de vigilancia (`watchlist_size`) y si se refrescan las noticias.

Prueba local (con el histórico de esa fecha ya generado):

```bash
python main.py --update --date 2026-09-25 --dry-run --force --no-wait   # muestra el mensaje, no envía
```

En GitHub Actions la ejecución manual de **Actualización post-apertura** admite
las mismas opciones (`date`, `dry_run`, `no_gemini`, `force`). Comparte grupo de
concurrencia con el workflow de las 09:00: nunca se ejecutan a la vez.

---

## 5. Claves y servicios (todo gratis)

### Gemini (IA)
1. Entra en **Google AI Studio** y crea una API key.
2. Guárdala como `GEMINI_API_KEY`.
3. El modelo se elige con `GEMINI_MODEL` (por ejemplo `gemini-2.5-flash`). Si lo
   dejas vacío se usa el valor por defecto de `config.py` (`DEFAULT_GEMINI_MODEL`).
   Los modelos cambian con el tiempo: si un día Gemini da error de «modelo no
   encontrado», cambia solo esa variable, sin tocar el código.
4. El agente hace **una sola llamada al día**, muy por debajo de las cuotas
   gratuitas. Los límites y las condiciones del plan gratuito (incluido el uso de
   los datos enviados) cambian: compruébalos en Google AI Studio.

### Telegram
1. En Telegram, escribe a **@BotFather** → `/newbot` → elige nombre y usuario.
   Te dará un **token** parecido a `123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.
   Guárdalo como `TELEGRAM_BOT_TOKEN`.
2. **Escríbele cualquier mensaje a tu bot** (obligatorio; si no, no puede escribirte).
3. Abre en el navegador `https://api.telegram.org/bot<TU_TOKEN>/getUpdates` y busca
   el número que aparece tras `"chat":{"id":`. Ese es tu `TELEGRAM_CHAT_ID`
   (si es un grupo, empieza por `-`).
4. Prueba la conexión (sustituyendo tus datos, sin los símbolos `<` `>`):
   - `https://api.telegram.org/bot<TU_TOKEN>/getMe` → debe responder `"ok":true`.
   - `https://api.telegram.org/bot<TU_TOKEN>/sendMessage?chat_id=<TU_CHAT_ID>&text=prueba` → te llega «prueba».

> No compartas nunca el token (ni en capturas). Si se expone, revócalo en @BotFather (`/mybots` → tu bot → API Token → Revoke).

### Alpha Vantage (opcional)
Segunda fuente de noticias. Clave gratuita en alphavantage.co, guardada como
`ALPHAVANTAGE_API_KEY`. Solo se consulta para las candidatas finalistas, para
respetar la cuota diaria gratuita (compruébala en su documentación). Si no la
pones, el agente funciona igual con las noticias de yfinance.

---

## 6. Puesta en marcha en GitHub Actions

1. **Sube el proyecto** a un repositorio de GitHub (incluido `lista_tickers.xlsx` y
   la carpeta oculta `.github/workflows/`). **No subas** `.env` ni `.venv`
   (ya están en `.gitignore`).
   - *Repositorio público:* minutos de Actions ilimitados, pero el histórico
     (`data/`) será visible para cualquiera.
   - *Repositorio privado:* la cuenta gratuita incluye 2.000 min/mes. El agente
     usa ≈ 25 min por día hábil (espera hasta las 08:45 ET), unos 500 min/mes.
2. **Secrets** (Settings → Secrets and variables → Actions → pestaña **Secrets** → New repository secret):

   | Nombre | Valor |
   |---|---|
   | `GEMINI_API_KEY` | tu clave de Gemini |
   | `TELEGRAM_BOT_TOKEN` | el token del bot |
   | `TELEGRAM_CHAT_ID` | tu chat_id |
   | `ALPHAVANTAGE_API_KEY` | (opcional) |

   Los nombres deben ser **exactamente** esos. Pega el valor sin comillas, sin
   espacios y sin saltos de línea al final.
3. **Variable** (misma pantalla, pestaña **Variables**): `GEMINI_MODEL` (por
   ejemplo `gemini-2.5-flash`). Es una *variable*, no un secret, para que un
   valor vacío nunca pise el modelo por defecto.
4. **Permisos:** Settings → Actions → General → Workflow permissions → marca
   **Read and write permissions**, para que el workflow pueda guardar el histórico.
5. **Primera prueba manual:** pestaña **Actions** → «Agente pre-market diario» →
   **Run workflow**. Marca `force` y **desmarca** `dry_run` para recibir un
   informe real de prueba; o deja `dry_run` marcado para probar sin enviar.
6. **Revisa el log:** abre la ejecución → paso «Ejecutar el agente». El log se
   puede descargar también como artefacto `agent-log`.

**Ejecución programada.** El workflow tiene dos cron (12:35 y 13:35 UTC, de lunes a
viernes) porque GitHub trabaja en UTC y Nueva York cambia de horario. Solo uno de
los dos cae a las ~08:35 ET; el otro se descarta solo. Así funciona todo el año
sin tocar nada, incluso en los cambios de hora de EE. UU. (**1-nov-2026** y
**14-mar-2027**).

Cosas a saber:
- Los cron de GitHub pueden retrasarse 10-30 min (a veces más). Por eso arrancan
  25 minutos antes del snapshot.
- Solo se ejecutan si el workflow está en la **rama principal**.
- En repositorios públicos, GitHub desactiva los workflows programados tras
  60 días sin actividad. Si un día deja de ejecutarse, entra en Actions y
  reactívalo (confirma los detalles en la documentación de GitHub).
- El histórico se guarda con un commit automático al final de cada ejecución.

---

## 7. Cómo decide el agente

### RVOL (volumen relativo pre-market)
```
RVOL = V_hoy / mediana(V_hist)
```
- `V_hoy`: volumen acumulado de las barras de 5 min con inicio entre las 04:00 y la
  hora de corte del snapshot (08:45 ET).
- `V_hist`: el volumen acumulado **en la misma ventana horaria** de cada una de las
  últimas 20 sesiones. Se usa la **mediana** para que un día extraordinario no
  distorsione la referencia.
- Se exigen al menos 10 sesiones históricas con datos. Si no, el RVOL es **N/A**
  con el motivo. Nunca se calcula como «volumen pre-market / volumen diario».
- Aparte se muestra `pm_pct_of_adv` (volumen pre-market / volumen medio diario, en %),
  que es otra métrica y **no** se llama RVOL.

### Filtro de candidatas (valores por defecto, editables en `config.py`)
**Puertas duras** (eliminan al ticker): precio ≥ 5 USD · volumen medio 20 d ≥
1.000.000 acciones · volumen medio en dólares ≥ 20 M · volumen pre-market ≥
25.000 · |gap| ≥ 1 % · datos pre-market disponibles.

**Puntuación 0–100:** RVOL 25 · gap 20 · catalizador confirmado 20 · liquidez 10 ·
fuerza relativa 10 · estructura técnica 10 · comportamiento pre-market 5.

**Penalizaciones de gap-and-fade:** gap > 3 veces el ATR% (extensión) · precio ≥ 2 %
por debajo del máximo pre-market · resistencia a menos de 0,5 ATR · sin catalizador
· volumen bajo para el tamaño del gap · mercado adverso (VIX alto, futuros negativos).

Se envían a Gemini las 6-10 mejores con puntuación ≥ 40. Si ninguna llega, **no se
llama a Gemini** y el informe dice NO OPERAR con el motivo. El registro completo
de cada ticker (puntuación y motivo de exclusión) queda en el histórico.

### Catalizadores
Titular, publisher, hora y URL de yfinance (y Alpha Vantage si la activas),
clasificados en earnings, guidance, FDA, contratos, M&A, regulación, cambios de
recomendación, precios objetivo, productos, acuerdos, litigios, financiación,
insiders, sector y macro. Solo las categorías «duras» cuentan como **catalizador
confirmado** y solo si son de las últimas 48 h. Si no hay: «Sin catalizador confirmado».

### Validación de Gemini
Python revisa cada respuesta antes de generar el informe:
- descarta tickers que no envió;
- **sobrescribe** precio, gap, volumen y RVOL con los datos de Python;
- comprueba que stop y objetivos tengan sentido para la dirección de la operación;
- exige que soportes y resistencias existan en la tabla de niveles;
- **calcula el R/R en Python**: (objetivo 1 − entrada) / (entrada − stop). Si es
  menor que 1,5 (`MIN_RR`), la candidata pasa a WAIT o se descarta.

Si Gemini falla o devuelve un JSON inválido, el informe se genera igualmente solo
con datos de Python y lo avisa («Análisis IA no disponible»).

### Entrada prevista vs. entrada confirmada
- **Entrada prevista:** rango basado en el pre-market y los niveles técnicos. Es una
  referencia, **nunca** un precio de ejecución garantizado.
- **Entrada confirmada:** depende del precio real de apertura. Cada candidata trae un
  **precio máximo de entrada válido**: si la apertura lo supera, el R/R deja de
  cumplirse y la regla es **NO ENTRAR**.

---

## 8. Mantenimiento

- **Eventos macro/Fed:** no hay fuente automática gratuita fiable. Edita
  `data/macro_events.json` a mano (FOMC, IPC, empleo…). Formato de cada evento:
  `{"date": "AAAA-MM-DD", "time_et": "HH:MM", "event": "texto", "importance": "HIGH|MEDIUM|LOW"}`.
  Si un día no está cargado, el informe lo dice; no inventa fechas.
- **Cambiar de modelo de Gemini:** modifica la variable `GEMINI_MODEL` (en `.env` o en
  las Variables de GitHub).
- **Ajustar el filtro, los pesos, `MIN_RR` o `DIRECTION_MODE`** (`long_only` o `both`):
  todo está en `config.py`.
- **Cambiar la lista de tickers:** edita `lista_tickers.xlsx` y súbelo a GitHub.

---

## 9. Histórico y validación posterior (backtest)

Cada ejecución real guarda `data/history/AAAA-MM-DD.json` (universo, datos,
indicadores, catalizadores, `filter_log`, respuesta de Gemini cruda y validada,
selección final, fuentes, timestamps) y añade filas a `data/history.csv`. También
guarda las barras de 5 min del día de las candidatas, porque yfinance no las
servirá después.

Para comparar lo que predijo el agente con lo que pasó, **al terminar la sesión**:

```bash
python main.py --backtest-date 2026-09-21
```

Calcula apertura, máximo, mínimo y cierre del día, máximo favorable/adverso y si se
tocó el objetivo o el stop. Si stop y objetivo caen en la misma barra de 5 min lo
marca como «ambiguo» y lo trata de forma conservadora (como stop). Los resultados
van a `data/backtests/` y `data/backtest_summary.csv`.

Usa **3-4 semanas de backtest** para ajustar umbrales y pesos antes de fiarte del
agente. No es un backtester complejo: es una primera medida de acierto.

---

## 10. Limitaciones conocidas

- **yfinance no es una API oficial:** puede fallar, cambiar o limitar peticiones, y
  Yahoo puede bloquear IPs compartidas de GitHub Actions en días de mucho tráfico.
- El **volumen pre-market** puede llegar incompleto o con retraso; las acciones poco
  líquidas tienen pocas barras. Cada dato lleva una etiqueta de calidad
  (`ok` / `sparse` / `stale` / `missing`).
- Las barras de 5 min solo existen para los últimos ~60 días.
- `ticker.news` es escaso y su formato cambia: algunos días saldrá «Sin catalizador
  confirmado» aunque exista uno.
- Los cron de GitHub pueden retrasarse; si el informe sale tarde, va marcado.
- Un informe no garantiza nada: el gap puede revertirse (*gap-and-fade*).

---

## 11. Solución de problemas

**No me llega nada a Telegram.** Abre el log de la ejecución (Actions → ejecución →
«Ejecutar el agente») y busca:

| Mensaje en el log | Causa y solución |
|---|---|
| «Fuera de la ventana de ejecución» | Lanzaste a mano fuera de 08:00–09:25 ET. Marca `force`. |
| «Sin sesión bursátil» | Fin de semana o festivo. Es normal. |
| «ya fue enviado» | El histórico de hoy ya está marcado como enviado. Usa `force` para repetir. |
| `Modo: dry-run` / `no-telegram` | Ese flag desactiva el envío. |
| `Telegram: no` en «Claves detectadas» | Los secrets no llegan. Revisa nombres exactos, que estén en **Secrets** (no en Variables) y sin espacios. |
| HTTP **401** | Token incorrecto o revocado. |
| HTTP **404** | El token está mal escrito o la URL tiene un error de formato (falta la `:`, espacios, o quedaron los símbolos `< >`). Genera uno nuevo en @BotFather. |
| HTTP **400** «chat not found» | `TELEGRAM_CHAT_ID` incorrecto, o nunca le escribiste al bot. |
| HTTP **403** | Bloqueaste el bot o lo sacaron del grupo. |

**El dato pre-market sale `stale` en los tests o en una prueba manual.** Si ejecutas
después de las 08:45 ET con la fecha de hoy, el corte pasa a ser la hora actual y
las barras de prueba quedan «viejas». En los tests se arregla congelando la hora;
en producción no ocurre porque el agente corre a las 08:45.

**Muchos `N/A` en una prueba local.** Prueba con `--date` de un día de mercado
reciente (últimos ~60 días) y con `--force`.

**El workflow no se ejecuta solo.** Comprueba que está en la rama principal, que
Actions está activado en el repositorio y que no lleva 60 días sin actividad.
Recuerda que puede retrasarse; espera al menos 30-40 minutos antes de preocuparte.

**Falla el paso «Guardar el histórico».** Revisa Settings → Actions → General →
Workflow permissions → **Read and write permissions**.

---

## 12. Verificación antes de dar el proyecto por bueno

```bash
python -m compileall .                                        # sin errores de sintaxis
python -m pytest                                              # todo en verde
python main.py --dry-run --no-gemini --no-telegram --force    # informe completo en pantalla
```

Y una ejecución manual en Actions con `force` = true y `dry_run` = false, para
confirmar que el informe llega a Telegram y que el histórico se guarda en `data/`.
