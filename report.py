import os
import html
from datetime import datetime
from typing import Dict, Any, List

def generate_markdown_report(data: Dict[Any, Any], session_type: str = "sesión normal", late: bool = False) -> str:
    """
    Genera el informe diario en formato Markdown canónico.
    """
    date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    timestamp = data.get("timestamp", datetime.now().strftime("%H:%M ET"))
    
    header_time = f"🕒 Hora del snapshot y envío: {timestamp} ET"
    if late:
        header_time += " ⏱ Informe tardío"
        
    md = []
    md.append(f"📅 **FECHA**: {date_str} ({session_type})")
    md.append(header_time)
    md.append(f"📊 **CONTEXTO DE MERCADO**: {data.get('market_view', 'N/A')}")
    
    universe_count = data.get("universe_count", 0)
    md.append(f"📋 **Universo**: {universe_count} tickers (Lista leída del archivo)")
    
    picks = data.get("picks", [])
    filtered_count = len(data.get("filtered_candidates", []))
    md.append(f"🔎 **CANDIDATAS ANALIZADAS**: {filtered_count} con puntuación, seleccionadas {len(picks)}")
    md.append("")
    
    if data.get("no_trade", False) or not picks:
        reason = data.get("no_trade_reason", "No se encontraron configuraciones con suficientes garantías.")
        md.append(f"🚫 **NO OPERAR**")
        md.append(f"Motivo: {reason}")
        md.append("")
    else:
        medals = ["🥇", "🥈", "🥉"]
        for idx, pick in enumerate(picks):
            medal = medals[idx] if idx < len(medals) else "🔹"
            md.append(f"{medal} **{pick.get('ticker')}** — {pick.get('company', 'N/A')}")
            md.append(f"- **Catalizador / Fuente**: {pick.get('catalyst', 'N/A')} / {pick.get('catalyst_source', 'N/A')}")
            md.append(f"- **Cierre anterior / Pre-market / Gap**: {pick.get('prev_close', 'N/A')} / {pick.get('premarket_price', 'N/A')} / {pick.get('gap_pct', 'N/A')}%")
            md.append(f"- **Volumen / RVOL**: {pick.get('volume', 'N/A')} / {pick.get('rvol', 'N/A')}")
            md.append(f"- **RSI / SMA20 / SMA50 / ATR**: {pick.get('rsi', 'N/A')} / {pick.get('sma20', 'N/A')} / {pick.get('sma50', 'N/A')} / {pick.get('atr', 'N/A')}")
            md.append(f"- **Soporte / Resistencia**: {pick.get('support_level', 'N/A')} / {pick.get('resistance_level', 'N/A')}")
            md.append(f"- **Fuerza relativa**: {pick.get('relative_strength', 'N/A')}")
            md.append(f"- **Riesgo gap & fade**: {pick.get('fade_risk', 'N/A')} — {pick.get('fade_risk_explanation', '')}")
            md.append(f"- **Entrada prevista**: {pick.get('entry_zone_low', 'N/A')} - {pick.get('entry_zone_high', 'N/A')}")
            md.append(f"- **Stop / Obj 1 / Obj 2 / R/R**: {pick.get('stop', 'N/A')} / {pick.get('target_1', 'N/A')} / {pick.get('target_2', 'N/A')} / {pick.get('rr_ratio', 'N/A')}")
            md.append(f"- **Precio máximo de entrada válido**: {pick.get('max_valid_entry', 'N/A')}")
            md.append(f"- **Hora objetivo / Invalidación**: {pick.get('exit_time', '09:30 - 16:00 ET')} / {pick.get('invalidation', 'N/A')}")
            md.append(f"- **Decisión / Razón**: {pick.get('decision', 'WAIT')} — {pick.get('reason', '')}")
            md.append("")
            
    md.append("⚠️ **RIESGOS**: Operar en apertura conlleva alta volatilidad. Utilice órdenes limitadas y respete estrictamente el stop loss.")
    md.append("\n*Herramienta informativa, no es asesoramiento financiero.*")
    
    return "\n".join(md)

def markdown_to_telegram_html(md_text: str) -> str:
    """
    Convierte el reporte en Markdown básico a HTML seguro compatible con la API de Telegram.
    """
    lines = md_text.split("\n")
    html_lines = []
    
    for line in lines:
        escaped = html.escape(line)
        # Reemplazar negritas simples **texto** por <b>texto</b>
        while "**" in escaped:
            escaped = escaped.replace("**", "<b>", 1)
            escaped = escaped.replace("**", "</b>", 1)
        # Reemplazar cursivas *texto* por <i>texto</i> (siempre que no rompa etiquetas)
        while "*" in escaped:
            escaped = escaped.replace("*", "<i>", 1)
            escaped = escaped.replace("*", "</i>", 1)
        html_lines.append(escaped)
        
    return "\n".join(html_lines)

def save_report_to_file(md_content: str, date_str: str) -> str:
    os.makedirs("data/reports", exist_ok=True)
    file_path = f"data/reports/{date_str}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return file_path