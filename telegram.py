import os
import time
import logging
import requests

logger = logging.getLogger("agent.telegram")

def send_telegram_message(html_content: str, token: str = None, chat_id: str = None) -> bool:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados.")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # Dividir mensajes largos respetando el límite de 4096 caracteres (margen a 3800)
    max_length = 3800
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in html_content.split("\n"):
        line_len = len(line) + 1
        if current_length + line_len > max_length:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = line_len
        else:
            current_chunk.append(line)
            current_length += line_len
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    success = True
    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML"
        }
        
        retries = 3
        backoff = 2
        sent = False
        
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    sent = True
                    break
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", backoff))
                    logger.warning(f"Telegram rate limit alcanzado. Reintentando en {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Error Telegram API ({response.status_code}): {response.text}")
                    break
            except Exception as e:
                logger.error(f"Excepción en envío de Telegram (intento {attempt+1}): {e}")
                time.sleep(backoff)
                backoff *= 2
                
        if not sent:
            success = False
            break
            
    return success