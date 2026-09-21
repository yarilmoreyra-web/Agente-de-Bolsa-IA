"""Tests de telegram.py con un cliente HTTP simulado (sin red)."""
from __future__ import annotations

import re
import unittest
from dataclasses import replace
from unittest import mock

import config
import report
import telegram as tg
from html_utils import is_balanced, strip_tags
from sample_data import make_report_data

TOKEN = "123456:ABC-super-secret-token"
CHAT = "-100999"
ENV = config.EnvSettings(telegram_bot_token=TOKEN, telegram_chat_id=CHAT)


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def ok(message_id=1):
    return FakeResponse(200, {"ok": True, "result": {"message_id": message_id}})


class FakePoster:
    """Devuelve (o lanza) respuestas en secuencia y guarda las llamadas."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": dict(json), "timeout": timeout})
        response = self.responses.pop(0) if self.responses else ok()
        if isinstance(response, Exception):
            raise response
        return response


def no_sleep():
    return mock.patch.object(tg.time_module, "sleep")


class TestSplitMessage(unittest.TestCase):
    def test_mensaje_corto_no_se_divide(self):
        self.assertEqual(tg.split_message("<b>hola</b>", 100), ["<b>hola</b>"])
        self.assertEqual(tg.split_message("  \n ", 100), [])

    def test_agrupa_bloques_enteros(self):
        bloques = [f"<b>Candidata {i}</b>\n" + ("x" * 900) for i in range(6)]
        partes = tg.split_message("\n\n".join(bloques), 2000)
        self.assertEqual(len(partes), 3)
        for parte in partes:
            self.assertLessEqual(len(parte), 2000)
            self.assertTrue(is_balanced(parte))
        # cada trozo termina en un límite de bloque: ningún bloque queda partido
        unidos = "\n\n".join(partes)
        self.assertEqual(unidos, "\n\n".join(bloques))

    def test_bloque_largo_se_divide_por_lineas(self):
        lineas = [f"• línea {i:03d} " + "y" * 80 for i in range(40)]
        partes = tg.split_message("\n".join(lineas), 500)
        self.assertGreater(len(partes), 1)
        self.assertTrue(all(len(p) <= 500 for p in partes))
        self.assertEqual("\n".join(partes), "\n".join(lineas))     # nada perdido

    def test_linea_enorme_con_etiquetas_se_reequilibra(self):
        texto = "<b>" + "palabra " * 300 + "</b> fin"
        partes = tg.split_message(texto, 200)
        self.assertGreater(len(partes), 5)
        for parte in partes:
            self.assertLessEqual(len(parte), 200)
            self.assertTrue(is_balanced(parte), parte)
        self.assertTrue(all(p.startswith("<b>") for p in partes[:-1]))
        original = " ".join(strip_tags(texto).split())
        recompuesto = " ".join(" ".join(strip_tags(p) for p in partes).split())
        self.assertEqual(recompuesto, original)

    def test_etiquetas_anidadas_y_enlace(self):
        texto = '<b>negrita <a href="https://x.test/a?b=1&amp;c=2">enlace largo</a> ' + \
                "texto " * 100 + "</b>"
        for parte in tg.split_message(texto, 150):
            self.assertLessEqual(len(parte), 150)
            self.assertTrue(is_balanced(parte), parte)

    def test_no_corta_entidades(self):
        texto = "a " + "&amp;" * 500 + " b"
        partes = tg.split_message(texto, 100)
        for parte in partes:
            self.assertLessEqual(len(parte), 100)
            self.assertIsNone(re.search(r"&(?!amp;)", parte), parte)
            self.assertIsNone(re.search(r"&amp(?!;)", parte), parte)

    def test_palabra_mas_larga_que_el_limite(self):
        partes = tg.split_message("<b>" + "z" * 1000 + "</b>", 300)
        self.assertTrue(all(len(p) <= 300 and is_balanced(p) for p in partes))
        self.assertEqual("".join(strip_tags(p) for p in partes), "z" * 1000)

    def test_etiquetas_que_abarcan_bloques_se_equilibran(self):
        partes = tg.split_message("<b>" + "a" * 50 + "\n\n" + "b" * 50 + "</b>", 60)
        self.assertEqual(len(partes), 2)
        self.assertTrue(all(is_balanced(p) for p in partes))
        self.assertTrue(partes[1].startswith("<b>"))

    def test_informe_real_respeta_el_limite_configurado(self):
        data = make_report_data()
        for pick in data.gemini["picks"]:
            pick["reason"] = "razón muy larga " * 150
        html_text = report.markdown_to_telegram_html(report.build_report(data))
        partes = tg.split_message(html_text)
        self.assertGreater(len(partes), 1)
        for parte in partes:
            self.assertLessEqual(len(parte), config.TELEGRAM.max_message_chars)
            self.assertLess(len(parte), 4096)
            self.assertTrue(is_balanced(parte))


class TestSendMessage(unittest.TestCase):
    def test_envio_correcto(self):
        poster = FakePoster(ok(77))
        message_id = tg.send_message("<b>hola</b>", TOKEN, CHAT, http_post=poster)
        self.assertEqual(message_id, 77)
        call = poster.calls[0]
        self.assertTrue(call["url"].endswith("/sendMessage"))
        self.assertEqual(call["json"]["parse_mode"], "HTML")
        self.assertEqual(call["json"]["chat_id"], CHAT)
        self.assertEqual(call["json"]["text"], "<b>hola</b>")
        self.assertEqual(call["timeout"], config.TELEGRAM.timeout_seconds)

    def test_429_respeta_retry_after(self):
        limitado = FakeResponse(429, {"ok": False, "error_code": 429,
                                      "description": "Too Many Requests: retry after 7",
                                      "parameters": {"retry_after": 7}})
        poster = FakePoster(limitado, ok(5))
        with no_sleep() as sleep:
            self.assertEqual(tg.send_message("hola", TOKEN, CHAT, http_post=poster), 5)
        sleep.assert_called_once_with(7.0)
        self.assertEqual(len(poster.calls), 2)

    def test_429_sin_retry_after_usa_backoff(self):
        poster = FakePoster(FakeResponse(429, {"ok": False}), FakeResponse(429, {"ok": False}), ok())
        with no_sleep() as sleep:
            tg.send_message("hola", TOKEN, CHAT, http_post=poster)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 4.0])

    def test_retry_after_excesivo_no_bloquea(self):
        poster = FakePoster(FakeResponse(429, {"ok": False, "parameters": {"retry_after": 3600}}))
        with no_sleep() as sleep, self.assertRaises(tg.TelegramError) as ctx:
            tg.send_message("hola", TOKEN, CHAT, http_post=poster)
        sleep.assert_not_called()
        self.assertIn("3600", str(ctx.exception))

    def test_5xx_reintenta_con_backoff_exponencial(self):
        poster = FakePoster(FakeResponse(502, {}), FakeResponse(500, {"ok": False}), ok(9))
        with no_sleep() as sleep:
            self.assertEqual(tg.send_message("hola", TOKEN, CHAT, http_post=poster), 9)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 4.0])

    def test_error_de_red_reintenta_y_no_expone_el_token(self):
        boom = ConnectionError(f"HTTPSConnectionPool: url /bot{TOKEN}/sendMessage failed")
        poster = FakePoster(boom, boom, boom)
        with no_sleep(), self.assertLogs(tg.logger, level="WARNING") as logs:
            with self.assertRaises(tg.TelegramError) as ctx:
                tg.send_message("hola", TOKEN, CHAT, http_post=poster)
        self.assertEqual(len(poster.calls), config.TELEGRAM.max_retries)
        self.assertNotIn(TOKEN, str(ctx.exception))
        self.assertNotIn(TOKEN, "\n".join(logs.output))
        self.assertIn("3 intentos", str(ctx.exception))

    def test_4xx_no_reintenta(self):
        poster = FakePoster(FakeResponse(401, {"ok": False, "description": "Unauthorized"}))
        with no_sleep() as sleep, self.assertRaises(tg.TelegramError) as ctx:
            tg.send_message("hola", TOKEN, CHAT, http_post=poster)
        self.assertEqual(len(poster.calls), 1)
        sleep.assert_not_called()
        self.assertIn("401", str(ctx.exception))

    def test_html_rechazado_se_reenvia_como_texto_plano(self):
        rechazo = FakeResponse(400, {"ok": False,
                                     "description": "Bad Request: can't parse entities: ..."})
        poster = FakePoster(rechazo, ok(3))
        with no_sleep(), self.assertLogs(tg.logger, level="WARNING"):
            message_id = tg.send_message("<b>Hola</b> &amp; adiós", TOKEN, CHAT, http_post=poster)
        self.assertEqual(message_id, 3)
        segunda = poster.calls[1]["json"]
        self.assertNotIn("parse_mode", segunda)
        self.assertEqual(segunda["text"], "Hola & adiós")

    def test_200_con_cuerpo_ilegible_cuenta_como_enviado_sin_duplicar(self):
        poster = FakePoster(FakeResponse(200, ValueError("no json")))
        with no_sleep():
            self.assertIsNone(tg.send_message("hola", TOKEN, CHAT, http_post=poster))
        self.assertEqual(len(poster.calls), 1)          # no se reenvía: evitaría duplicados

    def test_200_sin_message_id(self):
        poster = FakePoster(FakeResponse(200, {"ok": True}))
        self.assertIsNone(tg.send_message("hola", TOKEN, CHAT, http_post=poster))


class TestSendReport(unittest.TestCase):
    def test_sin_credenciales(self):
        with self.assertLogs(tg.logger, level="ERROR"):
            result = tg.send_report("hola", env=config.EnvSettings(), http_post=FakePoster())
        self.assertFalse(result.ok)
        self.assertIn("no configurados", result.error)

    def test_varias_partes_en_orden(self):
        poster = FakePoster(ok(1), ok(2), ok(3))
        cfg = replace(config.TELEGRAM, max_message_chars=400)
        texto = "\n\n".join(f"<b>Bloque {i}</b>\n" + "z" * 250 for i in range(3))
        result = tg.send_report(texto, env=ENV, http_post=poster, telegram_cfg=cfg)
        self.assertTrue(result.ok)
        self.assertEqual((result.parts_total, result.parts_sent), (3, 3))
        self.assertEqual(result.message_ids, [1, 2, 3])
        self.assertEqual([c["json"]["text"].split("\n")[0] for c in poster.calls],
                         ["<b>Bloque 0</b>", "<b>Bloque 1</b>", "<b>Bloque 2</b>"])

    def test_fallo_en_una_parte_registra_error_sin_token_y_no_sigue(self):
        boom = ConnectionError(f"fallo en https://api.telegram.org/bot{TOKEN}/sendMessage")
        poster = FakePoster(ok(1), boom, boom, boom, ok(3))
        cfg = replace(config.TELEGRAM, max_message_chars=400)
        texto = "\n\n".join(f"<b>Bloque {i}</b>\n" + "z" * 250 for i in range(3))
        with no_sleep(), self.assertLogs(tg.logger, level="ERROR") as logs:
            result = tg.send_report(texto, env=ENV, http_post=poster, telegram_cfg=cfg)
        self.assertFalse(result.ok)
        self.assertEqual((result.parts_total, result.parts_sent), (3, 1))
        self.assertNotIn(TOKEN, "\n".join(logs.output))
        self.assertNotIn(TOKEN, result.error)
        self.assertIn("parte 2/3", "\n".join(logs.output))
        self.assertEqual(len(poster.calls), 4)          # 1 correcta + 3 intentos; la parte 3 no se envía

    def test_informe_vacio(self):
        with self.assertLogs(tg.logger, level="ERROR"):
            result = tg.send_report("   ", env=ENV, http_post=FakePoster())
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "informe vacío")

    def test_informe_real_de_extremo_a_extremo(self):
        poster = FakePoster()
        html_text = report.markdown_to_telegram_html(report.build_report(make_report_data()))
        result = tg.send_report(html_text, env=ENV, http_post=poster)
        self.assertTrue(result.ok)
        self.assertTrue(all(len(c["json"]["text"]) <= config.TELEGRAM.max_message_chars
                            for c in poster.calls))
        self.assertEqual(result.to_dict()["ok"], True)


if __name__ == "__main__":
    unittest.main()
