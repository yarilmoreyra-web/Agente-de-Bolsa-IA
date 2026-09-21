"""Tests de report.py: contenido, reglas del informe y conversión a HTML de Telegram."""
from __future__ import annotations

import re
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import report
from html_utils import is_balanced
from sample_data import (
    SENT_LATE, SENT_ON_TIME, SESSION, make_candidate, make_context, make_pick,
    make_report_data,
)
from utils import NY_TZ


class TestFormatting(unittest.TestCase):
    def test_formatos_numericos(self):
        self.assertEqual(report.fmt_num(1234.5), "1,234.50")
        self.assertEqual(report.fmt_num("N/A"), "N/A")
        self.assertEqual(report.fmt_num(float("nan")), "N/A")
        self.assertEqual(report.fmt_pct(3.2), "+3.20%")
        self.assertEqual(report.fmt_pct(-1.234), "-1.23%")
        self.assertEqual(report.fmt_pct(12.5, 1, signed=False), "12.5%")
        self.assertEqual(report.fmt_volume(1_250_000), "1.25M")
        self.assertEqual(report.fmt_volume(850_000), "850K")
        self.assertEqual(report.fmt_volume(420), "420")
        self.assertEqual(report.fmt_volume(None), "N/A")
        self.assertEqual(report.fmt_rvol(3.4), "3.40x")
        self.assertEqual(report.fmt_pp(-0.5), "-0.50 pp")

    def test_fecha_en_espanol(self):
        self.assertEqual(report.spanish_date(date(2026, 9, 21)), "lunes 21 de septiembre de 2026")
        self.assertEqual(report.spanish_date(date(2026, 1, 3)), "sábado 3 de enero de 2026")

    def test_safe_text_neutraliza_markdown_y_saltos(self):
        text = report.safe_text("linea1\n\nlinea2 **negrita** `code` [x](https://malo.test)")
        self.assertNotIn("\n", text)
        self.assertNotIn("**", text)
        self.assertNotIn("](", text)
        self.assertEqual(report.safe_text(None), "N/A")
        self.assertEqual(report.safe_text("   "), "N/A")

    def test_informe_tardio(self):
        self.assertFalse(report.is_late(SENT_ON_TIME))
        self.assertTrue(report.is_late(SENT_LATE))
        self.assertFalse(report.is_late(None))
        self.assertFalse(report.is_late(datetime(2026, 9, 21, 9, 0, tzinfo=NY_TZ)))   # justo a la hora


class TestBuildReport(unittest.TestCase):
    def test_secciones_en_orden(self):
        md = report.build_report(make_report_data())
        marcas = ["📅 **lunes 21 de septiembre de 2026** (sesión normal)", "🕒 Snapshot 08:45 ET",
                  "📊 **CONTEXTO DE MERCADO**", "📋 Universo: 40 tickers",
                  "🔎 **CANDIDATAS ANALIZADAS** (3, con puntuación)", "🥇 **NVDA**",
                  "🥈 **AMD**", "🔔 Regla de apertura", "⚠️ **RIESGOS**", report.DISCLAIMER]
        posiciones = [md.find(m) for m in marcas]
        self.assertTrue(all(p >= 0 for p in posiciones), dict(zip(marcas, posiciones)))
        self.assertEqual(posiciones, sorted(posiciones))
        self.assertNotIn("🚫", md)
        self.assertTrue(md.rstrip().endswith("no es asesoramiento financiero."))

    def test_campos_del_pick_vienen_de_python(self):
        md = report.build_report(make_report_data())
        for esperado in (
            "Resultados por encima de lo esperado / Reuters",
            "**Cierre anterior / Pre-market / Gap:** 100.00 / 103.20 / +3.20%",
            "**Volumen / RVOL:** 1.25M / 3.40x",
            "**RSI / SMA20 / SMA50 / ATR:** 62.1 / 98.20 / 95.10 / 2.10",
            "**Soporte / Resistencia:** SMA20 (98.20) / máx. 20 sesiones (106.00)",
            "vs SPY +2.50 pp · vs QQQ +2.10 pp",
            "**Riesgo gap & fade:** MEDIUM",
            "**Entrada prevista:** 103.00 – 103.60",
            "**Stop / Objetivo 1 / Objetivo 2 / R/R (Python):** 101.50 / 105.50 / 108.00 / 2.10",
            "**Precio máximo de entrada válido:** 104.20 (por encima: NO ENTRAR)",
            "**Hora objetivo / Invalidación:** 10:30 ET / pierde 103.0 con volumen",
            "**Decisión / Razón:** BUY (confianza HIGH)",
        ):
            self.assertIn(esperado, md)

    def test_entrada_prevista_no_es_entrada_confirmada(self):
        md = report.build_report(make_report_data())
        self.assertIn("La entrada prevista se basa en el pre-market", md)
        self.assertIn("la entrada confirmada depende del precio real de apertura", md)
        self.assertIn("Regla de apertura: si el precio de apertura supera el precio máximo de "
                      "entrada válido, o invalida el R/R: NO ENTRAR.", md)

    def test_marca_de_informe_tardio(self):
        self.assertNotIn(report.LATE_MARK, report.build_report(make_report_data()))
        tarde = report.build_report(make_report_data(sent_at=SENT_LATE))
        self.assertIn("Envío 09:07 ET · ⏱ Informe tardío", tarde)
        sin_envio = report.build_report(make_report_data(sent_at=None))
        self.assertNotIn("Envío", sin_envio)

    def test_cierre_anticipado_en_la_cabecera(self):
        md = report.build_report(make_report_data(session_type="cierre anticipado"))
        self.assertIn("(cierre anticipado)", md)

    def test_calidad_del_dato_solo_si_no_es_ok(self):
        ok = report.build_report(make_report_data())
        self.assertNotIn("Calidad del dato", ok)
        self.assertNotIn(" ⚠", ok.split("⚠️ **RIESGOS**")[0])
        data = make_report_data()
        data.candidates[0] = make_candidate("NVDA", 82.5, quality="stale",
                                            quality_note="última barra hace 105 min")
        data.candidates[1] = make_candidate("AMD", 74.0, quality="sparse", quality_note="solo 2 barras")
        md = report.build_report(data)
        self.assertIn("NVDA 82.5 ⚠", md)
        self.assertIn("⚠ Calidad del dato pre-market: dato desactualizado (última barra hace 105 min)", md)
        self.assertIn("⚠ Calidad del dato pre-market: volumen parcial (solo 2 barras)", md)
        self.assertIn("NVDA: dato desactualizado", md.split("⚠️ **RIESGOS**")[1])

    def test_no_fuerza_tres_candidatas(self):
        data = make_report_data()
        data.gemini["picks"] = [make_pick("NVDA")]
        md = report.build_report(data)
        self.assertIn("🥇", md)
        self.assertNotIn("🥈", md)
        self.assertNotIn("🥉", md)

    def test_maximo_de_picks(self):
        data = make_report_data()
        tickers = ["NVDA", "AMD", "TSLA", "AAPL", "MSFT"]
        data.candidates = [make_candidate(t, 80 - i) for i, t in enumerate(tickers)]
        data.gemini["picks"] = [make_pick(t) for t in tickers]
        md = report.build_report(data)
        self.assertEqual(sum(md.count(m) for m in report.MEDALS), 3)
        self.assertNotIn("**AAPL**", md)

    def test_pick_con_ticker_inventado_se_ignora(self):
        data = make_report_data()
        data.gemini["picks"] = [make_pick("ZZZZ"), make_pick("NVDA")]
        md = report.build_report(data)
        self.assertNotIn("ZZZZ", md)
        self.assertIn("**NVDA**", md)

    def test_pick_short_usa_limite_minimo(self):
        data = make_report_data()
        data.gemini["picks"] = [make_pick("NVDA", direction="SHORT")]
        md = report.build_report(data)
        self.assertIn("**Precio mínimo de entrada válido:** 104.20 (por debajo: NO ENTRAR)", md)

    def test_datos_faltantes_se_muestran_como_na(self):
        data = make_report_data()
        data.gemini["picks"] = [{"ticker": "NVDA", "decision": "WAIT"}]
        data.context = None
        md = report.build_report(data)
        self.assertIn("**Stop / Objetivo 1 / Objetivo 2 / R/R (Python):** N/A / N/A / N/A / N/A", md)
        self.assertIn("**Precio máximo de entrada válido:** N/A", md)
        self.assertIn("Contexto de mercado no disponible (N/A)", md)
        self.assertIn("Ninguna candidata está en BUY", md)

    def test_gemini_no_trade(self):
        data = make_report_data()
        data.gemini.update(no_trade=True, no_trade_reason="Mercado en contra y gaps sin volumen",
                           picks=[])
        md = report.build_report(data)
        self.assertIn("🚫 **NO OPERAR**\n• Mercado en contra y gaps sin volumen", md)
        self.assertFalse(any(m in md for m in report.MEDALS))
        self.assertNotIn("Regla de apertura", md)

    def test_ia_no_disponible_solo_datos_de_python(self):
        md = report.build_report(make_report_data(gemini=None))
        self.assertIn("Análisis IA no disponible", md)
        self.assertIn("solo datos de Python", md)
        self.assertNotIn("BUY (", md)
        self.assertNotIn("Entrada prevista", md)
        self.assertIn("🚫 **NO OPERAR**\n• Análisis IA no disponible: no se emite selección BUY.", md)
        self.assertIn("**NVDA**", md)
        self.assertIn(report.DISCLAIMER, md)
        caido = report.build_report(make_report_data(gemini={"available": False, "picks": []}))
        self.assertIn("Análisis IA no disponible", caido)

    def test_sin_candidatas_del_filtro(self):
        md = report.build_report(make_report_data(
            candidates=[], gemini=None, filter_no_trade_reason="Ninguna de las 40 acciones superó las puertas duras"))
        self.assertIn("(0, con puntuación)", md)
        self.assertIn("🚫 **NO OPERAR**\n• Ninguna de las 40 acciones superó las puertas duras", md)
        vacio = report.build_report(make_report_data(candidates=[], gemini=None))
        self.assertIn("Ninguna candidata superó el filtro", vacio)

    def test_riesgo_alto_de_regimen_y_fade_alto(self):
        data = make_report_data()
        ctx = make_context()
        ctx["regime"] = {"risk_level": "alto", "summary": "Riesgo alto: VIX alto (31.0 ≥ 25)"}
        data.context = ctx
        data.gemini["picks"] = [make_pick("NVDA", fade_risk="HIGH")]
        md = report.build_report(data)
        riesgos = md.split("⚠️ **RIESGOS**")[1]
        self.assertIn("Riesgo alto: VIX alto", riesgos)
        self.assertIn("NVDA: riesgo alto de gap-and-fade", riesgos)

    def test_avisos_de_validacion_por_pick(self):
        data = make_report_data()
        data.gemini["picks"] = [make_pick("NVDA", warnings=["stop ajustado a la tabla de niveles"])]
        self.assertIn("⚠ Aviso: stop ajustado a la tabla de niveles", report.build_report(data))

    def test_bloques_separados_por_linea_en_blanco_sin_blancos_internos(self):
        md = report.build_report(make_report_data())
        for bloque in md.strip().split("\n\n"):
            self.assertNotIn("\n\n", bloque)
        pick_blocks = [b for b in md.split("\n\n") if b.startswith(("🥇", "🥈"))]
        self.assertEqual(len(pick_blocks), 2)

    def test_link_de_fuente_si_es_url(self):
        data = make_report_data()
        data.gemini["picks"] = [make_pick("NVDA", catalyst_source="https://x.test/a?b=1")]
        self.assertIn("[fuente](https://x.test/a?b=1)", report.build_report(data))

    def test_guardar_informe(self):
        md = report.build_report(make_report_data())
        with tempfile.TemporaryDirectory() as tmp:
            path = report.save_report(md, SESSION, Path(tmp))
            self.assertEqual(path.name, "2026-09-21.md")
            self.assertEqual(path.read_text(encoding="utf-8"), md)


class TestTelegramHtml(unittest.TestCase):
    def test_escapa_html_y_ampersand(self):
        out = report.markdown_to_telegram_html("a < b & c > d <script>alert(1)</script>")
        self.assertEqual(out, "a &lt; b &amp; c &gt; d &lt;script&gt;alert(1)&lt;/script&gt;")

    def test_negrita_codigo_y_enlace(self):
        out = report.markdown_to_telegram_html(
            "**Hola** `x < y` [fuente](https://x.test/a?b=1&c=2)")
        self.assertEqual(out, '<b>Hola</b> <code>x &lt; y</code> '
                              '<a href="https://x.test/a?b=1&amp;c=2">fuente</a>')

    def test_solo_enlaces_http(self):
        out = report.markdown_to_telegram_html("[malo](javascript:alert(1)) [ok](http://x.test)")
        self.assertNotIn("<a href=\"javascript", out)
        self.assertIn('<a href="http://x.test">ok</a>', out)

    def test_marcas_sin_cerrar_quedan_literales(self):
        out = report.markdown_to_telegram_html("**sin cerrar y `tampoco")
        self.assertEqual(out, "**sin cerrar y `tampoco")

    def test_las_etiquetas_no_cruzan_lineas(self):
        out = report.markdown_to_telegram_html("**uno\ndos**")
        self.assertNotIn("<b>", out)

    def test_informe_completo_produce_html_equilibrado_y_sin_etiquetas_ajenas(self):
        data = make_report_data()
        data.gemini["picks"][0]["reason"] = "<img src=x onerror=alert(1)> & más"
        out = report.markdown_to_telegram_html(report.build_report(data))
        self.assertTrue(is_balanced(out))
        self.assertNotIn("<img", out)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt; &amp; más", out)
        etiquetas = set(re.findall(r"</?([a-z]+)", out))
        self.assertTrue(etiquetas <= {"b", "a", "code"}, etiquetas)


if __name__ == "__main__":
    unittest.main()
