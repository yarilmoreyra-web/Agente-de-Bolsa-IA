"""Tests de history.py: JSON diario, CSV sin duplicados y accesores."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import history
from sample_data import SESSION, make_candidate, make_context, make_pick
from utils import already_sent


def make_record(**overrides) -> history.HistoryRecord:
    base = dict(
        date=SESSION, universe={"count": 40, "source": "lista_tickers.xlsx"},
        context=make_context(),
        candidates=[make_candidate("NVDA", 82.5), make_candidate("AMD", 74.0),
                    make_candidate("TSLA", 61.0)],
        filter_log=[{"ticker": "XYZ", "status": "excluida_puerta_dura", "reasons": ["gap"]}],
        gemini_raw={"picks": [{"ticker": "NVDA"}]},
        gemini_validated={"picks": [make_pick("NVDA")], "warnings": []},
        final_selection=[make_pick("NVDA"), make_pick("AMD", decision="WAIT")],
        timestamps={"snapshot": "2026-09-21T08:45:00-04:00"},
        sources={"prices": "yfinance", "news": ["yahoo"]},
    )
    base.update(overrides)
    return history.HistoryRecord(**base)


class Dirs:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.history_dir, self.csv_path = root / "history", root / "history.csv"
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()

    def save(self, record):
        return history.save_history(record, self.history_dir, self.csv_path)

    def rows(self):
        with open(self.csv_path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


class TestAccessors(unittest.TestCase):
    def test_pick_field_con_alias(self):
        self.assertEqual(history.pick_field({"rr": 2.1}, "rr"), 2.1)
        self.assertEqual(history.pick_field({"risk_reward": 1.8}, "rr"), 1.8)
        self.assertEqual(history.pick_field({"max_entry": 104.2}, "max_entry_price"), 104.2)
        self.assertEqual(history.pick_field({}, "rr"), "N/A")
        self.assertEqual(history.pick_field({"stop": 1.0}, "stop"), 1.0)

    def test_analysis_record_y_score(self):
        cand = make_candidate("NVDA", 82.5)
        self.assertIs(history.analysis_record(cand), cand["record"])
        self.assertIs(history.analysis_record(cand["record"]), cand["record"])   # ya es el registro
        self.assertEqual(history.analysis_record("basura"), {})
        self.assertEqual(history.candidate_score(cand), 82.5)
        self.assertEqual(history.candidate_score({"score_detail": {"total": 40}}), 40.0)
        self.assertEqual(history.candidate_score({}), "N/A")

    def test_find_level(self):
        record = make_candidate()["record"]
        self.assertEqual(history.find_level(record, "sma20")["price"], 98.2)
        self.assertIsNone(history.find_level(record, "nope"))
        self.assertIsNone(history.find_level(record, "N/A"))

    def test_to_plain_convierte_dataclasses_y_to_dict(self):
        @dataclass
        class Inner:
            x: int

        class WithDict:
            def to_dict(self):
                return {"y": Inner(2)}

        plain = history.to_plain({"a": Inner(1), "b": [WithDict()], "c": (1, 2)})
        self.assertEqual(plain, {"a": {"x": 1}, "b": [{"y": {"x": 2}}], "c": [1, 2]})


class TestSaveHistory(unittest.TestCase):
    def test_json_con_las_claves_del_historico(self):
        with Dirs() as d:
            path = d.save(make_record())
            self.assertEqual(path.name, "2026-09-21.json")
            data = json.loads(path.read_text(encoding="utf-8"))
        for clave in ("date", "universe", "market_data", "indicators", "catalysts", "filter_log",
                      "candidates", "gemini_raw", "gemini_validated", "final_selection",
                      "timestamps", "sources", "sent", "send_status", "no_trade"):
            self.assertIn(clave, data)
        self.assertEqual(data["date"], "2026-09-21")
        self.assertEqual(data["indicators"]["NVDA"]["rsi14"], 62.14)
        self.assertEqual(data["catalysts"]["AMD"]["catalyst_text"],
                         "Resultados (earnings): supera estimaciones")
        self.assertEqual(data["market_data"]["premarket"]["TSLA"]["premarket_price"], 103.2)
        self.assertEqual(data["market_data"]["context"]["regime"]["risk_level"], "normal")
        self.assertEqual(data["sent"], False)
        self.assertEqual(data["run_count"], 1)

    def test_nan_y_dataclasses_se_serializan(self):
        @dataclass
        class Noticias:
            ticker: str
            catalyst_confirmed: bool

        cand = make_candidate("NVDA")
        cand["news"] = Noticias("NVDA", True)
        cand["record"]["gap_pct"] = float("nan")
        with Dirs() as d:
            path = d.save(make_record(candidates=[cand]))
            data = json.loads(path.read_text(encoding="utf-8"))     # JSON válido (sin NaN)
        self.assertIsNone(data["candidates"][0]["record"]["gap_pct"])
        self.assertEqual(data["candidates"][0]["news"], {"ticker": "NVDA", "catalyst_confirmed": True})

    def test_reejecutar_la_fecha_sobrescribe_e_incrementa_run_count(self):
        with Dirs() as d:
            d.save(make_record())
            path = d.save(make_record(no_trade=True, no_trade_reason="cambio"))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["run_count"], 2)
            self.assertEqual(data["no_trade_reason"], "cambio")
            self.assertEqual(len(list(d.history_dir.glob("*.json"))), 1)

    def test_idempotencia_con_utils_already_sent(self):
        with Dirs() as d:
            d.save(make_record())
            self.assertFalse(already_sent(d.history_dir, SESSION))
            self.assertTrue(history.update_send_status(
                SESSION, True, {"parts_sent": 2}, d.history_dir, d.csv_path))
            self.assertTrue(already_sent(d.history_dir, SESSION))
            data = history.load_history(SESSION, d.history_dir)
            self.assertEqual(data["send_status"], {"parts_sent": 2})
            self.assertIn("sent", data["timestamps"])
            self.assertTrue(all(r["sent"] == "true" for r in d.rows()))

    def test_envio_fallido_no_marca_sent(self):
        with Dirs() as d:
            d.save(make_record())
            history.update_send_status(SESSION, False, {"error": "HTTP 401"}, d.history_dir, d.csv_path)
            self.assertFalse(already_sent(d.history_dir, SESSION))
            self.assertNotIn("sent", history.load_history(SESSION, d.history_dir)["timestamps"])

    def test_update_de_fecha_inexistente(self):
        with Dirs() as d, self.assertLogs(history.logger, level="WARNING"):
            self.assertFalse(history.update_send_status(date(2000, 1, 1), True,
                                                        history_dir=d.history_dir,
                                                        csv_path=d.csv_path))

    def test_load_history_inexistente_o_corrupto(self):
        with Dirs() as d:
            self.assertIsNone(history.load_history(SESSION, d.history_dir))
            d.history_dir.mkdir(parents=True)
            (d.history_dir / "2026-09-21.json").write_text("{corrupto", encoding="utf-8")
            with self.assertLogs("trading_agent.utils", level="WARNING"):
                self.assertIsNone(history.load_history(SESSION, d.history_dir))


class TestCsv(unittest.TestCase):
    def test_una_fila_por_candidata_con_columnas_planas(self):
        with Dirs() as d:
            d.save(make_record())
            rows = d.rows()
            with open(d.csv_path, encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
        self.assertEqual(header, history.CSV_COLUMNS)
        self.assertEqual([r["ticker"] for r in rows], ["NVDA", "AMD", "TSLA"])
        nvda, amd, tsla = rows
        self.assertEqual((nvda["date"], nvda["rank"], nvda["score"]), ("2026-09-21", "1", "82.5"))
        self.assertEqual((nvda["selected"], nvda["decision"], nvda["rr"], nvda["max_entry_price"]),
                         ("true", "BUY", "2.1", "104.2"))
        self.assertEqual((amd["selected"], amd["decision"]), ("true", "WAIT"))
        self.assertEqual((tsla["selected"], tsla["decision"], tsla["stop"], tsla["rr"]),
                         ("false", "N/A", "N/A", "N/A"))
        self.assertEqual(nvda["gap_pct"], "3.2")
        self.assertEqual(nvda["rvol"], "3.4")
        self.assertEqual(nvda["premarket_quality"], "ok")
        self.assertEqual(nvda["fade_flags"], "gap_extendido")
        self.assertEqual(nvda["catalyst_confirmed"], "true")
        self.assertEqual(nvda["rs_pm_vs_spy"], "2.5")
        self.assertEqual(nvda["sent"], "false")

    def test_reejecutar_la_misma_fecha_no_duplica(self):
        with Dirs() as d:
            d.save(make_record())
            d.save(make_record(candidates=[make_candidate("NVDA", 90.0),
                                           make_candidate("AMD", 74.0)]))
            rows = d.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["ticker"]: r["score"] for r in rows}, {"NVDA": "90.0", "AMD": "74.0"})

    def test_otra_fecha_se_anade(self):
        with Dirs() as d:
            d.save(make_record())
            d.save(make_record(date=date(2026, 9, 22)))
            rows = d.rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted({r["date"] for r in rows}), ["2026-09-21", "2026-09-22"])

    def test_conserva_columnas_extra_de_un_archivo_previo(self):
        with Dirs() as d:
            d.csv_path.parent.mkdir(parents=True, exist_ok=True)
            d.csv_path.write_text("date,ticker,nota\n2026-09-18,OLD,mi nota\n", encoding="utf-8")
            d.save(make_record())
            rows = d.rows()
        old = [r for r in rows if r["ticker"] == "OLD"][0]
        self.assertEqual(old["nota"], "mi nota")
        self.assertEqual(old["score"], "")
        self.assertEqual(len(rows), 4)

    def test_sin_candidatas_no_escribe_filas_de_esa_fecha(self):
        with Dirs() as d:
            d.save(make_record())
            d.save(make_record(candidates=[], final_selection=[]))
            with open(d.csv_path, encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), ",".join(history.CSV_COLUMNS))


if __name__ == "__main__":
    unittest.main()
