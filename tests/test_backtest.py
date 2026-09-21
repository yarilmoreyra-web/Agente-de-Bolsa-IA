"""Tests de backtest.py con barras sintéticas (sin red)."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

import backtest as bt
import history
import market_data as md
import utils
from sample_data import make_candidate, make_pick
from synthetic import make_5m_bars
from utils import NY_TZ

TARGET = date(2026, 9, 21)
TODAY = date(2026, 9, 22)


def session(rows, day="2026-09-21"):
    """Barras de 5 min desde las 09:30 a partir de tuplas (open, high, low, close)."""
    index = pd.date_range(f"{day} 09:30", periods=len(rows), freq="5min", tz=NY_TZ)
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])


def long_pick(**kw):
    """LONG: entrada 99.5-100.5, stop 98, objetivos 102 y 104, límite de entrada 100.8."""
    base = dict(entry_zone_low=99.5, entry_zone_high=100.5, stop=98.0, target_1=102.0,
                target_2=104.0, max_entry_price=100.8, rr=1.5)
    return make_pick("NVDA", **{**base, **kw})


def short_pick(**kw):
    """SHORT: entrada 99.5-100.5, stop 102, objetivos 98 y 96, límite de entrada 99.2."""
    base = dict(direction="SHORT", entry_zone_low=99.5, entry_zone_high=100.5, stop=102.0,
                target_1=98.0, target_2=96.0, max_entry_price=99.2)
    return make_pick("NVDA", **{**base, **kw})


class TestSimulatePick(unittest.TestCase):
    def test_objetivo_antes_que_stop(self):
        bars = session([(100, 100.5, 99.8, 100.2), (100.2, 102.5, 100.1, 102.2),
                        (102.2, 102.4, 97.5, 98.0)])
        res = bt.simulate_pick(long_pick(), bars)
        self.assertEqual(res["outcome"], "objetivo_1")
        self.assertFalse(res["ambiguous"])
        self.assertTrue(res["touched_target_1"])
        self.assertFalse(res["touched_target_2"])
        self.assertTrue(res["touched_stop"])                # toque suelto posterior, informado igual
        self.assertEqual((res["open"], res["high"], res["low"], res["close"]),
                         (100.0, 102.5, 97.5, 98.0))
        self.assertEqual(res["mfe_pct"], 2.5)
        self.assertEqual(res["mae_pct"], -2.5)

    def test_stop_antes_que_objetivo(self):
        bars = session([(100, 100.3, 97.8, 98.0), (98.0, 102.5, 97.9, 102.0)])
        res = bt.simulate_pick(long_pick(), bars)
        self.assertEqual(res["outcome"], "stop")
        self.assertFalse(res["ambiguous"])
        self.assertTrue(res["touched_stop"] and res["touched_target_1"])

    def test_stop_y_objetivo_en_la_misma_barra_es_ambiguo_y_cuenta_como_stop(self):
        res = bt.simulate_pick(long_pick(), session([(100, 102.5, 97.5, 100.0)]))
        self.assertEqual(res["outcome"], "stop")
        self.assertTrue(res["ambiguous"])
        self.assertTrue(any("ambiguo" in n for n in res["notes"]))

    def test_objetivo_2_antes_que_stop(self):
        bars = session([(100, 102.5, 99.9, 102.0), (102.0, 104.5, 101.9, 104.2)])
        res = bt.simulate_pick(long_pick(), bars)
        self.assertEqual(res["outcome"], "objetivo_2")
        self.assertTrue(res["touched_target_2"])

    def test_objetivo_1_y_luego_objetivo_2_y_stop_en_la_misma_barra(self):
        bars = session([(100, 102.5, 99.9, 102.0), (102.0, 104.5, 97.0, 100.0)])
        res = bt.simulate_pick(long_pick(), bars)
        self.assertEqual(res["outcome"], "objetivo_1")      # el objetivo 2 no se cuenta
        self.assertTrue(res["ambiguous"])
        self.assertTrue(res["touched_target_2"])

    def test_sin_resolucion(self):
        res = bt.simulate_pick(long_pick(), session([(100, 101, 99, 100.5), (100.5, 101.5, 99.2, 101)]))
        self.assertEqual(res["outcome"], "sin_resolucion")
        self.assertFalse(res["touched_stop"] or res["touched_target_1"])

    def test_short_espejo(self):
        bars = session([(100, 100.4, 97.5, 98.0), (98.0, 98.5, 95.5, 96.0)])
        res = bt.simulate_pick(short_pick(), bars)
        self.assertEqual(res["outcome"], "objetivo_2")
        self.assertEqual(res["mfe_pct"], 4.5)                # (100 - 95.5) / 100
        self.assertEqual(res["mae_pct"], -0.4)               # (100 - 100.4) / 100
        stop_first = bt.simulate_pick(short_pick(), session([(100, 102.3, 99.8, 101.0),
                                                            (101.0, 101.5, 97.0, 97.5)]))
        self.assertEqual(stop_first["outcome"], "stop")

    def test_reglas_de_entrada_long(self):
        dentro = bt.simulate_pick(long_pick(), session([(100.2, 101, 99, 100)]))
        self.assertTrue(dentro["open_in_entry_range"])
        self.assertTrue(dentro["open_within_entry_limit"])
        self.assertTrue(dentro["entry_valid"])
        alto = bt.simulate_pick(long_pick(), session([(101.0, 102.5, 100.5, 102.0)]))
        self.assertFalse(alto["open_in_entry_range"])
        self.assertFalse(alto["open_within_entry_limit"])      # 101.0 > 100.8
        self.assertFalse(alto["entry_valid"])
        self.assertTrue(any("NO ENTRAR" in n for n in alto["notes"]))
        self.assertEqual(alto["outcome"], "objetivo_1")        # se calcula igualmente
        bajo = bt.simulate_pick(long_pick(), session([(99.0, 100, 98.5, 99.5)]))
        self.assertFalse(bajo["open_in_entry_range"])          # por debajo del rango de entrada
        self.assertTrue(bajo["open_within_entry_limit"])       # pero por debajo del máximo válido
        self.assertTrue(bajo["entry_valid"])

    def test_reglas_de_entrada_short(self):
        self.assertTrue(bt.simulate_pick(short_pick(), session([(100.0, 100.4, 98, 99)]))["entry_valid"])
        self.assertFalse(bt.simulate_pick(short_pick(), session([(99.0, 99.4, 97, 98)]))["entry_valid"])

    def test_sin_limite_usa_el_rango_de_entrada(self):
        pick = long_pick()
        del pick["max_entry_price"]
        res = bt.simulate_pick(pick, session([(100.0, 101, 99, 100)]))
        self.assertEqual(res["open_within_entry_limit"], "N/A")
        self.assertTrue(res["entry_valid"])

    def test_stop_na_no_se_evalua(self):
        res = bt.simulate_pick(long_pick(stop=None), session([(100, 102.5, 90.0, 101)]))
        self.assertFalse(res["touched_stop"])
        self.assertEqual(res["outcome"], "objetivo_1")
        self.assertTrue(any("stop N/A" in n for n in res["notes"]))

    def test_barras_desordenadas(self):
        bars = session([(100, 100.3, 97.8, 98.0), (98.0, 102.5, 97.9, 102.0)]).iloc[::-1]
        self.assertEqual(bt.simulate_pick(long_pick(), bars)["outcome"], "stop")

    def test_casos_sin_datos(self):
        vacio = bt.simulate_pick(long_pick(), pd.DataFrame())
        self.assertEqual(vacio["outcome"], "N/A")
        self.assertEqual(vacio["open"], "N/A")
        self.assertTrue(any("sin barras" in n for n in vacio["notes"]))
        invalido = bt.simulate_pick(make_pick("NVDA", direction="??"), session([(100, 101, 99, 100)]))
        self.assertEqual(invalido["outcome"], "N/A")
        self.assertIn("dirección del pick no válida", invalido["notes"])


class FakeYF:
    """yfinance simulado: devuelve barras 5m de varios días para los tickers pedidos."""

    def __init__(self, frame_for, error=None):
        self.frame_for, self.error, self.calls = frame_for, error, []

    def download(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return pd.concat({t: self.frame_for(t) for t in kwargs["tickers"]}, axis=1)


def full_day(day):
    return make_5m_bars(day, "09:30", "16:00", price=100.0, slope=0.01)


class TestLoadSessionBars(unittest.TestCase):
    def test_usa_barras_guardadas(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(md, "_get_yf", side_effect=AssertionError("no debe descargar")):
            md.save_intraday_bars({"NVDA": full_day("2026-09-21")}, TARGET, tmp)
            bars, source, note = bt.load_session_bars("NVDA", TARGET, Path(tmp), TODAY)
        self.assertEqual(source, bt.SOURCE_SAVED)
        self.assertEqual(note, "")
        self.assertEqual((bars.index[0].hour, bars.index[0].minute), (9, 30))
        self.assertEqual(len(bars), 78)                       # 09:30-16:00 en barras de 5 min

    def test_guardadas_solo_premarket_cae_a_yfinance_y_lo_indica(self):
        fake = FakeYF(lambda t: pd.concat([full_day("2026-09-18"), full_day("2026-09-21")]))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md, "_get_yf", return_value=fake):
            md.save_intraday_bars({"NVDA": make_5m_bars("2026-09-21", "04:00", "08:45")}, TARGET, tmp)
            bars, source, note = bt.load_session_bars("NVDA", TARGET, Path(tmp), TODAY)
        self.assertEqual(source, bt.SOURCE_YFINANCE)
        self.assertIn("solo cubren el pre-market", note)
        self.assertIn("yfinance", note)
        self.assertTrue(all(ts.date() == TARGET for ts in bars.index))
        self.assertEqual(fake.calls[0]["period"], "60d")
        self.assertFalse(fake.calls[0]["prepost"])
        self.assertEqual(fake.calls[0]["interval"], "5m")

    def test_sin_barras_guardadas_cae_a_yfinance(self):
        fake = FakeYF(lambda t: full_day("2026-09-21"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md, "_get_yf", return_value=fake):
            _, source, note = bt.load_session_bars("NVDA", TARGET, Path(tmp), TODAY)
        self.assertEqual(source, bt.SOURCE_YFINANCE)
        self.assertIn("no hay barras guardadas", note)

    def test_fecha_fuera_del_rango_de_yfinance(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(md, "_get_yf", side_effect=AssertionError("no debe descargar")):
            bars, source, note = bt.load_session_bars("NVDA", date(2026, 5, 1), Path(tmp), TODAY)
        self.assertTrue(bars.empty)
        self.assertEqual(source, "N/A")
        self.assertIn("fuera del rango de yfinance", note)

    def test_yfinance_sin_esa_fecha_o_caido(self):
        fake = FakeYF(lambda t: full_day("2026-09-18"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md, "_get_yf", return_value=fake):
            bars, source, note = bt.load_session_bars("NVDA", TARGET, Path(tmp), TODAY)
        self.assertEqual((source, len(bars)), ("N/A", 0))
        self.assertIn("no devolvió barras de esa fecha", note)

        caido = FakeYF(lambda t: None, error=ConnectionError("sin red"))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(md, "_get_yf", return_value=caido), \
                mock.patch.object(utils.time_module, "sleep"), \
                self.assertLogs(md.logger, level="ERROR"):
            _, source, _ = bt.load_session_bars("NVDA", TARGET, Path(tmp), TODAY)
        self.assertEqual(source, "N/A")

    def test_descarga_desactivada_y_fecha_futura(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, source, note = bt.load_session_bars("NVDA", TARGET, Path(tmp), TODAY, download=False)
            self.assertEqual(source, "N/A")
            self.assertIn("descarga está desactivada", note)
            _, source, note = bt.load_session_bars("NVDA", date(2026, 10, 30), Path(tmp), TODAY)
            self.assertEqual(source, "N/A")
            self.assertIn("futura", note)


class Env:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.history_dir, self.intraday_dir = root / "history", root / "intraday"
        self.backtests_dir, self.summary = root / "backtests", root / "backtest_summary.csv"
        self.history_csv = root / "history.csv"
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()

    def save_history(self, picks):
        record = history.HistoryRecord(
            date=TARGET, candidates=[make_candidate(p["ticker"]) for p in picks],
            final_selection=picks)
        history.save_history(record, self.history_dir, self.history_csv)

    def run(self, **kw):
        return bt.run_backtest(TARGET, self.history_dir, self.intraday_dir, self.backtests_dir,
                               self.summary, today=TODAY, **kw)

    def summary_rows(self):
        with open(self.summary, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


class TestRunBacktest(unittest.TestCase):
    def _bars(self, kind):
        if kind == "target":
            rows = [(100, 100.5, 99.8, 100.2), (100.2, 102.5, 100.1, 102.2), (102.2, 102.4, 101.0, 101.5)]
        else:
            rows = [(100, 100.3, 97.8, 98.0), (98.0, 99.0, 97.0, 98.5)]
        df = session(rows)
        return df

    def test_flujo_completo_json_y_csv_sin_duplicados(self):
        picks = [long_pick(), {**long_pick(), "ticker": "AMD"}]
        with Env() as env:
            env.save_history(picks)
            md.save_intraday_bars({"NVDA": self._bars("target"), "AMD": self._bars("stop")},
                                  TARGET, env.intraday_dir)
            payload = env.run()
            self.assertTrue(payload["ok"])
            saved = json.loads((env.backtests_dir / "2026-09-21.json").read_text(encoding="utf-8"))
            rows = env.summary_rows()
            env.run()                                          # reejecutar: sin duplicados
            rows_again = env.summary_rows()
        self.assertEqual([r["ticker"] for r in saved["results"]], ["NVDA", "AMD"])
        self.assertEqual(saved["results"][0]["outcome"], "objetivo_1")
        self.assertEqual(saved["results"][1]["outcome"], "stop")
        self.assertEqual(saved["summary"]["outcomes"], {"objetivo_1": 1, "stop": 1})
        self.assertEqual(saved["results"][0]["data_source"], bt.SOURCE_SAVED)
        self.assertEqual(list(rows[0].keys()), bt.BACKTEST_COLUMNS)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows_again), 2)
        self.assertEqual({r["ticker"]: r["outcome"] for r in rows}, {"NVDA": "objetivo_1", "AMD": "stop"})
        self.assertEqual(rows[0]["date"], "2026-09-21")

    def test_sin_historico_no_escribe_nada(self):
        with Env() as env, self.assertLogs(bt.logger, level="ERROR"):
            payload = env.run()
            self.assertFalse(payload["ok"])
            self.assertIn("No hay histórico", payload["error"])
            self.assertFalse(env.backtests_dir.exists())
            self.assertFalse(env.summary.exists())

    def test_sin_barras_se_registra_pero_no_rompe(self):
        with Env() as env:
            env.save_history([long_pick()])
            payload = env.run(download=False)
            row = env.summary_rows()[0]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["results"][0]["data_source"], "N/A")
        self.assertEqual(payload["results"][0]["outcome"], "N/A")
        self.assertIn("no hay barras guardadas", row["notes"])

    def test_un_pick_defectuoso_no_detiene_a_los_demas(self):
        picks = [long_pick(), {**long_pick(), "ticker": "AMD"}]
        real = bt.simulate_pick
        with Env() as env:
            env.save_history(picks)
            md.save_intraday_bars({"NVDA": self._bars("target"), "AMD": self._bars("stop")},
                                  TARGET, env.intraday_dir)
            with mock.patch.object(bt, "simulate_pick", side_effect=[RuntimeError("boom"), real(picks[1], self._bars("stop"))]), \
                    self.assertLogs(bt.logger, level="ERROR"):
                payload = env.run()
        self.assertEqual(len(payload["results"]), 2)
        self.assertIn("error: RuntimeError", payload["results"][0]["notes"][0])
        self.assertEqual(payload["results"][1]["outcome"], "stop")

    def test_sin_picks(self):
        with Env() as env:
            env.save_history([])
            payload = env.run()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["summary"]["picks"], 0)
            self.assertEqual(env.summary_rows(), [])

    def test_reejecucion_sin_picks_borra_las_filas_de_esa_fecha(self):
        with Env() as env:
            env.save_history([long_pick()])
            md.save_intraday_bars({"NVDA": self._bars("target")}, TARGET, env.intraday_dir)
            env.run()
            self.assertEqual(len(env.summary_rows()), 1)
            env.save_history([])
            env.run()
            self.assertEqual(env.summary_rows(), [])


class TestCommand(unittest.TestCase):
    def test_fecha_invalida(self):
        with self.assertLogs(bt.logger, level="ERROR"):
            self.assertEqual(bt.run_backtest_command("21/09/2026"), 2)

    def test_historico_inexistente_devuelve_1(self):
        with self.assertLogs(bt.logger, level="ERROR"), mock.patch("builtins.print"):
            self.assertEqual(bt.run_backtest_command("2000-01-03"), 1)

    def test_formato_del_resumen(self):
        text = bt.format_summary({"ok": True, "date": "2026-09-21", "summary": {"picks": 1},
                                  "results": [{"ticker": "NVDA", "direction": "LONG",
                                               "outcome": "stop", "ambiguous": True,
                                               "entry_valid": True, "mfe_pct": 1.0,
                                               "mae_pct": -2.0, "data_source": "yfinance 5m"}]})
        self.assertIn("NVDA LONG: resultado=stop (ambiguo)", text)
        self.assertEqual(bt.format_summary({"ok": False, "error": "sin histórico"}), "sin histórico")


if __name__ == "__main__":
    unittest.main()
