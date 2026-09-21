"""Tests de market_data.py con un yfinance simulado (sin red)."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

import market_data as md
import config
import utils
from utils import NY_TZ
from synthetic import make_5m_bars, make_daily

SESSION = date(2026, 9, 21)
SNAP_TS = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
CFG = replace(config.MARKET_DATA, batch_size=30, max_retries=3, retry_base_delay=2.0)


class FakeYF:
    """Sustituto de ``yfinance`` que devuelve columnas MultiIndex (ticker, campo)."""

    def __init__(self, frame_for, fail_first=0, always_fail_if=None, empty_for=()):
        self.frame_for = frame_for
        self.fail_first = fail_first
        self.always_fail_if = always_fail_if
        self.empty_for = set(empty_for)
        self.calls: list[dict] = []

    def download(self, **kwargs):
        self.calls.append(kwargs)
        tickers = kwargs["tickers"]
        if self.always_fail_if and self.always_fail_if in tickers:
            raise ConnectionError("bloqueado")
        if len(self.calls) <= self.fail_first:
            raise ConnectionError("timeout simulado")
        frames = {t: self.frame_for(t) for t in tickers if t not in self.empty_for}
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _patch_yf(fake: FakeYF):
    return mock.patch.object(md, "_get_yf", return_value=fake)


def _no_sleep():
    """Evita esperas reales en ``utils.retry`` y permite inspeccionarlas."""
    return mock.patch.object(utils.time_module, "sleep")


class TestIndexAndSplit(unittest.TestCase):
    def test_to_ny_index_desde_utc_y_naive(self):
        utc = pd.DataFrame({"Close": [1.0]},
                           index=pd.DatetimeIndex(["2026-09-21 12:40:00"], tz="UTC"))
        out = md.to_ny_index(utc)
        self.assertEqual(str(out.index.tz), "America/New_York")
        self.assertEqual((out.index[0].hour, out.index[0].minute), (8, 40))
        naive = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2026-09-18"]))
        self.assertEqual(str(md.to_ny_index(naive).index.tz), "America/New_York")
        self.assertTrue(md.to_ny_index(None).empty)

    def test_split_multiindex_en_ambos_ordenes_y_formato_plano(self):
        a, b = make_daily("2026-09-18", 5), make_daily("2026-09-18", 5, start_price=50)
        por_ticker = pd.concat({"AAA": a, "BBB": b}, axis=1)
        for raw in (por_ticker, por_ticker.swaplevel(axis=1)):
            frames = md._split_download(raw, ["AAA", "BBB", "CCC"])
            self.assertEqual(set(frames), {"AAA", "BBB"})
            self.assertEqual(float(frames["BBB"]["Close"].iloc[0]), 50.0)
        plano = md._split_download(a, ["AAA"])
        self.assertEqual(set(plano), {"AAA"})

    def test_normalize_descarta_filas_sin_cierre_y_duplicados(self):
        df = make_daily("2026-09-18", 5)
        df.iloc[2, df.columns.get_loc("Close")] = float("nan")
        df = pd.concat([df, df.iloc[[-1]]])
        out = md._normalize_frame(df)
        self.assertEqual(len(out), 4)
        self.assertTrue(out.index.is_unique)


class TestDownloads(unittest.TestCase):
    def test_lotes_y_parametros(self):
        fake = FakeYF(lambda t: make_daily("2026-09-18", 30))
        tickers = [f"T{i:02d}" for i in range(70)]
        with _patch_yf(fake), _no_sleep():
            res = md.download_daily(tickers, CFG)
        self.assertEqual([len(c["tickers"]) for c in fake.calls], [30, 30, 10])
        self.assertEqual(len(res.data), 70)
        self.assertEqual(res.failed, {})
        call = fake.calls[0]
        self.assertEqual(call["group_by"], "ticker")
        self.assertEqual(call["interval"], "1d")
        self.assertFalse(call["prepost"])
        self.assertFalse(call["auto_adjust"])
        self.assertEqual(call["period"], CFG.daily_period)
        self.assertEqual(call["threads"], CFG.download_threads)

    def test_intradia_historico_pide_prepost(self):
        fake = FakeYF(lambda t: make_5m_bars("2026-09-18"))
        with _patch_yf(fake), _no_sleep():
            md.download_intraday_history(["AAA"], CFG)
        self.assertTrue(fake.calls[0]["prepost"])
        self.assertEqual(fake.calls[0]["interval"], "5m")
        self.assertEqual(fake.calls[0]["period"], "1mo")

    def test_reintentos_con_backoff_exponencial(self):
        fake = FakeYF(lambda t: make_daily("2026-09-18", 30), fail_first=2)
        with _patch_yf(fake), _no_sleep() as sleep:
            res = md.download_daily(["AAA"], CFG)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 4.0])
        self.assertIn("AAA", res.data)

    def test_un_lote_caido_no_detiene_los_demas(self):
        fake = FakeYF(lambda t: make_daily("2026-09-18", 30), always_fail_if="T00")
        tickers = [f"T{i:02d}" for i in range(40)]           # lotes de 30 y 10
        cfg = replace(CFG, max_retries=2)
        with _patch_yf(fake), _no_sleep(), self.assertLogs(md.logger, level="ERROR") as logs:
            res = md.download_daily(tickers, cfg)
        self.assertEqual(len(res.failed), 30)
        self.assertEqual(set(res.data), {f"T{i:02d}" for i in range(30, 40)})
        self.assertIn("T00: no se pudieron obtener datos diarios", "\n".join(logs.output))

    def test_ticker_con_punto_se_pide_a_yahoo_con_guion(self):
        fake = FakeYF(lambda t: make_daily("2026-09-18", 30))
        with _patch_yf(fake), _no_sleep():
            res = md.download_daily(["BRK.B", "AAA"], CFG)
        self.assertEqual(fake.calls[0]["tickers"], ["BRK-B", "AAA"])
        self.assertEqual(set(res.data), {"BRK.B", "AAA"})     # clave = ticker del universo

    def test_ticker_sin_datos_queda_en_failed(self):
        fake = FakeYF(lambda t: make_daily("2026-09-18", 30), empty_for={"BBB"})
        with _patch_yf(fake), _no_sleep(), self.assertLogs(md.logger, level="ERROR"):
            res = md.download_daily(["AAA", "BBB"], CFG)
        self.assertEqual(set(res.data), {"AAA"})
        self.assertIn("BBB", res.failed)

    def test_intradia_de_hoy_filtra_solo_el_dia(self):
        def frame(t):
            if t == "OLD":
                return make_5m_bars("2026-09-18")             # sin barras de hoy
            return pd.concat([make_5m_bars("2026-09-18"), make_5m_bars("2026-09-21", end="08:45")])
        fake = FakeYF(frame)
        with _patch_yf(fake), _no_sleep(), self.assertLogs(md.logger, level="WARNING"):
            res = md.download_intraday_today(["AAA", "OLD"], SESSION, CFG)
        self.assertEqual(set(res.data), {"AAA"})
        self.assertTrue(all(ts.date() == SESSION for ts in res.data["AAA"].index))
        self.assertEqual(res.failed["OLD"], "sin barras de hoy")


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.daily = pd.concat([make_daily("2026-09-18", 30),
                                make_daily("2026-09-21", 1, start_price=900.0)])  # fila de hoy

    def test_calidad_ok_y_valores(self):
        # barras 04:00-09:30 (66) => solo cuentan las 57 con inicio < 08:45
        bars = make_5m_bars("2026-09-21", "04:00", "09:30", volume=1000.0, price=100.0)
        bars = pd.concat([bars, make_5m_bars("2026-09-21", "09:30", "10:00", volume=1e9)])
        snap = md.extract_premarket_snapshot("AAA", self.daily, bars, SNAP_TS, CFG)
        self.assertEqual(snap.premarket_quality, "ok")
        self.assertEqual(snap.n_bars, 57)
        self.assertEqual(snap.premarket_volume, 57_000.0)
        self.assertEqual(snap.premarket_price, round(100.0 + 0.01 * 56, 4))
        self.assertEqual(snap.premarket_high, round(100.0 + 0.01 * 56 + 0.05, 4))
        self.assertEqual(snap.premarket_low, round(100.0 - 0.05, 4))
        self.assertEqual(snap.last_bar_age_min, 5.0)
        self.assertTrue(snap.last_bar_time.startswith("2026-09-21T08:40:00"))
        self.assertEqual(snap.prev_close, 129.0)              # cierre del 18, no la fila de hoy

    def test_calidad_stale(self):
        bars = make_5m_bars("2026-09-21", "04:00", "07:00")
        snap = md.extract_premarket_snapshot("AAA", self.daily, bars, SNAP_TS, CFG)
        self.assertEqual(snap.premarket_quality, "stale")
        self.assertGreater(snap.last_bar_age_min, CFG.stale_minutes)
        self.assertNotEqual(snap.premarket_price, md.NA)       # el dato se informa, pero marcado

    def test_calidad_sparse(self):
        bars = make_5m_bars("2026-09-21", "08:35", "08:45")    # 2 barras recientes
        snap = md.extract_premarket_snapshot("AAA", self.daily, bars, SNAP_TS, CFG)
        self.assertEqual(snap.premarket_quality, "sparse")
        self.assertEqual(snap.n_bars, 2)

    def test_calidad_missing_sin_rellenar(self):
        for bars in (None, pd.DataFrame(), make_5m_bars("2026-09-18"),
                     make_5m_bars("2026-09-21", "09:30", "10:00")):
            snap = md.extract_premarket_snapshot("AAA", self.daily, bars, SNAP_TS, CFG)
            self.assertEqual(snap.premarket_quality, "missing")
            self.assertEqual(snap.premarket_price, md.NA)
            self.assertEqual(snap.premarket_volume, md.NA)
            self.assertEqual(snap.n_bars, 0)
            self.assertEqual(snap.prev_close, 129.0)

    def test_sin_datos_diarios_prev_close_na(self):
        bars = make_5m_bars("2026-09-21", "04:00", "08:45")
        snap = md.extract_premarket_snapshot("AAA", None, bars, SNAP_TS, CFG)
        self.assertEqual(snap.prev_close, md.NA)

    def test_clasificador(self):
        stale, bars = CFG.stale_minutes, CFG.min_premarket_bars
        self.assertEqual(md.classify_premarket_quality(0, None, CFG)[0], "missing")
        self.assertEqual(md.classify_premarket_quality(10, stale + 1, CFG)[0], "stale")
        self.assertEqual(md.classify_premarket_quality(bars - 1, 5, CFG)[0], "sparse")
        self.assertEqual(md.classify_premarket_quality(bars, stale, CFG)[0], "ok")
        self.assertEqual(md.classify_premarket_quality(1, stale + 60, CFG)[0], "stale")  # stale > sparse

    def test_build_snapshots_aisla_fallos(self):
        bars = make_5m_bars("2026-09-21", "04:00", "08:45")
        with self.assertLogs(md.logger, level="ERROR"):
            snaps = md.build_snapshots(
                ["AAA", "BAD"], {"AAA": self.daily, "BAD": self.daily},
                {"AAA": bars, "BAD": "no es un DataFrame"}, SNAP_TS, CFG)
        self.assertEqual(list(snaps), ["AAA"])
        self.assertEqual(snaps["AAA"].to_dict()["premarket_quality"], "ok")


class TestIntervalMinutes(unittest.TestCase):
    def test_parseo(self):
        self.assertEqual(md.interval_minutes("5m"), 5)
        self.assertEqual(md.interval_minutes("15m"), 15)
        with self.assertRaises(ValueError):
            md.interval_minutes("1d")


class TestResolveSnapshotTs(unittest.TestCase):
    def test_casos(self):
        base = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)
        self.assertEqual(md.resolve_snapshot_ts(SESSION), base)
        antes = datetime(2026, 9, 21, 8, 35, tzinfo=NY_TZ)
        self.assertEqual(md.resolve_snapshot_ts(SESSION, antes), base)
        tarde = datetime(2026, 9, 21, 8, 52, 30, tzinfo=NY_TZ)
        self.assertEqual(md.resolve_snapshot_ts(SESSION, tarde),
                         datetime(2026, 9, 21, 8, 50, tzinfo=NY_TZ))
        despues_apertura = datetime(2026, 9, 21, 9, 40, tzinfo=NY_TZ)
        self.assertEqual(md.resolve_snapshot_ts(SESSION, despues_apertura),
                         datetime(2026, 9, 21, 9, 30, tzinfo=NY_TZ))
        otro_dia = datetime(2026, 9, 22, 8, 55, tzinfo=NY_TZ)
        self.assertEqual(md.resolve_snapshot_ts(SESSION, otro_dia), base)
        utc = datetime(2026, 9, 21, 12, 52, tzinfo=timezone.utc)      # 08:52 NY (verano)
        self.assertEqual(md.resolve_snapshot_ts(SESSION, utc),
                         datetime(2026, 9, 21, 8, 50, tzinfo=NY_TZ))


class TestIntradayPersistence(unittest.TestCase):
    def test_guardar_y_cargar(self):
        bars = make_5m_bars("2026-09-21", "04:00", "08:45", volume=1234.0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs(md.logger, level="WARNING"):
                paths = md.save_intraday_bars(
                    {"SPY": bars, "^VIX": bars, "VACIO": pd.DataFrame()}, SESSION, tmp)
            names = sorted(p.name for p in paths)
            self.assertEqual(names, ["SPY.csv.gz", "_VIX.csv.gz"])
            self.assertTrue(all(p.parent == Path(tmp) / "2026-09-21" for p in paths))
            back = md.load_intraday_bars("SPY", SESSION, tmp)
            self.assertEqual(len(back), len(bars))
            self.assertEqual(str(back.index.tz), "America/New_York")
            self.assertEqual(back.index[0], bars.index[0])
            self.assertEqual(float(back["Volume"].iloc[0]), 1234.0)
            self.assertTrue(md.load_intraday_bars("NOPE", SESSION, tmp).empty)


if __name__ == "__main__":
    unittest.main()
