"""Tests de technical_analysis.py con datos sintéticos (sin red)."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pandas as pd

import config
from utils import NY_TZ
from synthetic import business_days, make_5m_bars, make_daily, make_full_day
import technical_analysis as ta

SESSION = date(2026, 9, 21)                                   # lunes
SNAP_TS = datetime(2026, 9, 21, 8, 45, tzinfo=NY_TZ)


def _hist(volumes: list[float]) -> pd.DataFrame:
    """Histórico 5m: una sesión por volumen, terminando el viernes 2026-09-18."""
    days = business_days("2026-09-18", len(volumes))
    return pd.concat([make_full_day(d, v) for d, v in zip(days, volumes)])


class TestIndicatorsByHand(unittest.TestCase):
    def test_rsi_wilder_valores_calculados_a_mano(self):
        # deltas: +1 +1 -1 +1 +1 ; periodo 3
        # semilla: ganancia 2/3, pérdida 1/3 => RS 2 => RSI 66.6667
        # idx4: ganancia (2/3*2+1)/3=7/9, pérdida (1/3*2)/3=2/9 => RS 3.5 => 77.7778
        # idx5: ganancia (7/9*2+1)/3=23/27, pérdida (2/9*2)/3=4/27 => RS 5.75 => 85.1852
        close = pd.Series([10, 11, 12, 11, 12, 13], dtype=float)
        rsi = ta.rsi_wilder(close, 3)
        self.assertTrue(rsi.iloc[:3].isna().all())
        self.assertAlmostEqual(rsi.iloc[3], 66.6667, places=3)
        self.assertAlmostEqual(rsi.iloc[4], 77.7778, places=3)
        self.assertAlmostEqual(rsi.iloc[5], 85.1852, places=3)

    def test_rsi_casos_limite(self):
        subida = pd.Series(np.arange(1.0, 20.0))
        self.assertEqual(ta.rsi_wilder(subida, 14).iloc[-1], 100.0)
        plano = pd.Series(np.full(20, 5.0))
        self.assertEqual(ta.rsi_wilder(plano, 14).iloc[-1], 50.0)
        corto = pd.Series([1.0, 2.0, 3.0])
        self.assertTrue(ta.rsi_wilder(corto, 14).isna().all())

    def test_atr_wilder_valores_calculados_a_mano(self):
        # TR: 2, max(2,2,0)=2, max(3,2,1)=3, max(2,1,1)=2 ; periodo 3
        # ATR idx2 = 7/3 ; idx3 = (7/3*2+2)/3 = 20/9
        high = pd.Series([10, 11, 12, 12], dtype=float)
        low = pd.Series([8, 9, 9, 10], dtype=float)
        close = pd.Series([9, 10, 11, 11], dtype=float)
        atr = ta.atr_wilder(high, low, close, 3)
        self.assertTrue(atr.iloc[:2].isna().all())
        self.assertAlmostEqual(atr.iloc[2], 7 / 3, places=6)
        self.assertAlmostEqual(atr.iloc[3], 20 / 9, places=6)

    def test_ema_con_semilla_sma(self):
        # periodo 3 => alpha 0.5 ; semilla (1+2+3)/3=2 ; luego 3 y 4
        ema = ta.ema(pd.Series([1, 2, 3, 4, 5], dtype=float), 3)
        self.assertTrue(ema.iloc[:2].isna().all())
        self.assertEqual(list(ema.iloc[2:]), [2.0, 3.0, 4.0])

    def test_sma(self):
        sma = ta.sma(pd.Series([1, 2, 3, 4], dtype=float), 2)
        self.assertTrue(np.isnan(sma.iloc[0]))
        self.assertEqual(list(sma.iloc[1:]), [1.5, 2.5, 3.5])


class TestDailyIndicators(unittest.TestCase):
    def setUp(self):
        self.daily = make_daily("2026-09-18", 300)          # 300 sesiones completadas
        today = make_daily("2026-09-21", 1, start_price=500.0)
        today["High"] = 999.0                                # fila de hoy incompleta / absurda
        self.daily_with_today = pd.concat([self.daily, today])

    def test_excluye_la_sesion_de_hoy(self):
        ind = ta.compute_daily_indicators(self.daily_with_today, SESSION)
        closes = self.daily["Close"].to_numpy()
        self.assertEqual(ind["last_close"], round(closes[-1], 4))
        self.assertEqual(ind["last_close_date"], "2026-09-18")
        self.assertEqual(ind["max_5d"], round(closes[-1] + 1.0, 4))       # High = close + 1
        self.assertEqual(ind["min_5d"], round(closes[-5] - 1.0, 4))       # Low = close - 1
        self.assertEqual(ind["max_20d"], round(closes[-1] + 1.0, 4))
        self.assertEqual(ind["min_20d"], round(closes[-20] - 1.0, 4))

    def test_medias_y_retorno(self):
        ind = ta.compute_daily_indicators(self.daily_with_today, SESSION)
        closes = self.daily["Close"].to_numpy()
        self.assertAlmostEqual(ind["sma20"], closes[-20:].mean(), places=3)
        self.assertAlmostEqual(ind["sma50"], closes[-50:].mean(), places=3)
        self.assertAlmostEqual(ind["return_5d_pct"], (closes[-1] / closes[-6] - 1) * 100, places=3)
        self.assertEqual(ind["avg_volume_20d"], 2_000_000.0)
        self.assertAlmostEqual(ind["avg_dollar_volume_20d"], closes[-20:].mean() * 2_000_000.0,
                               delta=1.0)
        # tendencia lineal: ATR = TR constante de 2 (High-Low) => 2
        self.assertAlmostEqual(ind["atr14"], 2.0, places=3)
        self.assertAlmostEqual(ind["atr_pct"], 2.0 / closes[-1] * 100, places=3)
        self.assertEqual(ind["rsi14"], 100.0)

    def test_rango_52_semanas(self):
        ind = ta.compute_daily_indicators(self.daily, SESSION)
        idx_dates = [d.date() for d in self.daily.index]
        in_range = [i for i, d in enumerate(idx_dates) if (SESSION - d).days <= 52 * 7]
        closes = self.daily["Close"].to_numpy()
        self.assertEqual(ind["high_52w"], round(closes[in_range].max() + 1.0, 4))
        self.assertEqual(ind["low_52w"], round(closes[in_range].min() - 1.0, 4))

    def test_pocos_datos_devuelve_na(self):
        ind = ta.compute_daily_indicators(make_daily("2026-09-18", 10), SESSION)
        self.assertEqual(ind["sma50"], ta.NA)
        self.assertEqual(ind["avg_volume_20d"], ta.NA)
        self.assertEqual(ind["max_20d"], ta.NA)
        self.assertNotEqual(ind["max_5d"], ta.NA)
        vacio = ta.compute_daily_indicators(None, SESSION)
        self.assertEqual(vacio["last_close"], ta.NA)
        self.assertEqual(vacio["sessions_available"], 0)


class TestGapAndVolume(unittest.TestCase):
    def test_gap_pct(self):
        self.assertEqual(ta.gap_pct(102.0, 100.0), 2.0)
        self.assertEqual(ta.gap_pct(97.5, 100.0), -2.5)
        self.assertEqual(ta.gap_pct(ta.NA, 100.0), ta.NA)
        self.assertEqual(ta.gap_pct(100.0, 0.0), ta.NA)

    def test_gap_en_atr_y_rango_premarket(self):
        self.assertEqual(ta.gap_in_atr(103.0, 100.0, 2.0), 1.5)
        self.assertEqual(ta.gap_in_atr(103.0, 100.0, ta.NA), ta.NA)
        rng = ta.premarket_range(104.0, 101.0, 100.0, 2.0)
        self.assertEqual(rng["premarket_range_pct"], 3.0)
        self.assertEqual(rng["premarket_range_atr"], 1.5)
        self.assertEqual(ta.premarket_range(ta.NA, 101.0, 100.0, 2.0)["premarket_range_pct"], ta.NA)

    def test_pm_pct_of_adv_no_es_rvol(self):
        self.assertEqual(ta.pm_pct_of_adv(500_000, 2_000_000), 25.0)
        self.assertEqual(ta.pm_pct_of_adv(500_000, ta.NA), ta.NA)


class TestRvol(unittest.TestCase):
    def test_caso_normal_usa_mediana_y_ventana_correcta(self):
        # 12 sesiones con 57 barras (04:00-08:40) de 1000*(i+1); mediana = 57000*6.5
        hist = _hist([1000.0 * (i + 1) for i in range(12)])
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=13_000.0)
        res = ta.compute_rvol(hist, today, SNAP_TS)
        self.assertEqual(res.sessions_used, 12)
        self.assertEqual(res.numerator, 57 * 13_000.0)
        self.assertEqual(res.denominator, 57_000 * 6.5)
        self.assertAlmostEqual(res.rvol, 2.0, places=4)
        self.assertEqual(res.reason, "")

    def test_las_barras_de_hoy_a_partir_del_snapshot_no_cuentan(self):
        hist = _hist([1000.0] * 12)
        today = make_5m_bars("2026-09-21", "04:00", "09:30", volume=2000.0)
        res = ta.compute_rvol(hist, today, SNAP_TS)          # 57 barras de la ventana
        self.assertEqual(res.numerator, 57 * 2000.0)

    def test_solo_las_ultimas_n_sesiones(self):
        volumes = [1e6] * 5 + [1000.0] * 20                 # las 5 más antiguas quedan fuera
        hist = _hist(volumes)
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=3000.0)
        res = ta.compute_rvol(hist, today, SNAP_TS)
        self.assertEqual(res.sessions_used, 20)
        self.assertAlmostEqual(res.rvol, 3.0, places=4)

    def test_pocas_sesiones(self):
        hist = _hist([1000.0] * 5)
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=3000.0)
        res = ta.compute_rvol(hist, today, SNAP_TS)
        self.assertEqual(res.rvol, ta.NA)
        self.assertEqual(res.sessions_used, 5)
        self.assertIn("5 sesiones válidas", res.reason)
        self.assertEqual(res.numerator, 57 * 3000.0)         # el numerador sí se informa

    def test_umbral_minimo_configurable(self):
        hist = _hist([1000.0] * 5)
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=3000.0)
        rvol_cfg = replace(config.RVOL, min_sessions=5)
        self.assertAlmostEqual(ta.compute_rvol(hist, today, SNAP_TS, rvol_cfg).rvol, 3.0,
                               places=4)

    def test_ventana_vacia(self):
        hist = _hist([1000.0] * 12)
        solo_tarde = make_5m_bars("2026-09-21", "09:00", "09:30", volume=5000.0)
        res = ta.compute_rvol(hist, solo_tarde, SNAP_TS)
        self.assertEqual(res.rvol, ta.NA)
        self.assertIn("sin barras pre-market", res.reason)
        # snapshot a las 04:00: la ventana [04:00, 04:00) está vacía
        temprano = datetime(2026, 9, 21, 4, 0, tzinfo=NY_TZ)
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=5000.0)
        self.assertEqual(ta.compute_rvol(hist, today, temprano).rvol, ta.NA)
        self.assertEqual(ta.compute_rvol(hist, None, SNAP_TS).rvol, ta.NA)

    def test_sin_historico(self):
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=5000.0)
        res = ta.compute_rvol(None, today, SNAP_TS)
        self.assertEqual(res.rvol, ta.NA)
        self.assertIn("histórico", res.reason)

    def test_mediana_cero(self):
        hist = _hist([0.0] * 12)
        today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=5000.0)
        res = ta.compute_rvol(hist, today, SNAP_TS)
        self.assertEqual(res.rvol, ta.NA)
        self.assertIn("mediana", res.reason)


class TestLevelsAndRelativeStrength(unittest.TestCase):
    def setUp(self):
        self.indicators = {"max_5d": 103.0, "min_5d": 97.0, "sma20": 99.0, "ema9": 100.5,
                           "max_20d": ta.NA, "high_52w": 110.0}
        self.snapshot = {"prev_close": 98.0, "premarket_high": 101.0, "premarket_low": 99.5}

    def test_tabla_de_niveles(self):
        levels = ta.build_levels_table(100.0, 2.0, self.indicators, self.snapshot)
        labels = [r["label"] for r in levels]
        self.assertEqual(labels, ["high_52w", "max_5d", "premarket_high", "ema9",
                                  "premarket_low", "sma20", "prev_close", "min_5d"])
        self.assertNotIn("max_20d", labels)                 # sin dato => se omite
        by_label = {r["label"]: r for r in levels}
        self.assertEqual(by_label["max_5d"]["side"], "resistance")
        self.assertEqual(by_label["max_5d"]["distance_pct"], 3.0)
        self.assertEqual(by_label["max_5d"]["distance_atr"], 1.5)
        self.assertEqual(by_label["sma20"]["side"], "support")
        self.assertEqual(by_label["sma20"]["distance_pct"], -1.0)
        self.assertEqual(ta.nearest_level(levels, "resistance")["label"], "ema9")
        self.assertEqual(ta.nearest_level(levels, "support")["label"], "premarket_low")

    def test_nivel_en_el_precio_no_es_soporte_ni_resistencia(self):
        snap = dict(self.snapshot, premarket_high=100.0)
        levels = ta.build_levels_table(100.0, 2.0, self.indicators, snap)
        self.assertEqual({r["label"]: r["side"] for r in levels}["premarket_high"], "at_price")
        self.assertNotEqual(ta.nearest_level(levels, "resistance")["label"], "premarket_high")

    def test_sin_precio_o_sin_atr(self):
        self.assertEqual(ta.build_levels_table(ta.NA, 2.0, self.indicators, self.snapshot), [])
        levels = ta.build_levels_table(100.0, ta.NA, self.indicators, self.snapshot)
        self.assertTrue(levels)
        self.assertTrue(all(r["distance_atr"] == ta.NA for r in levels))
        self.assertEqual(ta.nearest_level([], "support"), ta.NA)

    def test_fuerza_relativa(self):
        bench = {"SPY": {"pm_change_pct": 0.5, "return_5d_pct": 1.0},
                 "QQQ": {"pm_change_pct": ta.NA, "return_5d_pct": 2.0}}
        rs = ta.relative_strength(3.0, 4.0, bench)
        self.assertEqual(rs["rs_pm_vs_spy"], 2.5)
        self.assertEqual(rs["rs_5d_vs_spy"], 3.0)
        self.assertEqual(rs["rs_pm_vs_qqq"], ta.NA)
        self.assertEqual(rs["rs_5d_vs_qqq"], 2.0)
        vacio = ta.relative_strength(ta.NA, ta.NA, None)
        self.assertTrue(all(v == ta.NA for v in vacio.values()))
        self.assertEqual(len(vacio), 4)


class TestAnalyzeTicker(unittest.TestCase):
    def setUp(self):
        self.daily = make_daily("2026-09-18", 120)
        self.hist = _hist([1000.0 * (i + 1) for i in range(12)])
        self.today = make_5m_bars("2026-09-21", "04:00", "08:45", volume=13_000.0, price=121.0)
        prev = float(self.daily["Close"].iloc[-1])
        self.snapshot = {
            "ticker": "AAA", "prev_close": prev, "premarket_price": prev * 1.02,
            "premarket_high": prev * 1.03, "premarket_low": prev * 1.00,
            "premarket_volume": 741_000.0, "premarket_quality": "ok",
        }
        self.bench = {"SPY": {"pm_change_pct": 0.5, "return_5d_pct": 1.0},
                      "QQQ": {"pm_change_pct": 0.7, "return_5d_pct": 1.5}}

    def test_registro_completo_y_serializable(self):
        rec = ta.analyze_ticker("AAA", self.daily, self.hist, self.today, self.snapshot,
                                SNAP_TS, self.bench)
        self.assertEqual(rec["gap_pct"], 2.0)
        self.assertAlmostEqual(rec["rvol"]["rvol"], 2.0, places=3)
        self.assertEqual(rec["pm_pct_of_adv"], round(741_000 / 2_000_000 * 100, 4))
        self.assertAlmostEqual(rec["relative_strength"]["rs_pm_vs_spy"], 1.5, places=3)
        self.assertNotEqual(rec["gap_atr"], ta.NA)
        self.assertTrue(rec["levels"])
        json.dumps(rec)                                      # sin tipos numpy

    def test_analyze_many_aisla_los_fallos(self):
        snaps = {"AAA": self.snapshot, "BAD": self.snapshot, "NOSNAP": None}
        del snaps["NOSNAP"]
        daily = {"AAA": self.daily, "BAD": "esto no es un DataFrame", "NOSNAP": self.daily}
        results, failures = ta.analyze_many(
            ["AAA", "BAD", "NOSNAP"], daily, {"AAA": self.hist}, {"AAA": self.today},
            snaps, SNAP_TS, self.bench)
        self.assertEqual(list(results), ["AAA"])
        self.assertEqual(set(failures), {"BAD", "NOSNAP"})


if __name__ == "__main__":
    unittest.main()
