"""Generadores de datos sintéticos para los tests (sin red)."""
from __future__ import annotations

from datetime import date, time
from typing import Callable

import numpy as np
import pandas as pd

from utils import NY_TZ


def make_5m_bars(
    day: date | str,
    start: str = "04:00",
    end: str = "09:30",
    volume: float | Callable[[pd.Timestamp], float] = 1000.0,
    price: float = 100.0,
    slope: float = 0.01,
) -> pd.DataFrame:
    """Barras de 5 minutos entre ``start`` (incl.) y ``end`` (excl.), índice en NY."""
    idx = pd.date_range(f"{day} {start}", f"{day} {end}", freq="5min", tz=NY_TZ,
                        inclusive="left")
    close = price + slope * np.arange(len(idx))
    vols = [volume(ts) if callable(volume) else volume for ts in idx]
    return pd.DataFrame(
        {"Open": close - 0.02, "High": close + 0.05, "Low": close - 0.05,
         "Close": close, "Volume": vols},
        index=idx,
    )


def make_full_day(day: date | str, pm_volume: float, trap: float = 1e9) -> pd.DataFrame:
    """Día completo 04:00-16:00. Volumen ``pm_volume`` antes de las 08:45 y ``trap``
    después (para comprobar que la ventana de RVOL no filtra barras de más)."""
    def vol(ts: pd.Timestamp) -> float:
        return pm_volume if ts.time() < time(8, 45) else trap

    return make_5m_bars(day, "04:00", "16:00", volume=vol)


def make_daily(end_day: str, periods: int, start_price: float = 100.0, step: float = 1.0,
               volume: float = 2_000_000.0) -> pd.DataFrame:
    """OHLCV diario en días laborables que termina en ``end_day`` (inclusive)."""
    idx = pd.bdate_range(end=end_day, periods=periods).tz_localize(NY_TZ)
    close = start_price + step * np.arange(periods)
    return pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0, "Close": close,
         "Volume": np.full(periods, volume)},
        index=idx,
    )


def business_days(end_day: str, periods: int) -> list[str]:
    """Fechas ISO de ``periods`` días laborables que terminan en ``end_day``."""
    return [d.date().isoformat() for d in pd.bdate_range(end=end_day, periods=periods)]
