"""Tests de la lectura del universo de tickers (PASO 0)."""
import pytest
from openpyxl import Workbook

import utils


def make_workbook(path, rows, extra_sheet_rows=None):
    """Crea un Excel de prueba con las filas dadas."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Hoja1"
    for row in rows:
        sheet.append(row)
    if extra_sheet_rows is not None:
        second = workbook.create_sheet("Hoja2")
        for row in extra_sheet_rows:
            second.append(row)
    workbook.save(path)


def test_lee_normaliza_y_elimina_duplicados_y_vacios(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_workbook(file, [
        ["Ticker", "Nombre"],
        ["aapl", "Apple"],
        [" nvda ", "Nvidia"],
        ["AAPL", "Duplicado"],
        [None, "Celda vacía"],
        ["$tsla", None],
        ["brk.b", "Berkshire"],
        ["???", "No válido"],
    ])
    result = utils.load_universe(file)
    assert result.tickers == ["AAPL", "NVDA", "TSLA", "BRK.B"]
    assert result.count == 4
    assert result.duplicates_dropped == 1
    assert result.empty_dropped == 1
    assert result.invalid_dropped == ["???"]


def test_cabecera_no_esta_en_la_primera_fila(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_workbook(file, [
        ["Mi lista de seguimiento"],
        [],
        ["Ticker"],
        ["MSFT"],
        ["GOOG"],
    ])
    assert utils.load_universe(file).tickers == ["MSFT", "GOOG"]


def test_cabecera_con_mayusculas_y_espacios(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_workbook(file, [["Nombre", " TICKER "], ["Apple", "AAPL"]])
    assert utils.load_universe(file).tickers == ["AAPL"]


def test_columna_en_segunda_hoja(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_workbook(
        file,
        [["Notas"], ["nada útil"]],
        extra_sheet_rows=[["Ticker"], ["AMD"]],
    )
    assert utils.load_universe(file).tickers == ["AMD"]


def test_sin_columna_ticker_da_error(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_workbook(file, [["Simbolo"], ["AAPL"]])
    with pytest.raises(utils.UniverseError):
        utils.load_universe(file)


def test_archivo_inexistente_da_error(tmp_path):
    with pytest.raises(utils.UniverseError):
        utils.load_universe(tmp_path / "no_existe.xlsx")


def test_sin_tickers_validos_da_error(tmp_path):
    file = tmp_path / "lista.xlsx"
    make_workbook(file, [["Ticker"], [None], [""]])
    with pytest.raises(utils.UniverseError):
        utils.load_universe(file)


def test_to_yahoo_symbol():
    assert utils.to_yahoo_symbol("BRK.B") == "BRK-B"
    assert utils.to_yahoo_symbol("AAPL") == "AAPL"
