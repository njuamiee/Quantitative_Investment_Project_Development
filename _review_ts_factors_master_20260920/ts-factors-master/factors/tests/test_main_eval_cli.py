"""main_eval CLI 的轻量测试：未知因子应抛出 KeyError。"""

import pandas as pd
import pytest

from factors.__main__ import main
from factors.library import FactorLibrary
from factors.main_eval import load_factor_signal


def test_unknown_factor_raises_key_error():
    lib = FactorLibrary()
    with pytest.raises(KeyError):
        load_factor_signal(lib, "NoSuchFactor999")


def test_list_cli_prints_stock_codes(monkeypatch, capsys):
    monkeypatch.setattr(
        FactorLibrary,
        "list_factors",
        lambda self: pd.DataFrame([
            {"factor_id": "Alpha001", "name": "Alpha001", "field": "oho_alpha_001", "table": "c01", "note": ""},
        ]),
    )
    monkeypatch.setattr(FactorLibrary, "list_stocks", lambda self: ["BJ920010", "BJ920011"])

    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "股票代码列表:" in out
    assert "BJ920010" in out
