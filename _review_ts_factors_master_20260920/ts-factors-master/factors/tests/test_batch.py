"""main_batch 纯函数测试：全局配置读取、出图子集裁剪与断点续算（不触真实因子库 IO）。"""

import json

import pandas as pd
import pytest

from factors.main_batch import (
    _DEFAULT_BATCH_CONFIG,
    _check_run_params,
    _clip_plot_subset,
    _find_resume_dir,
    _load_batch_config,
    _load_partial,
)


def test_load_batch_config_missing_file(tmp_path):
    """配置文件不存在时回退默认值。"""
    cfg = _load_batch_config(str(tmp_path / "nope.json"))
    assert cfg == _DEFAULT_BATCH_CONFIG


def test_load_batch_config_partial_override(tmp_path):
    """只覆盖出现的字段，未知字段忽略，其余保持默认。"""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"batch": {"njobs": 3, "junk": 1}, "other": {}}),
                 encoding="utf-8")
    cfg = _load_batch_config(str(p))
    assert cfg["njobs"] == 3
    assert cfg["min_bars"] == _DEFAULT_BATCH_CONFIG["min_bars"]
    assert cfg["plot_sides"] == _DEFAULT_BATCH_CONFIG["plot_sides"]
    assert "junk" not in cfg


def test_load_batch_config_broken_json(tmp_path):
    """JSON 损坏时回退默认值，不抛异常。"""
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert _load_batch_config(str(p)) == _DEFAULT_BATCH_CONFIG


def test_clip_plot_subset():
    """side 裁剪到 1..n_buckets 并去重排序；k 只保留 horizons 中的值且按 horizons 顺序。"""
    sides, ks = _clip_plot_subset([1, 5, 99, 0], [20, 5, 1, 7], 5,
                                  [1, 2, 3, 5, 10, 15, 20])
    assert sides == [1, 5]          # 0→1，99→5（n_buckets），去重
    assert ks == [1, 5, 20]         # 7 不在 horizons 剔除；顺序随 horizons


# ----------------------------------------------------------------------
# 断点续算
# ----------------------------------------------------------------------
def _make_incomplete_dir(path):
    """造一个"未完成"的批量目录：含 partial 明细文件 + 参数快照。"""
    import os

    os.makedirs(path, exist_ok=True)
    pd.DataFrame([{"factor_id": "Alpha001", "stock": "BJ920010",
                   "n_bars": 300, "error": ""}]).to_csv(
        os.path.join(path, "summary_partial_detail.csv"),
        index=False, encoding="utf-8-sig")


def test_find_resume_dir(tmp_path):
    """只认含 partial 明细文件且名字匹配 batch_top<N>* 的目录，取 mtime 最新者。"""
    import os

    assert _find_resume_dir(2, str(tmp_path)) is None
    older = tmp_path / "batch_top2"
    newer = tmp_path / "batch_top2_1"
    _make_incomplete_dir(str(older))
    _make_incomplete_dir(str(newer))
    (tmp_path / "batch_top2_2").mkdir()          # 无 partial → 不算未完成
    _make_incomplete_dir(str(tmp_path / "batch_top20"))  # top 不匹配
    os.utime(older / "summary_partial_detail.csv", (1000, 1000))
    os.utime(newer / "summary_partial_detail.csv", (2000, 2000))
    assert _find_resume_dir(2, str(tmp_path)) == str(newer)
    assert _find_resume_dir(20, str(tmp_path)) == str(tmp_path / "batch_top20")
    assert _find_resume_dir(3, str(tmp_path)) is None


def test_check_run_params(tmp_path):
    """参数一致放行；不一致或参数文件缺失 → SystemExit。"""
    d = tmp_path / "batch_top2"
    d.mkdir()
    params = {"top": 2, "params": {"horizons": [1, 2], "fee": 5e-05}}
    (d / "run_params.json").write_text(json.dumps(params), encoding="utf-8")
    _check_run_params(str(d), params)            # 一致：不抛
    with pytest.raises(SystemExit):
        _check_run_params(str(d), {"top": 2,
                                   "params": {"horizons": [1, 2, 3], "fee": 5e-05}})
    with pytest.raises(SystemExit):
        _check_run_params(str(tmp_path / "nope"), params)   # 文件缺失


def test_load_partial(tmp_path):
    """载入 partial 明细/长表；error 空值归一为 ""；长表缺失时返回空列表。"""
    import os

    d = tmp_path / "d"
    _make_incomplete_dir(str(d))
    pd.DataFrame([{"factor_id": "Alpha001", "stock": "BJ920010", "combo": "s1_long",
                   "k": 1, "side": 1, "direction": "long", "total_ret": 0.1,
                   "error": ""}]).to_csv(
        os.path.join(str(d), "summary_partial_long.csv"),
        index=False, encoding="utf-8-sig")
    det, lon = _load_partial(str(d))
    assert len(det) == 1 and det[0]["n_bars"] == 300 and det[0]["error"] == ""
    assert len(lon) == 1 and lon[0]["combo"] == "s1_long"
    # 无长表时
    (tmp_path / "d2").mkdir()
    pd.DataFrame([{"factor_id": "A", "stock": "B", "error": "skipped"}]).to_csv(
        os.path.join(str(tmp_path / "d2"), "summary_partial_detail.csv"),
        index=False, encoding="utf-8-sig")
    det2, lon2 = _load_partial(str(tmp_path / "d2"))
    assert det2[0]["error"] == "skipped"
    assert lon2 == []
