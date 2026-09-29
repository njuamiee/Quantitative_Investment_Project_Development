"""命令行入口：在 src 目录下运行

    python -m factors init                                   # 全量重建因子库
    python -m factors update                                 # 增量追加新日期数据
    python -m factors list                                   # 列出所有因子
    python -m factors eval Alpha001 [--stock BJ920010] ...   # 因子 IC 评价
    python -m factors eval all --stock BJ920725              # 单只股票 × 全部因子
    python -m factors sign Alpha098 BJ920725                 # rolling IC 正负预测面板图
    python -m factors plot Alpha001 BJ920010 [start] [end]   # 绘制行情+因子曲线

注意使用 64 位 Python 3.11（D:\\jzhang\\setup\\python64\\Python311\\python.exe）。
"""

from __future__ import annotations

import argparse
import sys

from .library import FactorLibrary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m factors", description="因子库管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="全量重建因子库")
    sub.add_parser("update", help="增量追加新日期数据")
    sub.add_parser("list", help="列出所有因子")

    p_eval = sub.add_parser("eval", help="因子 IC 评价（详见 factors/main_eval.py）")
    p_eval.add_argument("eval_args", nargs=argparse.REMAINDER,
                        help="透传给 factors/main_eval.py 的参数")

    p_sign = sub.add_parser("sign", help="rolling IC 正负预测模型对比面板图"
                                         "（详见 factors/main_sign.py）")
    p_sign.add_argument("sign_args", nargs=argparse.REMAINDER,
                        help="透传给 factors/main_sign.py 的参数")

    p_plot = sub.add_parser("plot", help="绘制个股日 K 线 + 因子曲线")
    p_plot.add_argument("factor", help="因子标识或因子名，如 Alpha001")
    p_plot.add_argument("stockid", help="股票代码，如 BJ920010")
    p_plot.add_argument("start", nargs="?", default=None, help="起始日期（可选）")
    p_plot.add_argument("end", nargs="?", default=None, help="结束日期（可选）")
    p_plot.add_argument("--save", default=None, help="图片保存路径（可选，不给则不存盘）")

    args = parser.parse_args(argv)

    if args.cmd == "init":
        FactorLibrary().init()
    elif args.cmd == "update":
        FactorLibrary().update()
    elif args.cmd == "list":
        lib = FactorLibrary()
        reg = lib.list_factors()
        print(reg.to_string(index=False))
        print(f"\n共 {len(reg)} 个因子")

        try:
            stocks = lib.list_stocks()
            print(f"共 {len(stocks)} 只股票")
            print("股票代码列表:")
            print(", ".join(stocks))
        except (FileNotFoundError, KeyError, ValueError):
            print("股票代码列表: 未构建因子库或尚无 parquet 数据文件")

        print(f"\n库目录: {lib.lib_dir}")
    elif args.cmd == "eval":
        from . import main_eval

        main_eval.main(args.eval_args or None)
    elif args.cmd == "sign":
        from . import main_sign

        main_sign.main(args.sign_args or None)
    elif args.cmd == "plot":
        from .plotting import plot_factor

        plot_factor(args.factor, args.stockid, args.start, args.end,
                    save_path=args.save, show=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
