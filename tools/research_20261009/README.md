# 首次完整汇报与解释图

本次按首次汇报重组已有研究，未重新训练或覆盖历史实验结果。

- 当前PPT：`学习笔记/10月9日汇报/量化选股项目_首次完整汇报_20261009.pptx`，37页，主汇报32页、附录5页。
- 逐页讲稿：同目录`首次完整汇报_逐页讲稿.md`；同样内容写入PPT讲者备注。
- 九项建议按9月25日整理记录逐项解释，不声称是老师讲话逐字稿。
- 8张扩展实验图完整保留，另加入原北交所账户图、五分组图和已有HMM结果的解释图。

`plot_hmm_explanation.py`读取已保存的`hmm_strict_probabilities.csv`与`hmm_strict_account.csv`，按日期一对一关联，分别画出高波动状态概率、允许仓位上限和实际股票仓位。三者含义不同；此图没有新增模型或收益。

```powershell
& '.venv_quant/Scripts/python.exe' 'tools/research_20261009/plot_hmm_explanation.py'
```

输出位于`training_runs/first_report_20261009/`：PNG、PDF及输入哈希`figure_provenance.json`。PPT使用artifact-tool JSX，全部页经渲染、逐页目视与布局检查。`presentation_qa.json`记录本版页数、备注、图片来源及文件哈希；旧23页稿的验证记录仍保留，不应当作本版记录。
