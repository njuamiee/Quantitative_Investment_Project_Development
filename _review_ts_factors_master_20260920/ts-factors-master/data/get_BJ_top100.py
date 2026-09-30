import akshare as ak
import pandas as pd

# 获取北交所所有股票列表
stock_list = ak.stock_bj_a_spot_em()  # 东方财富北交所行情

# 遍历每只股票获取近 30 日 K 线并累加成交额
result = []
for _, row in stock_list.iterrows():
    code = row['代码']
    name = row['名称']
    try:
        kline = ak.stock_bj_a_hist(symbol=code, period="daily", 
                                   start_date="20260803", end_date="20260902")
        total_amount = kline['成交额'].sum()  # 单位：元
        result.append({'代码': code, '名称': name, '近30日成交额': total_amount})
    except:
        continue

# 排序并取前 100
result_df = pd.DataFrame(result).sort_values('近30日成交额', ascending=False).head(100)
print(result_df)

