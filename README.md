# BUPT 自然语言数据分析 Agent

当前版本包含：Excel 导入 SQLite、Schema/数据验证、5 个题目示例的
Golden SQL Benchmark，以及最小可运行的 CLI Text-to-SQL Agent。暂不包含
Streamlit、LangChain、LangGraph、向量数据库或多 Agent。

## 项目结构

```text
data/                 原始 Excel 和生成的 business.db
knowledge/            业务术语和表字段说明
outputs/              Agent 按需生成的图表
prepare_db.py         Excel -> SQLite 导入与验证
smoke_test.py         Golden SQL 和确定性断言
golden_results.json   Golden SQL 的实际运行结果
agent.py              业务知识、Text-to-SQL、SQL 安全、查询、结论与图表
app.py                CLI 入口
online_test.py        真实 LLM 五题测试与 Golden Result 自动对账
.env.example          LLM 配置模板，不包含真实 Key
```

## 安装与验证

```powershell
python -m pip install -r requirements.txt
python prepare_db.py
python smoke_test.py
```

`prepare_db.py` 不会修改原始 Excel。它先生成 `data/business.db.tmp`，只有
完整性验证通过后才替换 `data/business.db`。`smoke_test.py` 会重新生成
`golden_results.json`。

## LLM 配置

复制 `.env.example` 为 `.env`，然后填写：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
```

`LLM_BASE_URL` 留空时使用 OpenAI Python SDK 的默认 API 地址。不要将真实
API Key 写入 `.env.example` 或提交到版本库。

## 运行 CLI

```powershell
python app.py
```

Agent 会加载 `knowledge/*.md` 和 SQLite `PRAGMA table_info`，生成一条只读
SQLite SQL，安全校验后执行，最多展示 200 行，再让模型仅基于实际
查询结果生成简短结论。SQL 失败时最多自动修复一次。

## 真实 LLM Golden 对账

配置 `.env` 后运行：

```powershell
python online_test.py
```

脚本依次测试五个自然语言问题，并把首次 SQL、自动修复信息、最终结果、
模型结论和 Golden 对账状态写入 `outputs/online_test_report.txt`。缺少
`LLM_API_KEY` 或 `LLM_MODEL` 时不会发起 API 请求，报告状态为 `SKIPPED`，
进程退出码为 `2`。

## 关键口径

- 实际表名是 `store_info` 和 `product_info`，不是 Markdown 中出现的
  `dim_store` / `dim_product`。
- 成交销额 = `quantity * sale_price - discount_amount`。
- 汇总毛利率 = `SUM(毛利额) / SUM(成交销额)`，不对单行毛利率求平均。
- 销售期间使用 `order_date`，退款期间使用 `refund_date`。
- SKU 放量与毛利率下降只能作为相关证据，不宣称已证明因果。
