# 智能数据分析 Agent

> 基于业务知识与数据库 Schema 的自然语言 Text-to-SQL 数据分析系统

这是一个面向业务数据分析场景的轻量级 LLM Agent。用户无需编写 SQL，只需输入中文问题，系统便会结合 Markdown 业务知识、SQLite Schema 和数据库时间范围理解业务语义，生成并安全执行只读 SQL，最终返回数据表格、自然语言结论和必要的可视化图表。

它不仅调用一次模型，而是完成“构建业务上下文 → 生成 SQL → 安全校验 → 查询数据 → 分析结果”的完整闭环，因此具备一个数据分析 Agent 的基本工作流。

## 项目背景

项目包含四类业务数据：

| 数据表 | 含义 |
| --- | --- |
| `store_info` | 门店及所属区域、城市等信息 |
| `product_info` | 商品、品类、成本和标价信息 |
| `sales_order` | 销售订单、渠道、数量、售价和折扣 |
| `refund_record` | 退款记录、退款金额和退款原因 |

`knowledge/` 中的 `business_terms.md` 和各数据表说明 Markdown 定义了字段含义、业务术语及指标口径，使 Agent 能理解“华东战区”“即时零售”“SKU”“成交销额”“退损率”“毛利率”“放量”等业务表达，并将其转换为正确的数据查询。

## 核心功能

1. **自然语言数据查询**：直接使用中文提出门店、商品、销售、退款和利润分析问题。
2. **Markdown 业务知识注入**：启动时读取 `knowledge/*.md`，将业务术语映射到实际字段和 SQL 条件。例如：
   - 战区 → `store_info.region`
   - 即时零售 → `sales_order.channel_code = 'O2O'`
   - SKU → `product_info.product_id`
3. **动态 Schema 感知**：通过 SQLite `PRAGMA table_info` 和外键信息读取真实数据库结构，而非在 Prompt 中写死 Schema。
4. **Text-to-SQL**：LLM 综合“用户问题 + 业务知识 + Schema + Temporal Context”，生成 SQLite 兼容的只读 SQL。
5. **SQL 安全执行**：只允许 `SELECT` 或合法的 `WITH ... SELECT`，拒绝写操作、DDL、管理语句和多语句执行；数据库同时使用只读 URI、`query_only` 和 SQLite authorizer，并限制查询结果行数。LLM 不会生成并执行任意 Python。
6. **有限错误恢复**：SQL 校验或执行失败时，最多允许模型自动修复 1 次，不进行无限循环。
7. **数据分析与可视化**：查询结果以 pandas DataFrame 返回，并按结果需要生成 `bar`、`line` 或 `pie` 图表。
8. **自然语言总结**：模型只能依据实际 SQL 结果总结。无法证明因果关系时，使用“可能”“数据显示”“存在拖累迹象”等谨慎表达。

> `business_terms.md` 中保留了题目描述使用的 `dim_store` / `dim_product` 名称；Agent 会明确纠正为真实表名 `store_info` / `product_info`。

## 业务口径

| 指标 | 定义 |
| --- | --- |
| 成交销额 | `quantity * sale_price - discount_amount` |
| 销量 | `SUM(quantity)` |
| 毛利额 | `成交销额 - quantity * unit_cost` |
| 毛利率 | `SUM(毛利额) / SUM(成交销额)` |
| 退损金额 | `SUM(refund_amount)` |
| 退损率 | `退款金额 / 成交销额` |
| 净销额 | `成交销额 - 退款金额` |

业务正确性约束：

- 聚合毛利率必须使用“汇总毛利额 ÷ 汇总成交销额”，不能使用 `AVG(单笔毛利率)`。
- 销售分析使用 `sales_order.order_date`。
- 退款分析使用 `refund_record.refund_date`。
- 同时分析销售和退款时先分别按各自日期聚合，再进行 JOIN，避免明细关联造成重复计算。

## 系统架构

```text
用户自然语言问题
        │
        ▼
┌───────────────────────┐
│ Agent Context Builder │
│ Markdown + Schema     │
│ Temporal Context      │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│          LLM          │
│      Text-to-SQL      │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│   SQL Safety Layer    │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│        SQLite         │
│    Read-only Query    │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ pandas + matplotlib   │
└───────────┬───────────┘
            ▼
        分析结论
```

- **Context Builder**：加载 Markdown、真实表结构、外键和日期范围。
- **LLM Text-to-SQL**：理解问题并输出结构化 SQL 计划、简短推理摘要和图表类型。
- **SQL Safety Layer**：完成语句类型、危险关键字、多语句和 SQLite authorizer 检查。
- **SQLite / pandas**：负责确定性的数值计算和结构化结果承载。
- **结果层**：基于真实结果生成谨慎的业务结论和按需图表。

## 数据表关系

```text
store_info
    │ store_id
    ▼
sales_order
    │
    ├── product_id ──→ product_info
    │
    └── order_id ───→ refund_record
```

实际外键关系为：

```text
sales_order.store_id   → store_info.store_id
sales_order.product_id → product_info.product_id
refund_record.order_id → sales_order.order_id
```

`refund_record` 不直接保存门店或商品字段，因此退款关联门店、商品时必须经过 `sales_order`。

## 时间消歧机制

Agent 使用只读查询动态获取：

```sql
SELECT MIN(order_date), MAX(order_date) FROM sales_order;
SELECT MIN(refund_date), MAX(refund_date) FROM refund_record;
```

由此构造 Temporal Context，并遵循：

1. 用户明确指定年份时，优先使用用户年份。
2. 用户只写 Q1/Q2 等季度表达，且相关数据只覆盖一个自然年份时，自动使用该年份。
3. 数据跨多个年份且用户未指定年份时，不允许模型随意猜测，应提示时间范围存在歧义。

当前数据库动态读取到的销售时间范围为 `2025-01-01 ～ 2025-06-30`，退款时间范围为 `2025-01-03 ～ 2025-07-12`。这些日期不是写死在 Prompt 中的常量。

## 项目目录

```text
bupt_data_agent/
├─ src/
│  └─ bupt_data_agent/
│     ├─ __init__.py
│     ├─ paths.py                # 项目资源路径
│     ├─ prepare_db.py           # Excel → SQLite 初始化与验证
│     ├─ agent.py                # Agent 主流程与 SQL 安全执行
│     ├─ business_validator.py   # 确定性业务规则校验
│     ├─ conversation.py         # 轻量对话上下文
│     ├─ evidence.py             # 查询依据提取
│     ├─ app.py                  # CLI 入口
│     └─ streamlit_app.py        # Streamlit 单页界面
├─ tests/
│  ├─ smoke_test.py              # 确定性 Golden SQL Benchmark
│  ├─ online_test.py             # 真实 LLM Core Benchmark
│  ├─ clarification_test.py
│  ├─ business_validator_test.py
│  └─ conversation_test.py
├─ evaluation/
│  ├─ evaluation_cases.json
│  ├─ evaluation_golden.py
│  ├─ evaluation_runner.py
│  ├─ evaluation_report.json
│  ├─ evaluation_report.md
│  └─ golden_results.json
├─ data/
│  ├─ store_info.xlsx
│  ├─ product_info.xlsx
│  ├─ sales_order.xlsx
│  ├─ refund_record.xlsx
│  └─ business.db                # prepare_db 生成
├─ knowledge/
│  ├─ business_terms.md
│  ├─ store_info.md
│  ├─ product_info.md
│  ├─ sales_order.md
│  └─ refund_record.md
├─ outputs/                      # 图表与在线测试报告
├─ pyproject.toml                # setuptools src-layout 配置
├─ requirements.txt
├─ .env.example                  # LLM 配置占位，不含真实 Key
└─ 题目说明_精简版.md
```

## 安装与运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

第二条命令以 editable 模式安装本项目，使 `bupt_data_agent` package
可从 `src/` 正常导入。

### 2. 配置模型

复制 `.env.example` 为 `.env`，填写 OpenAI-compatible API 配置：

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your_model_name
```

使用 OpenAI SDK 默认接口时可将 `LLM_BASE_URL` 留空；使用 DeepSeek 等兼容服务时填写服务商提供的地址。不要将真实 Key 写入 `.env.example`。

### 3. 初始化数据库

```bash
python -m bupt_data_agent.prepare_db
```

该命令从四张 Excel 创建 `data/business.db`，并验证行数、Schema、主外键、日期范围和数据库完整性。

### 4. 运行 CLI

```bash
python -m bupt_data_agent.app
```

### 5. 运行 Streamlit

```bash
streamlit run src/bupt_data_agent/streamlit_app.py
```

### 6. 运行确定性测试

```bash
python tests/smoke_test.py
python tests/clarification_test.py
python tests/business_validator_test.py
python tests/conversation_test.py
```

### 7. 运行真实 LLM 在线测试

```bash
python tests/online_test.py
```

在线测试需要有效的 API 配置，会依次执行五个自然语言问题，并将业务结果与 Golden Result 自动对账。

## 示例问题与验收状态

| # | 已验收示例 | 分析目标 |
| --- | --- | --- |
| 1 | 查询2025年上半年各门店的成交销额，并按销额降序排列。 | 门店销额排名 |
| 2 | 查询2025年上半年华东战区即时零售动销最好的3个SKU。 | 华东 O2O Top 3 SKU |
| 3 | 计算2025年上半年各门店退损率，找出超过5%的门店，并分析主要退款原因。 | 高退损门店及原因下钻 |
| 4 | 比较2025年Q1和Q2各门店成交销额，找出增长超过10%的门店。 | Q1/Q2 门店销额增长 |
| 5 | 找出Q2相比Q1成交销额增长但毛利率下降的门店，并分析是否可能存在低毛利SKU放量拖累。 | 门店筛选及 SKU 下钻 |

真实在线 Golden Benchmark：

```text
Q1 PASS
Q2 PASS
Q3 PASS
Q4 PASS
Q5 PASS
OVERALL PASS
```

## 代表性结果

### 华东即时零售 Top 3 SKU

```text
P008  24寸显示器
P007  激光打印机
P001  商务耳机
```

其中“华东战区”映射为 `region = '华东'`，“即时零售”映射为 `channel_code = 'O2O'`。

### 销额增长但毛利率下降

命中门店：

```text
S001
S002
S003
```

系统会继续按 SKU 下钻。数据中，`P007 激光打印机`、`P008 24寸显示器` 在 S001、S003 表现出较明显的低毛利放量迹象。这说明相关商品结构变化**可能**形成毛利拖累，但不能据此证明因果关系。

## 工程设计特点

1. 业务知识与数据库结构解耦，Markdown 可独立维护。
2. Markdown 实际参与动态语义理解，不只是说明文档。
3. LLM 负责语义理解、SQL生成和结果表述，确定性数值计算交给 SQLite。
4. 语句校验、只读连接、`query_only` 与 authorizer 形成多层安全保护。
5. Golden Benchmark 比较最终业务结果，而不是要求 SQL 文本完全相同。
6. Temporal Context 从数据库动态生成，减少季度年份猜测。
7. 确定性 Golden SQL 与真实 LLM 在线测试共同验证正确性。
8. CLI 与 Streamlit 共用同一个 Agent 主链，不复制业务逻辑。

## 技术栈

- Python
- SQLite
- pandas
- OpenAI-compatible LLM API
- matplotlib
- Streamlit
- openpyxl
- python-dotenv

项目未引入 LangChain、LangGraph、向量数据库或 Docker，保持最小可交付和易于演示。

## 安全说明

- `.env` 已加入 `.gitignore`，不会提交到 Git。
- `.env.example` 只保留配置占位，不包含真实 API Key。
- `data/business.db`、`outputs/` 和 Python 缓存文件也不会提交。
- 页面、CLI 和测试日志均不展示 API Key。
