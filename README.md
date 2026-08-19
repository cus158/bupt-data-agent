# 智能数据分析 Agent

> 基于业务知识与数据库 Schema 的自然语言 Text-to-SQL 数据分析系统

这是一个面向业务数据分析场景的轻量级 LLM Agent。用户无需编写 SQL，只需输入中文问题，系统便会结合 Markdown 业务知识、SQLite Schema 和数据库时间范围理解业务语义，生成并安全执行只读 SQL，最终返回数据表格、自然语言结论和必要的可视化图表。

它不仅调用一次模型，而是完成“构建业务上下文 → 生成 SQL → 安全校验 → 查询数据 → 分析结果”的完整闭环，因此具备一个数据分析 Agent 的基本工作流。

## 项目背景

项目包含四类业务数据：

| 数据表          | 含义                             |
| --------------- | -------------------------------- |
| `store_info`    | 门店及所属区域、城市等信息       |
| `product_info`  | 商品、品类、成本和标价信息       |
| `sales_order`   | 销售订单、渠道、数量、售价和折扣 |
| `refund_record` | 退款记录、退款金额和退款原因     |

`knowledge/` 中的 `business_terms.md` 和各数据表说明 Markdown 定义了字段含义、业务术语及指标口径，使 Agent 能理解“华东战区”“即时零售”“SKU”“成交销额”“退损率”“毛利率”“放量”等业务表达，并将其转换为正确的数据查询。

## 核心功能

1. **自然语言数据查询**：直接使用中文提出门店、商品、销售、退款和利润分析问题。
2. **Markdown 业务知识注入**：处理查询时读取 `knowledge/*.md` 并作为 Prompt Context 注入，将业务术语映射到实际字段和 SQL 条件。例如：
   - 战区 → `store_info.region`
   - 即时零售 → `sales_order.channel_code = 'O2O'`
   - SKU → `product_info.product_id`
3. **动态 Schema 感知**：通过 SQLite `PRAGMA table_info` 和外键信息读取真实数据库结构，而非在 Prompt 中写死 Schema。
4. **Text-to-SQL 与 SemanticPlan**：LLM 综合“用户问题 + 业务知识 + Schema + Temporal Context + 可选对话上下文”，在同一次调用中返回结构化问题理解和 SQLite 兼容的只读 SQL。
5. **歧义识别与主动澄清**：当指标或指代无法唯一确定时返回简洁澄清问题，不生成或执行 SQL；能由业务知识和数据库上下文唯一确定时自动消歧。
6. **SQL 安全执行**：只允许 `SELECT` 或合法的 `WITH ... SELECT`，拒绝写操作、DDL、管理语句和多语句执行；数据库同时使用只读 URI、`query_only` 和 SQLite authorizer，并限制查询结果行数。LLM 不会生成并执行任意 Python。
7. **Business Rule Validator**：在 SQL Safety 之后确定性检查成交销额、汇总毛利率、退款时间、退损率等高置信度业务规则错误。
8. **有限错误恢复**：SQL 安全、业务规则或执行阶段失败时共享一次 repair budget，修复后的 SQL 会重新经过安全与业务规则校验。
9. **轻量多轮上下文**：从真实 DataFrame 提取上一轮门店、SKU、时间和指标，仅在当前问题包含指代或省略时补全条件；聊天记录独立持久化到本地 `chat_history.db`。
10. **按需可视化**：查询结果以 pandas DataFrame 返回；用户明确要求绘图时生成 `bar`、`line` 或 `pie`，未提出绘图需求时默认不画图。
11. **自然语言总结**：模型只能依据实际 SQL 结果总结。无法证明因果关系时，使用“可能”“数据显示”“存在拖累迹象”等谨慎表达。
12. **查询依据展示**：展示实际使用的数据表、业务口径、时间范围、筛选聚合和校验状态，不公开 Chain-of-Thought。

> `business_terms.md` 中保留了题目描述使用的 `dim_store` / `dim_product` 名称；Agent 会明确纠正为真实表名 `store_info` / `product_info`。

## 业务口径

| 指标     | 定义                                      |
| -------- | ----------------------------------------- |
| 成交销额 | `quantity * sale_price - discount_amount` |
| 销量     | `SUM(quantity)`                           |
| 毛利额   | `成交销额 - quantity * unit_cost`         |
| 毛利率   | `SUM(毛利额) / SUM(成交销额)`             |
| 退损金额 | `SUM(refund_amount)`                      |
| 退损率   | `退款金额 / 成交销额`                     |
| 净销额   | `成交销额 - 退款金额`                     |

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
│ Conversation Resolver │
│ 指代解析 / 主动澄清   │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Agent Context Builder │
│ Markdown + Schema     │
│ Temporal Context      │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│          LLM          │
│ SemanticPlan + SQLPlan│
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│   SQL Safety Layer    │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│Business Rule Validator│
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ SQLite Read-only Query│
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ DataFrame + Summary   │
│ Chart + Evidence      │
└───────────┬───────────┘
            ▼
 Streamlit / CLI 分析结果
```

- **Conversation Resolver**：只使用最近一轮结构化上下文解决“它”“这些门店”等追问；无法唯一解析时先澄清。
- **Context Builder**：加载 Markdown、真实表结构、外键和动态日期范围。
- **LLM Text-to-SQL**：理解问题并在一次调用中输出 SemanticPlan、SQLPlan 和图表类型。
- **SQL Safety Layer**：完成语句类型、危险关键字、多语句和 SQLite authorizer 检查。
- **Business Rule Validator**：确定性检查 SQL 是否明显违反业务指标口径。
- **SQLite / pandas**：负责确定性的数值计算和结构化结果承载。
- **结果层**：基于真实结果生成谨慎结论、按需图表和可审计 Evidence，并由 Streamlit 或 CLI 展示。

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
│     ├─ chat_history.py         # 本地持久化聊天历史
│     ├─ evidence.py             # 查询依据提取
│     ├─ app.py                  # CLI 入口
│     └─ streamlit_app.py        # Streamlit 单页界面
├─ tests/
│  ├─ smoke_test.py              # 确定性 Golden SQL Benchmark
│  ├─ online_test.py             # 真实 LLM Core Benchmark
│  ├─ clarification_test.py
│  ├─ business_validator_test.py
│  ├─ conversation_test.py
│  ├─ semantic_plan_test.py
│  ├─ chat_history_test.py
│  └─ streamlit_app_test.py
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
│  ├─ business.db                # prepare_db 生成的业务数据库
│  └─ chat_history.db            # Streamlit 自动生成的聊天历史库
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
python tests/semantic_plan_test.py
python tests/chat_history_test.py
python tests/streamlit_app_test.py
```

### 7. 运行真实 LLM 在线测试

```bash
python tests/online_test.py
```

在线测试需要有效的 API 配置，会依次执行五个自然语言问题，并将业务结果与 Golden Result 自动对账。

### 8. 扩展评测

```bash
python evaluation/evaluation_runner.py
```

扩展评测包含在线自然语言案例和确定性离线案例。若只需运行离线部分或重新生成已有报告，可使用：

```bash
python evaluation/evaluation_runner.py --offline-only
python evaluation/evaluation_runner.py --report-only
```

## 示例问题与验收状态

| #   | 已验收示例                                                                                                                                          | 分析目标                     |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 1   | 查询 2025 年上半年每家门店的销售额，从高到低排序，并画一个销售额柱状图。                                                                            | 门店销额排名及柱状图         |
| 2   | 查询 2025 年上半年华东战区即时零售渠道动销最好的 3 个 SKU，按销额排序，并给出每个 SKU 的销量、销额和所属品类。                                      | 华东 O2O Top 3 SKU           |
| 3   | 查询各门店 2025 年上半年的退损情况，找出退损率超过 5% 的门店，画出各门店退损率对比图，并分析退损率较高门店的主要退款原因。                          | 高退损门店、对比图及原因下钻 |
| 4   | 比较每家门店 2025 年第一季度和第二季度的销售额，找出第二季度销售额比第一季度增长超过 10% 的门店，并生成两个季度销售额对比图。                       | Q1/Q2 门店销额增长及对比图   |
| 5   | 找出 2025 年第二季度销额比第一季度增长超过 10%，但毛利率下降的门店。进一步分析这些门店是否存在低毛利 SKU 放量导致整体毛利率下降，并生成合适的图表。 | 门店筛选、SKU 下钻及图表     |

真实在线 Golden Benchmark：

```text
Q1 PASS
Q2 PASS
Q3 PASS
Q4 PASS
Q5 PASS
OVERALL PASS
```

扩展评测报告：

```text
Core Benchmark：5 / 5 PASS
Extended Evaluation：24 / 24 PASS
```

以上结果仅代表当前固定评测集，不表示对任意自然语言问题都能达到 100% 准确率。详细结果见 `evaluation/evaluation_report.md`。

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
S003
```

系统会继续按 SKU 下钻。数据中，`P007 激光打印机`、`P008 24寸显示器` 在 S001、S003 表现出较明显的低毛利放量迹象。这说明相关商品结构变化**可能**形成毛利拖累，但不能据此证明因果关系。

## 技术栈

- Python
- SQLite
- pandas
- OpenAI-compatible LLM API
- matplotlib
- Streamlit
- openpyxl
- python-dotenv
