# sales_order.xlsx 数据说明

`sales_order.xlsx` 为销售明细表，每行代表一笔商品销售记录。

`order_id`：订单编号；`order_date`：销售日期；`store_id`：门店编号；`product_id`：商品编号；`quantity`：销售数量；`sale_price`：实际销售单价（元）；`discount_amount`：整笔订单优惠金额（元）；`channel_code`：销售渠道编码。

渠道编码含义：

- `POS`：线下到店零售；
- `O2O`：即时零售；
- `B2B`：企业团购。

约定：

- `成交销额 = quantity × sale_price - discount_amount`
- 业务人员口语中的“销额”默认指“成交销额”
- `毛利额 = 成交销额 - quantity × unit_cost`
- `毛利率 = 毛利额 / 成交销额`

