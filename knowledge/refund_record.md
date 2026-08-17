# refund_record.xlsx 数据说明

`refund_id`：退款记录编号；`order_id`：原订单编号；`refund_date`：退款日期；`refund_quantity`：退款数量；`refund_amount`：退款金额（元）；`refund_reason`：退款原因。

业务人员有时将退款相关损失简称为“退损”。

统一约定：

- `退损金额 = refund_amount`
- `退损率 = 退款金额 / 成交销额`
- `净销额 = 成交销额 - 退款金额`

统计某时间段“退损”时，以 `refund_date` 判断退款发生时间。

