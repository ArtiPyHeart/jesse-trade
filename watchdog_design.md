# Jesse Watchdog 方案设计草案

## 目标
- 解决“进程未退出但交易逻辑停止”的假存活问题
- 出现断联/卡死时能自动恢复，无需人工重启
- 可观测、可验证、可回滚

## 现状痛点
- FastAPI 进程还在，但内部协程/交易循环已失活
- DB/WS 断联后，业务可能进入不可恢复状态
- 现有重启方案仅在进程崩溃时生效

## 约束
- 生产系统正确性优先，需可控、可审计
- 不依赖外部人工操作
- 与现有 Jesse 结构兼容（不破坏策略层）

## 推荐方案（强方案）
在 Jesse 内部实现心跳 + systemd WatchdogSec：

### 1) Jesse 内部心跳
设计一个轻量心跳模块，周期性更新“我还活着”的信号。建议包含三类信号：
- 交易循环心跳：核心策略 loop 的 tick 时间戳
- 数据流心跳：最新 candle 的时间戳 / websocket 最新消息时间
- DB 心跳：最近一次成功写入的时间戳（或轻量 SELECT）

心跳输出位置（推荐其一或组合）：
- 本地文件：`/root/jesse-trade/storage/health/heartbeat.json`
- Redis：`jesse:heartbeat:{session_id}`，TTL=10s
- PostgreSQL：表 `health_heartbeat`（只写一行）

关键点：
- 心跳必须从“真实业务路径”触发，不能只从 HTTP 线程触发
- 心跳更新失败应计数并写日志
- 心跳时间应使用 UTC，避免时区误差

### 2) systemd WatchdogSec
使用 systemd 的 watchdog 机制，只有在进程持续“喂狗”时才算存活：

- systemd unit 关键字段：
  - `Type=notify`
  - `WatchdogSec=30s`
  - `Restart=always`
  - `RestartSec=3s`

- Python 侧需周期性调用 `sd_notify("WATCHDOG=1")`
  - 可用 `python-systemd` / `systemd-python` 或直接写 socket

### 3) 心跳与 watchdog 的联动
规则示例：
- 只有当业务心跳“新鲜”(<= N 秒)时才 `WATCHDOG=1`
- 若发现心跳过期，直接停止喂狗，让 systemd 重启

这能保证：
- FastAPI 即使还在，但业务死锁或断联后无法继续喂狗，系统会自动重启

## 轻量替代方案（中方案）
单独写一个 watchdog 脚本（cron / systemd timer），每 X 秒检查：
- 最新 candle 是否推进
- DB 写入是否更新
- Websocket 是否有新消息

不满足条件则执行 `systemctl restart jesse`。

优点：实现快  
缺点：不是“内生心跳”，容易误判或漏判

## 落地步骤建议
1) 增加内部心跳模块（不影响交易逻辑）
2) 增加健康状态记录（文件或 Redis）
3) systemd 服务改为 `Type=notify` + `WatchdogSec`
4) 业务心跳过期时停止喂狗，触发重启
5) 线上验证：模拟断联/阻塞，确认能重启

## 验证指标
- 心跳文件每秒更新（或 Redis TTL 保持）
- DB/WS 断开时触发重启
- 重启后策略恢复
- 重启次数有明确日志记录

## 预期风险
- 心跳误判导致频繁重启（需合理阈值）
- sd_notify 依赖缺失（需加入环境依赖）
- 多策略并发时需区分 session_id

## 后续可选增强
- 引入“状态机”区分 Warmup / Trading
- 将健康状态暴露给 Prometheus
- 把断联错误计数与报警接入外部监控
