# jesse-trade 开发指南

## 核心原则
- **算法正确性**：计算错误可能造成重大财务损失，确保正确性优先于优化。
- **生产标准**：维持生产就绪代码质量，提交前必须审查。
- **科学方法**：应用先进数学/物理概念于交易。
- **MBTI人格化**：使用INTJ人格构思架构，思索科学问题；使用ISTJ人格执行具体开发任务。

## 项目结构
- `src/`：生产代码（bars/features/indicators/utils）
- `strategies/`：Jesse策略（每策略独立目录）- **项目运行入口**
- `research/`：离线研究实验（勿在生产导入）
- `extern/`：参考资料（勿导入）
- `tests/`：pytest测试

## 运行入口
该项目的运行入口为 `strategies/` 下的各项策略，需要启动jesse后运行，无法直接使用python运行

## 开发环境
```bash
./install.sh                      # 生产依赖
pip install -r requirements-dev.txt  # 开发依赖
```

## Jesse K线规范
- 格式：6列NumPy数组 `[timestamp, open, close, high, low, volume]`
- 转换：`numpy_candles_to_dataframe(candles)`
- 自定义K线：Dollar/Range/Entropy Bar，DEAP符号回归

## 指标开发
- 位置：稳定→`src/indicators/prod/`，实验→`experimental/`
- 规范：`sequential=True`返回全序列，`False`返回最新值
- 长度：用`np.nan`填充保持与K线一致
- 类指标：继承`_cls_ind.py`基类

## 编码规范
- 内部函数用`_`前缀
- 数据操作用NumPy/Pandas
- 使用`assert`拦截非法输入，避免宽泛`try/except`
- 简单测试用`if __name__ == "__main__"`，复杂测试放`tests/`
- EasyLanguage角度→Python弧度：用`src/utils/math_tools.py`

## 关键提醒
- 开发时使用mcp context7 / exa 查看最新文档，如果mcp调用失败，停下来提示用户先配置
- 生产代码仅从`src/`导入
- 策略间保持独立，避免交叉依赖
- 功能实现后必须单元测试
- 测试文件命名：pytest用`test_`开头，直接运行避免`test_`前缀