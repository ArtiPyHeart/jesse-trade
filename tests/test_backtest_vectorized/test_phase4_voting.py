"""
Phase 4: 投票聚合测试（deque 对齐）

测试目标:
- aggregate_votes 按同一时刻投票，不做手工 N 对齐
- 预测输出含 0 时应输出 flat
- 输出长度与输入一致

运行方式:
    pytest tests/test_backtest_vectorized/test_phase4_voting.py -v
"""

import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 导入被测函数
from backtest_no_jesse import aggregate_votes  # noqa: E402


class TestAggregateVotesDequeAligned:
    """测试 deque 对齐后的投票逻辑"""

    def test_single_model_direct(self):
        predictions = {"c_L5_N1": [1, -1, 0, 1]}
        models = ["c_L5_N1"]

        signals = aggregate_votes(predictions, models)

        assert signals == ["long", "short", "flat", "long"]

    def test_multi_model_with_warmup_zero(self):
        predictions = {
            "c_L5_N1": [1, 1, -1, 1],
            "c_L5_N3": [0, 0, 1, 1],
        }
        models = ["c_L5_N1", "c_L5_N3"]

        signals = aggregate_votes(predictions, models)

        assert signals == ["flat", "flat", "flat", "long"]


class TestAggregateVotesVoting:
    """测试投票逻辑"""

    def test_all_long_votes(self):
        predictions = {
            "c_L5_N1": [1, 1, 1, 1, 1],
            "r_L5_N1": [1, 1, 1, 1, 1],
        }
        models = ["c_L5_N1", "r_L5_N1"]

        signals = aggregate_votes(predictions, models)

        assert all(s == "long" for s in signals)

    def test_all_short_votes(self):
        predictions = {
            "c_L5_N1": [-1, -1, -1, -1, -1],
            "r_L5_N1": [-1, -1, -1, -1, -1],
        }
        models = ["c_L5_N1", "r_L5_N1"]

        signals = aggregate_votes(predictions, models)

        assert all(s == "short" for s in signals)

    def test_mixed_votes_give_flat(self):
        predictions = {
            "c_L5_N1": [1, 1, 1, 1, 1],
            "r_L5_N1": [-1, -1, -1, -1, -1],
        }
        models = ["c_L5_N1", "r_L5_N1"]

        signals = aggregate_votes(predictions, models)

        assert all(s == "flat" for s in signals)

    def test_zero_votes_give_flat(self):
        predictions = {
            "c_L5_N1": [1, 0, 1, 0, 1],
            "r_L5_N1": [1, 1, 1, 1, 1],
        }
        models = ["c_L5_N1", "r_L5_N1"]

        signals = aggregate_votes(predictions, models)

        assert signals == ["long", "flat", "long", "flat", "long"]


class TestAggregateVotesEdgeCases:
    """测试边界条件"""

    def test_empty_predictions(self):
        predictions = {"c_L5_N1": []}
        models = ["c_L5_N1"]

        with pytest.raises(ZeroDivisionError):
            aggregate_votes(predictions, models)

    def test_output_length_matches_input(self):
        n_samples = 100
        predictions = {
            "c_L5_N1": [1] * n_samples,
            "c_L5_N2": [-1] * n_samples,
        }
        models = ["c_L5_N1", "c_L5_N2"]

        signals = aggregate_votes(predictions, models)

        assert len(signals) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
