"""
Example usage of DeepSSM JAX implementation
DeepSSM JAX实现的使用示例
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.deep_ssm import (
    DeepSSMRealTime,
)
from src.models.deep_ssm.training import train_from_csv


def example_complete_workflow():
    """
    完整工作流程示例：从CSV训练到特征生成
    """
    print("=" * 60)
    print("DeepSSM完整工作流程示例")
    print("=" * 60)

    # 配置参数
    config = {
        "csv_path": "extern/DeepSSM/np_fracdiff_features.csv",  # 输入数据
        "model_path": "deep_ssm_model_jax.pkl",  # 模型保存路径
        "feature_save_path": "deep_ssm_features_jax.csv",  # 特征保存路径
        "state_dim": 5,  # 潜在状态维度（输出特征数）
        "lstm_hidden": 64,  # LSTM隐藏层维度
        "max_epochs": 50,  # 最大训练轮数
        "patience": 5,  # 早停耐心值
        "learning_rate": 0.001,  # 学习率
    }

    print("\n配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    try:
        # 执行训练和特征生成
        model, params, features = train_from_csv(
            csv_path=config["csv_path"],
            model_save_path=config["model_path"],
            feature_save_path=config["feature_save_path"],
            state_dim=config["state_dim"],
            lstm_hidden=config["lstm_hidden"],
            max_epochs=config["max_epochs"],
            patience=config["patience"],
            learning_rate=config["learning_rate"],
        )

        print("\n✅ 训练完成!")
        print(f"  生成的特征形状: {features.shape}")
        print(f"  模型已保存到: {config['model_path']}")
        print(f"  特征已保存到: {config['feature_save_path']}")

    except FileNotFoundError:
        print("\n⚠️ 找不到输入数据文件，使用合成数据演示...")
        example_with_synthetic_data()


def example_with_synthetic_data():
    """
    使用合成数据的示例
    """
    print("\n" + "=" * 60)
    print("使用合成数据的DeepSSM示例")
    print("=" * 60)

    # 生成合成时间序列数据
    T = 1000  # 时间步数
    obs_dim = 20  # 观测维度

    # 创建具有不同频率成分的合成数据
    t = np.linspace(0, 10 * np.pi, T)
    data = []
    for i in range(obs_dim):
        signal = (
            np.sin(t * (i + 1) * 0.1)  # 不同频率的正弦波
            + 0.5 * np.cos(t * (i + 1) * 0.05)  # 余弦成分
            + 0.1 * np.random.randn(T)  # 噪声
        )
        data.append(signal)

    synthetic_data = np.column_stack(data)

    print(f"\n生成的合成数据:")
    print(f"  形状: {synthetic_data.shape}")
    print(f"  时间步: {T}")
    print(f"  特征维度: {obs_dim}")

    # 保存合成数据
    df = pd.DataFrame(synthetic_data, columns=[f"feature_{i}" for i in range(obs_dim)])
    df.to_csv("synthetic_data.csv", index=False)
    print(f"  已保存到: synthetic_data.csv")

    # 训练模型
    print("\n开始训练模型...")
    model, params, features = train_from_csv(
        csv_path="synthetic_data.csv",
        model_save_path="synthetic_model.pkl",
        feature_save_path="synthetic_features.csv",
        state_dim=5,
        lstm_hidden=32,
        max_epochs=30,
        patience=5,
        learning_rate=0.001,
    )

    print("\n✅ 合成数据训练完成!")
    print(f"  提取的特征形状: {features.shape}")

    # 展示特征统计
    feature_df = pd.read_csv("synthetic_features.csv")
    print("\n特征统计信息:")
    print(feature_df.describe())

    return model, params, features


def example_realtime_processing():
    """
    实时处理示例
    """
    print("\n" + "=" * 60)
    print("DeepSSM实时处理示例")
    print("=" * 60)

    # 尝试加载已训练的模型
    try:
        processor = DeepSSMRealTime("deep_ssm_model_jax.pkl")
        print("✓ 从文件加载模型成功")
    except:
        print("使用新初始化的模型进行演示")
        from src.models.deep_ssm.inference import create_realtime_processor

        processor = create_realtime_processor(obs_dim=77, state_dim=5)

    print(f"\n模型配置:")
    print(f"  输入维度: {processor.obs_dim}")
    print(f"  输出特征维度: {processor.state_dim}")

    # 模拟实时数据流
    print("\n模拟实时数据处理:")
    print("-" * 40)

    window_size = 10
    features_buffer = []

    for i in range(window_size):
        # 模拟新数据到达
        new_data = np.random.randn(processor.obs_dim)

        # 处理数据获取特征
        feature = processor.process(new_data)
        features_buffer.append(feature)

        # 显示前几步的结果
        if i < 3:
            print(f"时间步 {i+1}:")
            print(
                f"  输入数据统计: mean={new_data.mean():.3f}, std={new_data.std():.3f}"
            )
            print(f"  生成特征: {feature[:3].round(3)}...")

    # 计算窗口统计
    features_array = np.array(features_buffer)
    print(f"\n窗口统计 (最近{window_size}步):")
    print(f"  特征均值: {features_array.mean(axis=0).round(3)}")
    print(f"  特征标准差: {features_array.std(axis=0).round(3)}")

    # 演示状态管理
    print("\n状态管理演示:")
    current_state = processor.get_state()
    print(f"  当前状态已保存")

    # 处理更多数据
    for _ in range(5):
        processor.process(np.random.randn(processor.obs_dim))
    print(f"  处理了5步新数据")

    # 恢复状态
    processor.set_state(current_state)
    print(f"  状态已恢复到之前的检查点")

    # 重置处理器
    processor.reset()
    print(f"  处理器已重置到初始状态")


def example_batch_processing():
    """
    批处理示例
    """
    print("\n" + "=" * 60)
    print("DeepSSM批处理示例")
    print("=" * 60)

    from src.models.deep_ssm.inference import create_realtime_processor

    # 创建处理器
    obs_dim = 20
    state_dim = 5
    processor = create_realtime_processor(obs_dim, state_dim)

    # 准备批量数据
    batch_size = 100
    batch_data = np.random.randn(batch_size, obs_dim)

    print(f"批处理配置:")
    print(f"  批大小: {batch_size}")
    print(f"  输入维度: {obs_dim}")
    print(f"  输出维度: {state_dim}")

    # 执行批处理
    import time

    start_time = time.time()
    batch_features = processor.process_batch(batch_data)
    process_time = time.time() - start_time

    print(f"\n批处理结果:")
    print(f"  输出形状: {batch_features.shape}")
    print(f"  处理时间: {process_time:.3f}秒")
    print(f"  处理速度: {batch_size/process_time:.1f}样本/秒")

    # 显示批处理特征统计
    print(f"\n批特征统计:")
    print(f"  均值: {batch_features.mean(axis=0).round(3)}")
    print(f"  标准差: {batch_features.std(axis=0).round(3)}")
    print(f"  最小值: {batch_features.min(axis=0).round(3)}")
    print(f"  最大值: {batch_features.max(axis=0).round(3)}")


def main():
    """
    主函数：运行所有示例
    """
    print("🚀 DeepSSM JAX实现使用示例")
    print("=" * 60)

    # 选择要运行的示例
    print("\n请选择要运行的示例:")
    print("1. 完整工作流程（需要真实数据）")
    print("2. 合成数据示例")
    print("3. 实时处理示例")
    print("4. 批处理示例")
    print("5. 运行所有示例")

    choice = input("\n请输入选项 (1-5): ").strip()

    if choice == "1":
        example_complete_workflow()
    elif choice == "2":
        example_with_synthetic_data()
    elif choice == "3":
        example_realtime_processing()
    elif choice == "4":
        example_batch_processing()
    elif choice == "5":
        example_with_synthetic_data()
        example_realtime_processing()
        example_batch_processing()
    else:
        print("无效选项，运行默认示例...")
        example_with_synthetic_data()

    print("\n" + "=" * 60)
    print("✅ 示例运行完成!")


if __name__ == "__main__":
    main()
