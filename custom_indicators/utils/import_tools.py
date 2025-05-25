import subprocess
import sys
from importlib import import_module


def ensure_package(package_name: str):
    """确保包已安装，如果未安装则自动安装"""
    try:
        import_module(package_name)
    except ImportError:
        print(f"正在安装 {package_name} 包...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            print(f"{package_name} 安装成功！")
        except subprocess.CalledProcessError as e:
            raise ImportError(f"无法安装 {package_name} 包。错误信息: {str(e)}")
