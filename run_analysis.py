#!/usr/bin/env python3
"""
NPV vs T 分析 - 快速启动脚本
"""

import os
import sys
import subprocess

def main():
    print("🚀 ISRU NPV vs T 分析工具")
    print("=" * 50)
    print("这个工具将执行:")
    print("1. 批量求解 T=1到50年的ISRU优化问题")
    print("2. 生成NPV vs T的可视化分析图表")
    print("=" * 50)
    
    # 确保在正确的目录
    project_root = "/Users/lixiaoxiao/ISRU/isru_codebase"
    if not os.path.exists(os.path.join(project_root, "optimal")):
        print(f"❌ 错误: 请确认项目目录存在: {project_root}")
        return
    
    os.chdir(project_root)
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 检查conda环境
    try:
        result = subprocess.run("conda info --envs | grep isru", shell=True, capture_output=True, text=True)
        if "isru" not in result.stdout:
            print("❌ 错误: 未找到conda环境 'isru'")
            print("请先运行: conda env create -f environment.yml")
            return
    except Exception:
        print("⚠️  无法检查conda环境，继续运行...")
    
    print("\n🎯 选择运行模式:")
    print("1. 完整分析 (T=1到50，约1分钟)")
    print("2. 快速测试 (T=1到10，约10秒)")
    print("3. 仅可视化 (使用已有数据)")
    
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("请输入 1, 2 或 3")
    
    if choice == '1':
        # 完整分析
        print("\n🚀 开始完整分析...")
        cmd = "conda run -n isru python optimal/npv_time_analysis.py"
        os.system(cmd)
        
        print("\n📈 生成可视化...")
        cmd = "conda run -n isru python optimal/visualization.py"
        os.system(cmd)
        
    elif choice == '2':
        # 快速测试
        print("\n🚀 开始快速测试...")
        # 创建临时测试脚本
        test_code = """
from optimal.npv_time_analysis import batch_solve_npv_analysis
df, csv_file = batch_solve_npv_analysis(t_min=1, t_max=10, random_seed=42)
print(f"\\n✅ 测试完成! 结果保存到: {csv_file}")
"""
        with open("temp_test.py", "w") as f:
            f.write(test_code)
        
        cmd = "conda run -n isru python temp_test.py"
        os.system(cmd)
        
        print("\n📈 生成可视化...")
        cmd = "conda run -n isru python optimal/visualization.py"
        os.system(cmd)
        
        # 清理临时文件
        os.remove("temp_test.py")
        
    elif choice == '3':
        # 仅可视化
        print("\n📈 生成可视化...")
        cmd = "conda run -n isru python optimal/visualization.py"
        os.system(cmd)
    
    print("\n🎉 分析完成!")
    print("📋 生成的文件:")
    print("   • optimal/results/ - 批量求解结果CSV文件")
    print("   • charts/ - NPV vs T 可视化图表")

if __name__ == "__main__":
    main()
