"""
NPV vs T 完整分析流程主控脚本
一键运行批量求解和可视化分析
"""

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """
    运行Python脚本
    
    Args:
        script_path: 脚本路径
        description: 描述信息
    """
    print(f"\n🚀 {description}")
    print("="*60)
    
    try:
        # 激活conda环境并运行脚本
        cmd = f"conda run -n isru python {script_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ {description} 完成!")
        else:
            print(f"❌ {description} 失败!")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        return False
    
    return True

def main():
    """主函数：运行完整的NPV vs T分析流程"""
    
    print("NPV vs T 完整分析流程")
    print("=" * 60)
    print("本脚本将依次执行:")
    print("1. 批量求解 T=1到50年的ISRU优化问题")
    print("2. 生成NPV vs T的可视化分析")
    print("=" * 60)
    
    # 确认环境
    print("🔧 检查环境...")
    if not os.path.exists("data/parameters.json"):
        print("❌ 未找到参数文件: data/parameters.json")
        return
    
    # 切换到项目根目录
    os.chdir("/Users/lixiaoxiao/ISRU/isru_codebase")
    print("📁 工作目录:", os.getcwd())
    
    start_time = time.time()
    
    # 步骤1: 批量求解
    success = run_script("optimal/npv_time_analysis.py", "批量NPV分析")
    if not success:
        print("❌ 批量分析失败，停止执行")
        return
    
    # 步骤2: 可视化
    success = run_script("optimal/visualization.py", "生成可视化图表")
    if not success:
        print("❌ 可视化失败")
        return
    
    # 完成
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 NPV vs T 完整分析流程完成!")
    print(f"⏱️  总用时: {total_time:.2f} 秒")
    print("="*60)
    print("\n📋 生成的文件:")
    print("   • optimal/results/ - 批量求解结果CSV文件")
    print("   • charts/ - NPV vs T 可视化图表")
    print("\n💡 提示:")
    print("   • 可以查看 optimal/results/ 目录中的CSV文件获取详细数据")
    print("   • 可以查看 charts/ 目录中的PNG文件查看可视化结果")

if __name__ == "__main__":
    main()
