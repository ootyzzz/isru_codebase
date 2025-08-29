# ISRU策略仿真系统 - 重构版本

## 🎯 系统概述

这是一个完全重构的ISRU（原位资源利用）策略仿真系统，用于分析和比较不同部署策略在月球氧气生产中的表现。

### 🔄 重构亮点

- ✅ **真正的策略仿真**：基于规则驱动，非优化求解
- ✅ **策略差异化**：三种策略显示明显不同的性能特征
- ✅ **美观的终端输出**：进度条、表格、图表等可视化
- ✅ **全面的性能分析**：财务、运营、风险三维度评估
- ✅ **批量仿真支持**：蒙特卡洛、时间跨度、并行分析

## 📁 项目结构

```
strategies/
├── core/                          # 核心仿真引擎
│   ├── strategy_definitions.py    # 策略参数定义
│   ├── simulation_engine.py       # 仿真引擎
│   ├── decision_logic.py          # 决策逻辑
│   └── state_manager.py           # 状态管理
├── analysis/                      # 分析工具
│   ├── batch_runner.py            # 批量仿真执行器
│   └── performance_analyzer.py    # 性能分析器
├── utils/                         # 工具模块
│   └── terminal_display.py        # 终端显示工具
├── visualization/                 # 可视化模块
│   ├── strategy_visualizer.py     # 策略可视化器
│   └── example_usage.py           # 使用示例
├── simulation_results/            # 仿真结果存储
│   ├── raw/                       # 原始数据（按时间跨度分类）
│   ├── summary/                   # 统计摘要
│   ├── reports/                   # 分析报告
│   └── exports/                   # 导出文件
└── main.py                        # 主程序入口
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate isru

# 确保在项目根目录
cd c:/Users/feifa/Documents/lgy/isru_codebase
```

### 2. 推荐使用示例

```bash
# 🔥 最常用：运行50年时间跨度的策略对比仿真并显示可视化图表
python strategies/main.py --time-horizon 20 --visualize --n-simulations 100

# 快速测试：运行10年时间跨度的策略对比仿真（默认）
python strategies/main.py --visualize

# 单个策略的蒙特卡洛仿真
python strategies/main.py monte-carlo --strategy aggressive --time-horizon 30 --n-simulations 500 --visualize

# 策略对比分析并保存结果
python strategies/main.py compare --time-horizon 25 --n-simulations 200 --visualize --save

# 显示已有结果的可视化图表
python strategies/main.py visualize

# 导出结果到Excel
python strategies/main.py results export --strategies conservative aggressive moderate --time-horizons 10 20 30
```

### 3. 基本命令

```bash
# 查看所有可用命令
python strategies/main.py --help

# 策略对比（推荐开始）
python strategies/main.py compare

# 时间跨度分析
python strategies/main.py horizon

# 单次仿真
python strategies/main.py single --strategy moderate
```

## 📊 主要功能

### 1. 策略对比分析

比较三种策略在相同条件下的表现：

```bash
# 基本对比
python strategies/main.py compare --time-horizon 30 --n-simulations 100

# 指定策略对比
python strategies/main.py compare --strategies conservative aggressive --n-simulations 50

# 详细分析报告
python strategies/main.py compare --detailed-analysis --save
```

**输出示例：**
```
📊 T=10年策略对比
┌────────────────────┬───────────────┬───────────────┬───────────────┐
│ 指标                │  Conservative │    Aggressive │      Moderate │
├────────────────────┼───────────────┼───────────────┼───────────────┤
│ NPV均值             │          1.3M │          1.7M │          1.7M │
│ NPV标准差           │        327.1K │        524.9K │        509.6K │
│ 平均利用率          │         87.4% │         53.5% │         65.8% │
│ 自给自足率          │         72.4% │         84.2% │         83.3% │
└────────────────────┴───────────────┴───────────────┴───────────────┘
```

### 2. 时间跨度影响分析

分析不同时间跨度对策略表现的影响：

```bash
# 标准时间跨度分析
python strategies/main.py horizon --time-horizons 10 20 30 40 50

# 自定义时间跨度
python strategies/main.py horizon --time-horizons 15 25 35 --strategies moderate aggressive
```

### 3. 蒙特卡洛仿真

对单个策略进行大量随机仿真：

```bash
# 蒙特卡洛仿真
python strategies/main.py monte-carlo --strategy conservative --n-simulations 1000

# 保存结果
python strategies/main.py monte-carlo --strategy aggressive --n-simulations 500 --save
```

### 4. 并行批量仿真

高效执行大规模仿真：

```bash
# 并行仿真所有策略和时间跨度
python strategies/main.py parallel --n-simulations 100

# 指定并行进程数
python strategies/main.py parallel --max-workers 4 --time-horizons 10 20 30
```

## 🎯 三种策略详解

### 1. Conservative（保守策略）
- **特点**：谨慎扩张，重视成本控制
- **初始部署**：60%的预期需求
- **扩张触发**：85%利用率
- **扩张幅度**：25%
- **适用场景**：资源有限，风险厌恶

### 2. Aggressive（激进策略）
- **特点**：快速扩张，追求市场占有
- **初始部署**：120%的预期需求
- **扩张触发**：70%利用率
- **扩张幅度**：50%
- **适用场景**：资源充足，追求增长

### 3. Moderate（温和策略）
- **特点**：平衡扩张，稳健发展
- **初始部署**：90%的预期需求
- **扩张触发**：80%利用率
- **扩张幅度**：35%
- **适用场景**：平衡风险与收益

## 📈 性能指标说明

### 财务指标
- **NPV（净现值）**：项目的财务价值
- **NPV标准差**：收益的波动性
- **成功率**：NPV为正的概率
- **夏普比率**：风险调整后收益

### 运营指标
- **平均利用率**：产能使用效率
- **自给自足率**：本地生产占总需求比例
- **产能扩张次数**：策略的扩张频率
- **最终产能**：项目结束时的总产能

### 风险指标
- **波动率**：NPV的标准差
- **下行风险**：负收益的风险
- **最大回撤**：最大损失幅度
- **索提诺比率**：下行风险调整收益

## 🔧 高级用法

### 1. 自定义参数

修改 `data/parameters.json` 中的参数：

```json
{
  "demand": {
    "D0": 10,        // 初始需求
    "mu": 0.2,       // 需求增长率
    "sigma": 0.2     // 需求波动率
  },
  "costs": {
    "c_dev": 10000,  // 开发成本
    "c_op": 3000,    // 运营成本
    "c_E": 20000     // 地球补给成本
  }
}
```

### 2. 结果文件

仿真结果自动保存在 `strategies/simulation_results/` 目录：

```
simulation_results/
├── T10/
│   ├── conservative_detailed.json    # 详细仿真数据
│   ├── conservative_summary.json     # 统计摘要
│   └── ...
├── T20/
└── ...
```

### 3. 与全局最优解对比

```bash
# 与test_fixed_model.py的最优解对比
python strategies/main.py optimal --time-horizon 30
```

## 🎨 可视化功能

系统提供两种可视化方式：

### 终端可视化
- **进度条**：实时显示仿真进度
- **表格**：结构化显示对比结果
- **摘要框**：突出显示关键指标
- **ASCII图表**：简单的趋势可视化

### 图形可视化（新增）
使用 `--visualize` 参数启用图形可视化：

```bash
# 启用可视化图表
python strategies/main.py --visualize --time-horizon 50 --n-simulations 100
```

**可视化图表包括：**
1. **决策变量对比图**：生产量、产能、库存、地球供应、产能扩张、利用率
2. **需求与供应对比图**：需求曲线与各策略供应能力对比
3. **成本分析图**：成本构成堆叠图和NPV对比图

**图表特点：**
- 支持中文显示
- 三种策略用不同颜色区分
- 交互式图表，可放大缩小
- 自动适配时间跨度（T10, T20, T30, T40, T50）

## 🔍 故障排除

### 常见问题

1. **ImportError: No module named 'seaborn'**
   - 解决：可视化模块是可选的，不影响核心功能

2. **策略结果相同**
   - 检查：确保使用新的仿真系统，不是旧的strategy_runner.py

3. **内存不足**
   - 解决：减少仿真次数或使用并行仿真

### 性能优化

```bash
# 减少仿真次数进行快速测试
python strategies/main.py compare --n-simulations 20

# 使用并行处理提高效率
python strategies/main.py parallel --max-workers 8
```

## 📚 技术架构

### 核心设计原则

1. **规则驱动**：策略基于预定义规则执行，非优化求解
2. **状态管理**：清晰的状态转换和历史记录
3. **模块化**：核心组件独立，易于扩展
4. **可视化**：美观的终端输出和进度显示

### 扩展指南

添加新策略：

```python
# 在 strategies/core/strategy_definitions.py 中添加
@staticmethod
def get_new_strategy() -> StrategyParams:
    return StrategyParams(
        name="new_strategy",
        description="新策略描述",
        initial_deployment_ratio=0.8,
        utilization_threshold=0.75,
        expansion_ratio=0.3,
        risk_tolerance=0.6,
        cost_sensitivity=0.4
    )
```

## 🎯 使用建议

### 研究场景

1. **策略选择**：使用 `compare` 命令比较策略
2. **时间规划**：使用 `horizon` 分析长期影响
3. **风险评估**：关注NPV标准差和成功率
4. **敏感性分析**：修改参数后重新仿真

### 最佳实践

1. **先小后大**：从小规模仿真开始测试
2. **保存结果**：重要分析使用 `--save` 参数
3. **多次运行**：使用不同随机种子验证结果
4. **文档记录**：记录参数设置和分析结论

## 📞 支持

如有问题或建议，请检查：

1. 参数文件格式是否正确
2. 环境依赖是否满足
3. 文件路径是否正确
4. 仿真参数是否合理

---

**重构完成时间**：2025年8月27日  
**版本**：2.0  
**状态**：生产就绪 ✅