#!/usr/bin/env python3
"""
Terminal Display Tools
Provide beautiful terminal output including progress bars, tables, charts, etc.
"""

import sys
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TableColumn:
    """Table column definition"""
    name: str
    width: int
    align: str = 'left'  # 'left', 'right', 'center'
    format_func: Optional[callable] = None


class TerminalDisplay:
    """Terminal display utility class"""
    
    # Color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'end': '\033[0m'
    }
    
    # Special characters
    CHARS = {
        'box_top_left': '┌',
        'box_top_right': '┐',
        'box_bottom_left': '└',
        'box_bottom_right': '┘',
        'box_horizontal': '─',
        'box_vertical': '│',
        'box_cross': '┼',
        'box_t_down': '┬',
        'box_t_up': '┴',
        'box_t_right': '├',
        'box_t_left': '┤',
        'progress_full': '█',
        'progress_empty': '░',
        'check': '✓',
        'cross': '✗',
        'arrow_right': '→',
        'bullet': '•'
    }
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Add color to text"""
        if color in cls.COLORS:
            return f"{cls.COLORS[color]}{text}{cls.COLORS['end']}"
        return text
    
    @classmethod
    def print_header(cls, title: str, width: int = 80, color: str = 'cyan'):
        """Print title header"""
        print()
        print(cls.colorize('═' * width, color))
        title_line = f"  {title}  "
        padding = (width - len(title_line)) // 2
        centered_title = ' ' * padding + title_line + ' ' * (width - len(title_line) - padding)
        print(cls.colorize(centered_title, 'bold'))
        print(cls.colorize('═' * width, color))
        print()
    
    @classmethod
    def print_section(cls, title: str, color: str = 'yellow'):
        """Print section title"""
        print()
        print(cls.colorize(f"- {title}", color))
        print(cls.colorize('─' * (len(title) + 3), color))
    
    @classmethod
    def print_progress_bar(cls, current: int, total: int,
                          prefix: str = '', suffix: str = '',
                          width: int = 50, color: str = 'green'):
        """Print progress bar"""
        if total == 0:
            percent = 100
        else:
            percent = int(100 * current / total)
        
        filled_width = int(width * current / total) if total > 0 else width
        bar = cls.CHARS['progress_full'] * filled_width + cls.CHARS['progress_empty'] * (width - filled_width)
        
        progress_text = f"\r{prefix} |{bar}| {percent:3d}% {suffix}"
        print(cls.colorize(progress_text, color), end='', flush=True)
        
        if current >= total:
            print()  # New line after completion
    
    @classmethod
    def print_table(cls, data: List[Dict], columns: List[TableColumn],
                   title: Optional[str] = None, show_index: bool = False):
        """Print table"""
        if not data:
            print("No data to display")
            return
        
        # Calculate column widths
        if show_index:
            index_width = len(str(len(data))) + 2
            total_width = index_width + sum(col.width for col in columns) + len(columns) + 1
        else:
            total_width = sum(col.width for col in columns) + len(columns) + 1
        
        # Print title
        if title:
            print()
            print(cls.colorize(f"[TABLE] {title}", 'bold'))
            print()
        
        # Print table top
        if show_index:
            print(cls.CHARS['box_top_left'] + cls.CHARS['box_horizontal'] * (index_width - 1), end='')
            print(cls.CHARS['box_t_down'], end='')
        else:
            print(cls.CHARS['box_top_left'], end='')
        
        for i, col in enumerate(columns):
            print(cls.CHARS['box_horizontal'] * col.width, end='')
            if i < len(columns) - 1:
                print(cls.CHARS['box_t_down'], end='')
        print(cls.CHARS['box_top_right'])
        
        # Print table header
        if show_index:
            print(cls.CHARS['box_vertical'] + ' ' * (index_width - 1), end='')
            print(cls.CHARS['box_vertical'], end='')
        else:
            print(cls.CHARS['box_vertical'], end='')
        
        for col in columns:
            header_text = cls._format_cell(col.name, col.width, col.align)
            print(cls.colorize(header_text, 'bold'), end='')
            print(cls.CHARS['box_vertical'], end='')
        print()
        
        # Print separator line
        if show_index:
            print(cls.CHARS['box_t_right'] + cls.CHARS['box_horizontal'] * (index_width - 1), end='')
            print(cls.CHARS['box_cross'], end='')
        else:
            print(cls.CHARS['box_t_right'], end='')
        
        for i, col in enumerate(columns):
            print(cls.CHARS['box_horizontal'] * col.width, end='')
            if i < len(columns) - 1:
                print(cls.CHARS['box_cross'], end='')
        print(cls.CHARS['box_t_left'])
        
        # Print data rows
        for idx, row in enumerate(data):
            if show_index:
                index_text = cls._format_cell(str(idx + 1), index_width - 1, 'right')
                print(cls.CHARS['box_vertical'] + index_text, end='')
                print(cls.CHARS['box_vertical'], end='')
            else:
                print(cls.CHARS['box_vertical'], end='')
            
            for col in columns:
                value = row.get(col.name, '')
                if col.format_func:
                    value = col.format_func(value)
                cell_text = cls._format_cell(str(value), col.width, col.align)
                print(cell_text, end='')
                print(cls.CHARS['box_vertical'], end='')
            print()
        
        # Print table bottom
        if show_index:
            print(cls.CHARS['box_bottom_left'] + cls.CHARS['box_horizontal'] * (index_width - 1), end='')
            print(cls.CHARS['box_t_up'], end='')
        else:
            print(cls.CHARS['box_bottom_left'], end='')
        
        for i, col in enumerate(columns):
            print(cls.CHARS['box_horizontal'] * col.width, end='')
            if i < len(columns) - 1:
                print(cls.CHARS['box_t_up'], end='')
        print(cls.CHARS['box_bottom_right'])
        print()
    
    @classmethod
    def _format_cell(cls, text: str, width: int, align: str) -> str:
        """Format cell text"""
        if len(text) > width:
            text = text[:width-3] + '...'
        
        if align == 'left':
            return f" {text:<{width-1}}"
        elif align == 'right':
            return f"{text:>{width-1}} "
        elif align == 'center':
            return f"{text:^{width}}"
        else:
            return f" {text:<{width-1}}"
    
    @classmethod
    def print_comparison_table(cls, comparison_data: Dict[str, Dict[str, float]],
                             title: str = "Strategy Comparison Analysis"):
        """Print strategy comparison table"""
        if not comparison_data:
            print("No comparison data")
            return
        
        # Prepare table data
        strategies = list(comparison_data.keys())
        metrics = list(next(iter(comparison_data.values())).keys())
        
        # Define columns
        columns = [TableColumn("Metric", 20, 'left')]
        for strategy in strategies:
            columns.append(TableColumn(strategy.title(), 15, 'right', cls._format_number))
        
        # Prepare data
        table_data = []
        for metric in metrics:
            row = {"Metric": cls._format_metric_name(metric)}
            for strategy in strategies:
                row[strategy.title()] = comparison_data[strategy].get(metric, 0)
            table_data.append(row)
        
        cls.print_table(table_data, columns, title)
    
    @classmethod
    def _format_metric_name(cls, metric: str) -> str:
        """Format metric name"""
        name_map = {
            'npv_mean': 'NPV Mean',
            'npv_std': 'NPV Std Dev',
            'npv_min': 'NPV Min',
            'npv_max': 'NPV Max',
            'utilization_mean': 'Avg Utilization',
            'self_sufficiency_mean': 'Self-Sufficiency',
            'total_cost_mean': 'Avg Total Cost',
            'probability_positive_npv': 'Positive NPV Prob'
        }
        return name_map.get(metric, metric)
    
    @classmethod
    def _format_number(cls, value: Any) -> str:
        """Format number display"""
        if isinstance(value, (int, float)):
            if abs(value) >= 1e6:
                return f"{value/1e6:.1f}M"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.1f}K"
            elif 0 < abs(value) < 1:
                return f"{value:.3f}"
            else:
                return f"{value:.1f}"
        return str(value)
    
    @classmethod
    def print_simulation_status(cls, strategy: str, T: int, current: int, total: int):
        """Print simulation status"""
        status_text = f"{strategy.title()} (T={T})"
        cls.print_progress_bar(current, total, prefix=status_text, suffix=f"{current}/{total}")
    
    @classmethod
    def print_summary_box(cls, title: str, data: Dict[str, Any], color: str = 'green'):
        """Print summary box"""
        max_key_length = max(len(str(k)) for k in data.keys()) if data else 0
        max_value_length = max(len(str(v)) for v in data.values()) if data else 0
        box_width = max(len(title) + 4, max_key_length + max_value_length + 6, 40)
        
        print()
        print(cls.colorize(cls.CHARS['box_top_left'] + cls.CHARS['box_horizontal'] * (box_width - 2) + cls.CHARS['box_top_right'], color))
        
        # Title
        title_line = f"{cls.CHARS['box_vertical']} {title:^{box_width-4}} {cls.CHARS['box_vertical']}"
        print(cls.colorize(title_line, color))
        
        if data:
            # Separator line
            sep_line = f"{cls.CHARS['box_t_right']}{cls.CHARS['box_horizontal'] * (box_width - 2)}{cls.CHARS['box_t_left']}"
            print(cls.colorize(sep_line, color))
            
            # Data rows
            for key, value in data.items():
                if isinstance(value, float):
                    if abs(value) >= 1e6:
                        value_str = f"${value/1e6:.1f}M"
                    elif abs(value) >= 1e3:
                        value_str = f"${value/1e3:.1f}K"
                    elif 0 < abs(value) < 1:
                        value_str = f"{value:.1%}"
                    else:
                        value_str = f"{value:.1f}"
                else:
                    value_str = str(value)
                
                data_line = f"{cls.CHARS['box_vertical']} {key:<{max_key_length}} : {value_str:>{max_value_length}} {cls.CHARS['box_vertical']}"
                print(cls.colorize(data_line, color))
        
        print(cls.colorize(cls.CHARS['box_bottom_left'] + cls.CHARS['box_horizontal'] * (box_width - 2) + cls.CHARS['box_bottom_right'], color))
        print()
    
    @classmethod
    def print_ascii_chart(cls, data: List[float], title: str = "", width: int = 60, height: int = 10):
        """Print ASCII chart"""
        if not data:
            print("No data to plot chart")
            return
        
        print()
        if title:
            print(cls.colorize(f"[CHART] {title}", 'bold'))
            print()
        
        # Normalize data
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            normalized = [height // 2] * len(data)
        else:
            normalized = [int((val - min_val) / (max_val - min_val) * (height - 1)) for val in data]
        
        # Draw chart
        for y in range(height - 1, -1, -1):
            line = ""
            for x in range(len(data)):
                if normalized[x] >= y:
                    line += "█"
                else:
                    line += " "
            
            # Add Y-axis labels
            y_val = min_val + (max_val - min_val) * y / (height - 1)
            print(f"{y_val:8.1f} │{line}")
        
        # X-axis
        print(" " * 9 + "└" + "─" * len(data))
        
        # X-axis labels
        x_labels = ""
        for i in range(0, len(data), max(1, len(data) // 10)):
            x_labels += f"{i:>6}"
        print(" " * 10 + x_labels)
        print()


if __name__ == "__main__":
    # Test code
    print("=== Terminal Display Tools Test ===")
    
    # Test title
    TerminalDisplay.print_header("ISRU Strategy Simulation Analysis System", width=60)
    
    # Test section
    TerminalDisplay.print_section("Simulation Progress")
    
    # Test progress bar
    for i in range(101):
        TerminalDisplay.print_progress_bar(i, 100, prefix="Conservative", suffix="Complete")
        time.sleep(0.01)
    
    # Test table
    test_data = [
        {"Strategy": "Conservative", "NPV": 2450000, "Utilization": 0.873, "Cost": 1200000},
        {"Strategy": "Aggressive", "NPV": 2680000, "Utilization": 0.918, "Cost": 1450000},
        {"Strategy": "Moderate", "NPV": 2590000, "Utilization": 0.895, "Cost": 1350000}
    ]
    
    columns = [
        TableColumn("Strategy", 12, 'left'),
        TableColumn("NPV", 12, 'right', TerminalDisplay._format_number),
        TableColumn("Utilization", 10, 'right', lambda x: f"{x:.1%}"),
        TableColumn("Cost", 12, 'right', TerminalDisplay._format_number)
    ]
    
    TerminalDisplay.print_table(test_data, columns, "Strategy Comparison Results", show_index=True)
    
    # Test summary box
    summary_data = {
        "Total Simulations": 1000,
        "Best Strategy": "Aggressive",
        "Average NPV": 2573333.33,
        "Success Rate": 0.95
    }
    
    TerminalDisplay.print_summary_box("Simulation Summary", summary_data)
    
    # Test ASCII chart
    test_chart_data = [10, 15, 12, 18, 25, 22, 30, 28, 35, 32, 40, 38, 45]
    TerminalDisplay.print_ascii_chart(test_chart_data, "NPV Trend Chart", width=50, height=8)