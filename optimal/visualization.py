"""
NPV vs T Visualization Script
Read batch analysis results and generate visualization charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

# Set matplotlib backend and chart style
import matplotlib
matplotlib.use('TkAgg')  # Ensure correct backend is used
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100  # Lower DPI to improve performance
plt.rcParams['savefig.dpi'] = 300
# Don't set plt.ioff(), keep default interactive mode

def load_latest_results(results_dir="optimal/results"):
    """
    Load latest analysis result file
    
    Args:
        results_dir: Results directory
        
    Returns:
        pandas.DataFrame: Analysis result data
        str: File path
    """
    # Find all result files
    pattern = f"{results_dir}/npv_vs_time_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No result files found in directory {results_dir}")
    
    # Select the latest file
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 Loading data file: {latest_file}")
    
    # Read data
    df = pd.read_csv(latest_file)
    print(f"📊 Data overview: {len(df)} cases")
    
    return df, latest_file

def create_simple_npv_plot(df, output_dir="optimal/charts"):
    """
    Create simplified NPV vs T main chart
    
    Args:
        df: DataFrame containing analysis results
        output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successfully solved cases
    df_success = df[df['Status'] == 'optimal'].copy()
    
    if df_success.empty:
        print("❌ No successfully solved cases, cannot generate charts")
        return
    
    # Create single main chart
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # NPV vs T main chart
    ax.plot(df_success['T'], df_success['NPV'], 'b-', linewidth=3, marker='o', markersize=6, label='NPV')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Break-even')
    ax.set_xlabel('Time Horizon (Years)', fontsize=12)
    ax.set_ylabel('NPV (Currency Units)', fontsize=12)
    ax.set_title('ISRU Project NPV vs Time Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Annotate optimal point
    max_npv_idx = df_success['NPV'].idxmax()
    max_npv_t = df_success.loc[max_npv_idx, 'T']
    max_npv_value = df_success.loc[max_npv_idx, 'NPV']
    ax.annotate(f'Optimal Point\nT={max_npv_t} years\nNPV={max_npv_value:,.0f}',
                xy=(max_npv_t, max_npv_value),
                xytext=(max_npv_t+5, max_npv_value+max_npv_value*0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/npv_main_plot_{timestamp}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"📈 Main chart saved to: {output_file}")
    
    # Show chart
    plt.show()
    
    return output_file

def create_npv_vs_t_visualization(df, output_dir="optimal/charts"):
    """
    Create NPV vs T visualization charts - displayed separately in 4 independent windows
    
    Args:
        df: DataFrame containing analysis results
        output_dir: Output directory
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successfully solved cases
    df_success = df[df['Status'] == 'optimal'].copy()
    
    if df_success.empty:
        print("❌ No successfully solved cases, cannot generate charts")
        return
    
    print(f"📈 Number of successful cases: {len(df_success)}")
    
    # Close all existing charts
    plt.close('all')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = []
    
    # Use matplotlib default behavior
    
    # 1. Main chart: NPV vs T
    fig1 = plt.figure(1, figsize=(10, 6))
    plt.plot(df_success['T'], df_success['NPV'], 'b-', linewidth=2, marker='o', markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break-even')
    plt.xlabel('Time Horizon (Years)', fontsize=12)
    plt.ylabel('NPV (Currency Units)', fontsize=12)
    plt.title('NPV vs Time Horizon', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate optimal point
    max_npv_idx = df_success['NPV'].idxmax()
    max_npv_t = df_success.loc[max_npv_idx, 'T']
    max_npv_value = df_success.loc[max_npv_idx, 'NPV']
    plt.annotate(f'Max NPV\nT={max_npv_t}, NPV={max_npv_value:,.0f}',
                xy=(max_npv_t, max_npv_value), xytext=(max_npv_t+5, max_npv_value+max_npv_value*0.1),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    output_file1 = f"{output_dir}/npv_main_plot_{timestamp}.png"
    plt.savefig(output_file1, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file1)
    print(f"📈 Main chart saved to: {output_file1}")
    
    # 2. NPV growth rate
    fig2 = plt.figure(2, figsize=(10, 6))
    df_success_sorted = df_success.sort_values('T')
    npv_growth = df_success_sorted['NPV'].pct_change() * 100
    plt.plot(df_success_sorted['T'].iloc[1:], npv_growth.iloc[1:], 'g-', linewidth=2, marker='s', markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Time Horizon (Years)', fontsize=12)
    plt.ylabel('NPV Growth Rate (%)', fontsize=12)
    plt.title('NPV Period-over-Period Growth Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = f"{output_dir}/npv_growth_rate_{timestamp}.png"
    plt.savefig(output_file2, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file2)
    print(f"📈 Growth rate chart saved to: {output_file2}")
    
    # 3. Solve time analysis
    fig3 = plt.figure(3, figsize=(10, 6))
    plt.scatter(df_success['T'], df_success['Solve_Time'], alpha=0.6, s=60, color='blue')
    plt.xlabel('Time Horizon (Years)', fontsize=12)
    plt.ylabel('Solve Time (Seconds)', fontsize=12)
    plt.title('Computational Time vs Problem Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_success['T'], df_success['Solve_Time'], 1)
    p = np.poly1d(z)
    plt.plot(df_success['T'], p(df_success['T']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    plt.legend()
    
    plt.tight_layout()
    output_file3 = f"{output_dir}/solve_time_analysis_{timestamp}.png"
    plt.savefig(output_file3, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file3)
    print(f"📈 Solve time chart saved to: {output_file3}")
    
    # 4. NPV distribution histogram
    fig4 = plt.figure(4, figsize=(10, 6))
    plt.hist(df_success['NPV'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(x=df_success['NPV'].mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {df_success["NPV"].mean():,.0f}')
    plt.axvline(x=df_success['NPV'].median(), color='g', linestyle='--', linewidth=2,
                label=f'Median: {df_success["NPV"].median():,.0f}')
    plt.xlabel('NPV (Currency Units)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('NPV Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file4 = f"{output_dir}/npv_distribution_{timestamp}.png"
    plt.savefig(output_file4, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file4)
    print(f"📈 Distribution chart saved to: {output_file4}")
    
    # Show all charts - use default behavior
    plt.show()
    
    print(f"\n🎯 Generated 4 independent chart windows!")
    
    return output_files

def print_analysis_summary(df):
    """
    Print analysis summary
    
    Args:
        df: Analysis results DataFrame
    """
    print("\n" + "="*60)
    print("📊 NPV vs T Analysis Summary")
    print("="*60)
    
    # Basic statistics
    df_success = df[df['Status'] == 'optimal']
    total_cases = len(df)
    success_cases = len(df_success)
    
    print(f"Total cases: {total_cases}")
    print(f"Successfully solved: {success_cases} ({success_cases/total_cases*100:.1f}%)")
    
    if success_cases > 0:
        print(f"\nNPV statistics:")
        print(f"  Minimum: {df_success['NPV'].min():,.2f}")
        print(f"  Maximum: {df_success['NPV'].max():,.2f}")
        print(f"  Mean: {df_success['NPV'].mean():,.2f}")
        print(f"  Median: {df_success['NPV'].median():,.2f}")
        
        # Find optimal T value
        max_npv_idx = df_success['NPV'].idxmax()
        optimal_t = df_success.loc[max_npv_idx, 'T']
        optimal_npv = df_success.loc[max_npv_idx, 'NPV']
        print(f"\nOptimal time horizon:")
        print(f"  T = {optimal_t} years")
        print(f"  NPV = {optimal_npv:,.2f}")
        
        # Cases with positive NPV
        positive_npv = df_success[df_success['NPV'] > 0]
        if not positive_npv.empty:
            min_positive_t = positive_npv['T'].min()
            print(f"\nNPV break-even point: T = {min_positive_t} years")
        
        # Solve time statistics
        print(f"\nSolve time statistics:")
        print(f"  Average time: {df_success['Solve_Time'].mean():.3f} seconds")
        print(f"  Maximum time: {df_success['Solve_Time'].max():.3f} seconds")
    
    # Failed cases
    failed_cases = df[df['Status'] != 'optimal']
    if not failed_cases.empty:
        print(f"\nFailed cases: {len(failed_cases)} cases")
        print(f"Failed T values: {failed_cases['T'].tolist()}")

def main():
    """Main function"""
    print("NPV vs T Visualization Analysis Tool")
    print("=" * 60)
    
    try:
        # Load data
        df, data_file = load_latest_results()
        
        # Print analysis summary
        print_analysis_summary(df)
        
        # Directly create 4 independent window visualizations
        print("\n📊 Generating 4 independent chart windows...")
        chart_files = create_npv_vs_t_visualization(df)
        chart_file = f"4 chart files: {', '.join([f.split('/')[-1] for f in chart_files])}"
        
        print(f"\n🎯 Analysis completed!")
        print(f"   Data file: {data_file}")
        print(f"   Chart files: {chart_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Hint:")
        print("   Please run batch analysis first: python optimal/npv_time_analysis.py")

if __name__ == "__main__":
    main()
