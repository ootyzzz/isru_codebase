"""
NPV vs T Batch Analysis Script
Performs batch optimization solving for T=1 to 50 years, collecting NPV data for subsequent analysis
"""

import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import solving function
from optimal.optimal_solu import solve_isru_optimization

def batch_solve_npv_analysis(t_min=1, t_max=50, random_seed=42, output_dir="optimal/results"):
    """
    Batch solve ISRU optimization problems for different T values
    
    Args:
        t_min: Minimum T value
        t_max: Maximum T value
        random_seed: Random seed
        output_dir: Output directory
        
    Returns:
        pandas.DataFrame: Data frame containing all results
    """
    
    print(f"🚀 Starting batch NPV analysis: T={t_min} to {t_max} years")
    print(f"📁 Results will be saved to: {output_dir}")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store results
    results = []
    failed_cases = []
    
    # Record total start time
    total_start_time = time.time()
    
    # Batch solving
    for T in range(t_min, t_max + 1):
        print(f"📊 Solving T={T} years...")
        
        try:
            # Solve single case
            start_time = time.time()
            result = solve_isru_optimization(T, random_seed=random_seed, verbose=False)
            solve_time = time.time() - start_time
            
            # Record result
            result_record = {
                'T': T,
                'NPV': result['npv'],
                'Status': result['status'],
                'Solve_Time': solve_time,
                'Demand_Sum': sum(result['demand_path']) if result['demand_path'] is not None else None,
                'Demand_Mean': np.mean(result['demand_path']) if result['demand_path'] is not None else None,
                'Demand_Std': np.std(result['demand_path']) if result['demand_path'] is not None else None
            }
            
            results.append(result_record)
            
            # Print progress
            if result['status'] == 'optimal':
                print(f"   ✅ NPV = {result['npv']:,.2f} (Time: {solve_time:.2f}s)")
            else:
                print(f"   ❌ Failed: {result['status']} (Time: {solve_time:.2f}s)")
                failed_cases.append(T)
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
            failed_cases.append(T)
            results.append({
                'T': T,
                'NPV': None,
                'Status': 'error',
                'Solve_Time': None,
                'Demand_Sum': None,
                'Demand_Mean': None,
                'Demand_Std': None
            })
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📈 Batch analysis completed!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Successfully solved: {len(df_results[df_results['Status'] == 'optimal'])} / {len(df_results)}")
    
    if failed_cases:
        print(f"❌ Failed cases: {failed_cases}")
    
    # If there are successful cases, print NPV statistics
    successful_results = df_results[df_results['Status'] == 'optimal']
    if not successful_results.empty:
        print(f"💰 NPV range: {successful_results['NPV'].min():,.2f} to {successful_results['NPV'].max():,.2f}")
        print(f"📈 Optimal T value: T={successful_results.loc[successful_results['NPV'].idxmax(), 'T']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/npv_vs_time_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"💾 Results saved to: {csv_filename}")
    
    return df_results, csv_filename

def main():
    """Main function"""
    print("NPV vs T Batch Analysis Tool")
    print("=" * 60)
    
    # Run batch analysis
    df_results, csv_file = batch_solve_npv_analysis(t_min=1, t_max=50)
    
    # Simple statistical output
    print(f"\n📋 Quick preview:")
    print(df_results.head(10))
    
    print(f"\n🎯 Next steps:")
    print(f"   1. View complete results: {csv_file}")
    print(f"   2. Run visualization script: python optimal/visualization.py")

if __name__ == "__main__":
    main()
