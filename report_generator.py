#!/usr/bin/env python3
"""
Report Generator class for creating styled HTML and CSV reports from langtest results
"""

import pandas as pd
import os
from pathlib import Path


class ReportGenerator:
    """Generates styled HTML and CSV reports from test results"""
    
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detailed Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .table {{ border-collapse: collapse; width: 100%; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #4CAF50; color: white; }}
            .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .pass-true {{ background-color: #d4edda; }}
            .pass-false {{ background-color: #f8d7da; }}
            .summary {{ margin-bottom: 30px; padding: 15px; background-color: #f0f0f0; }}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    
    def __init__(self, output_dir="reports"):
        """Initialize the report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_detailed_html(self, df: pd.DataFrame, filename="detailed_report.html"):
        """Generate a styled HTML report from detailed results dataframe
        
        Args:
            df: DataFrame with test results
            filename: Name of the output HTML file
            
        Returns:
            Path to the generated file
        """
        if df is None or df.empty:
            return None
        
        # Generate HTML table
        html_str = df.to_html(classes='table table-striped', border=0, index=False)
        
        # Create summary stats
        passed = df['pass'].sum() if 'pass' in df.columns else 0
        failed = (~df['pass']).sum() if 'pass' in df.columns else 0
        
        summary = f"""
        <h1>Detailed Test Results</h1>
        <div class="summary">
            <strong>Total Tests:</strong> {len(df)}<br>
            <strong>Passed:</strong> {passed}<br>
            <strong>Failed:</strong> {failed}<br>
            <strong>Pass Rate:</strong> {(passed/len(df)*100):.1f}%
        </div>
        """
        
        # Combine summary and table
        full_html = summary + html_str
        
        # Wrap in template
        styled_html = self.HTML_TEMPLATE.format(content=full_html)
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(styled_html)
        
        return output_path
    
    def generate_detailed_csv(self, df: pd.DataFrame, filename="detailed_results.csv"):
        """Save detailed results to CSV
        
        Args:
            df: DataFrame with test results
            filename: Name of the output CSV file
            
        Returns:
            Path to the generated file
        """
        if df is None or df.empty:
            return None
        
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_summary_html(self, df: pd.DataFrame, filename="summary_report.html"):
        """Generate a summary HTML report with statistics
        
        Args:
            df: DataFrame with summary results
            filename: Name of the output HTML file
            
        Returns:
            Path to the generated file
        """
        if df is None or df.empty:
            return None
        
        # Generate HTML table
        html_str = df.to_html(classes='table table-striped', border=0, index=False)
        
        content = f"""
        <h1>Test Summary Report</h1>
        {html_str}
        """
        
        # Wrap in template
        styled_html = self.HTML_TEMPLATE.format(content=content)
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(styled_html)
        
        return output_path
    
    def print_stats(self, df: pd.DataFrame):
        """Print statistics about the test results
        
        Args:
            df: DataFrame with test results
        """
        if df is None or df.empty:
            print("No results to display")
            return
        
        if 'pass' in df.columns:
            passed = df['pass'].sum()
            failed = (~df['pass']).sum()
            total = len(df)
            pass_rate = (passed/total*100) if total > 0 else 0
            
            print(f"\n{'='*80}")
            print(f"TEST RESULTS SUMMARY")
            print(f"{'='*80}")
            print(f"Total tests: {total}")
            print(f"Passed: {passed} ({pass_rate:.1f}%)")
            print(f"Failed: {failed} ({100-pass_rate:.1f}%)")
            print(f"{'='*80}\n")
        else:
            print(f"Total tests: {len(df)}")


if __name__ == "__main__":
    # Example usage
    print("ReportGenerator - Use with langtest results")

