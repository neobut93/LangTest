
import openai
import os
from langtest import Harness
from report_generator import ReportGenerator

# Define the task and model
h = Harness(
    task="question-answering",
    model={"model": "gpt-3.5-turbo", "hub": "openai"},
    data={
        "source": "huggingface",
        "data_source": "squad_v2",   # standard QA dataset
        "split": "validation[:10]"   # small slice for cost control
    },
    config="config.yaml"
)

# Generate test cases, run the evaluation, and generate a report
h.generate()
h.run()

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Generate summary report (aggregated stats)
h.report(format="html", save_dir="reports/report.html")
print("Summary report written to ./reports/report.html")

# Initialize report generator
report_gen = ReportGenerator(output_dir="reports")

# Generate detailed results
detailed_df = h.generated_results()
if detailed_df is not None and not detailed_df.empty:
    # Generate detailed HTML and CSV reports using the ReportGenerator
    html_path = report_gen.generate_detailed_html(detailed_df, "detailed_report.html")
    csv_path = report_gen.generate_detailed_csv(detailed_df, "detailed_results.csv")
    
    print(f"\nDetailed report written to {html_path}")
    print(f"Detailed CSV written to {csv_path}")
    
    # Print statistics
    report_gen.print_stats(detailed_df)
else:
    print("No detailed results generated")
