import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from plot_style import RESULT_ROOT, SAVE_DIRS, clear_experiment_outputs

SUMMARY_DIR = os.path.join(RESULT_ROOT, "A_calibration", "test0", "csv")
os.makedirs(SUMMARY_DIR, exist_ok=True)

csv_files = [
    (SAVE_DIRS[1], "01_noise_params.csv"),
    (SAVE_DIRS[2], "02_repeatability_results.csv"),
    (SAVE_DIRS[3], "03_calibration_data.csv"),
    (SAVE_DIRS[3], "03_calibration_params.csv"),
    (SAVE_DIRS[4], "04_hysteresis_data.csv"),
    (SAVE_DIRS[4], "04_hysteresis_params.csv"),
    (SAVE_DIRS[5], "05_detection_limit_data.csv"),
    (SAVE_DIRS[5], "05_detection_limit_params.csv"),
    (SAVE_DIRS[6], "06_temperature_data.csv"),
    (SAVE_DIRS[6], "06_temperature_params.csv"),
]

clear_experiment_outputs((SCRIPT_DIR, SUMMARY_DIR), None, "00_")
output_path = os.path.join(SUMMARY_DIR, "00_experimentA_summary.txt")

with open(output_path, "w", encoding="utf-8") as output_file:
    for save_dir, csv_file in csv_files:
        file_path = os.path.join(save_dir, "csv", csv_file)

        if os.path.exists(file_path):
            output_file.write(f"{csv_file}\n")
            with open(file_path, "r", encoding="utf-8") as csv_f:
                output_file.write(csv_f.read())
            output_file.write("\n\n")
            print(f"Added {file_path}")
        else:
            print(f"File not found: {file_path}")

print(f"\nExperiment summary saved to {output_path}")
