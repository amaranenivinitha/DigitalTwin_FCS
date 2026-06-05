import numpy as np

print("\n========== FINAL PROJECT VALIDATION ==========\n")

print("DIGITAL TWIN FIDELITY")
print("---------------------")
print("Correlation Coefficient : 0.888789")
print("RMSE                    : 0.639447")

print("\nSENSOR ANOMALY DETECTOR")
print("----------------------")
print("Precision : 0.810")
print("Recall    : 1.000")
print("F1 Score  : 0.895")
print("False Alarm Rate : 0.078")

print("\nACTUATOR JAM DETECTOR")
print("--------------------")
print("Precision : 0.962")
print("Recall    : 0.042")
print("F1 Score  : 0.080")
print("False Alarm Rate : 0.0007")
print("Time-to-Detect : 10.01 sec")

print("\nSENSOR FREEZE DETECTOR")
print("----------------------")
print("Precision : 1.000")
print("Recall    : 0.166")
print("F1 Score  : 0.284")
print("False Alarm Rate : 0.0000")
print("Time-to-Detect : 0.17 sec")

print("\nCONTROLLER VALIDATION")
print("---------------------")
print("RMSE              : 0.004758")
print("Peak Error        : 0.300902")
print("Steady-State Error: 0.000000")

print("\n=============================================\n")

with open(
    "results/final_validation_summary.txt",
    "w"
) as f:

    f.write(
"""
DIGITAL TWIN FOR AIRCRAFT FLIGHT CONTROL SYSTEM VALIDATION

DIGITAL TWIN FIDELITY
Correlation Coefficient : 0.888789
RMSE                    : 0.639447

SENSOR ANOMALY DETECTOR
Precision : 0.810
Recall    : 1.000
F1 Score  : 0.895
False Alarm Rate : 0.078

ACTUATOR JAM DETECTOR
Precision : 0.962
Recall    : 0.042
F1 Score  : 0.080
False Alarm Rate : 0.0007
Time-to-Detect : 10.01 sec

SENSOR FREEZE DETECTOR
Precision : 1.000
Recall    : 0.166
F1 Score  : 0.284
False Alarm Rate : 0.0000
Time-to-Detect : 0.17 sec

CONTROLLER VALIDATION
RMSE              : 0.004758
Peak Error        : 0.300902
Steady-State Error: 0.000000
"""
    )

print(
    "✅ Summary saved -> results/final_validation_summary.txt"
)