# Bias Detection in Shortlisting-Style Employment Systems

This repository contains the code, outputs, and supporting materials for my master’s thesis on fairness auditing in employment-related machine learning systems.

## Overview

The purpose of this project is to evaluate whether standard machine learning classifiers produce measurable group-level disparities in structured employment-related datasets.

The study compares model performance and fairness across two datasets:

- **UCI Adult dataset** — a public benchmark dataset widely used in fairness research
- **Merged organisational HR dataset** — a private dataset created by merging three internal Excel files

Three baseline models are used:

- Logistic Regression
- Random Forest
- Gradient Boosting

The models are evaluated using:

- Accuracy
- Statistical Parity Difference (SPD)
- Disparate Impact (DI)
- Equal Opportunity Difference (EOD)

## Datasets Used

### 1. UCI Adult Dataset
This is the public benchmark dataset used in the analysis.

Files in `data/`:
- `adult.data` → used in the main analysis
- `adult.names` → metadata/documentation
- `adult.test` → not used in the main script

### 2. Merged HR Dataset
The second dataset is built by merging three internal Excel files:

- `HRDatabase.xlsx`
- `EmployeeInformation.xlsx`
- `DepartmentInformation.xlsx`

These files are merged into one organisational HR dataset during preprocessing.

**Important:**  
These HR files are private and should not be uploaded publicly if they contain confidential information.

## Project Structure

```text
M599_Thesis/
├── data/
│   ├── adult.data
│   ├── adult.names
│   ├── adult.test
│   ├── HRDatabase.xlsx
│   ├── EmployeeInformation.xlsx
│   └── DepartmentInformation.xlsx
│
├── Results/
│   ├── baseline_results_with_fairness.csv
│   ├── adult_dataset_accuracy.png
│   ├── adult_dataset_spd.png
│   ├── adult_dataset_di.png
│   ├── adult_dataset_eod.png
│   ├── merged_hr_dataset_accuracy.png
│   ├── merged_hr_dataset_spd.png
│   ├── merged_hr_dataset_di.png
│   └── merged_hr_dataset_eod.png
│
├── src/
│   └── main.py
│
├── Thesis/
│   └── final_thesis.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
How to Run the Project
1. Clone the repository
git clone https://github.com/SeverusSnapee/M599_THESIS.git
cd M599_THESIS
2. Install dependencies

Make sure Python is installed, then run:

pip install -r requirements.txt
3. Make sure the input files are inside the data/ folder

The script expects the following files inside data/:

adult.data
HRDatabase.xlsx
EmployeeInformation.xlsx
DepartmentInformation.xlsx
4. Run the script

From the project root folder, run:

python src/main.py
What the Script Does

The script performs the following steps:

Loads the Adult dataset
Loads and merges the three HR Excel files
Cleans and encodes both datasets
Creates the HR target variable
Splits both datasets into train and test sets
Trains three baseline machine learning models
Evaluates performance using accuracy
Evaluates fairness using SPD, DI, and EOD
Saves the results to CSV
Generates and saves plots for all metrics
Output Files

After running the script, the following outputs are saved inside the Results/ folder:

baseline_results_with_fairness.csv
accuracy comparison charts
SPD charts
DI charts
EOD charts
Notes on Reproducibility

The Adult dataset can be reproduced fully because it is public.

The HR dataset cannot be shared publicly if it contains confidential organisational data.
However, the full preprocessing, merging, modelling, and fairness evaluation workflow is included in the code.

Main Findings
Adult Dataset
Gradient Boosting achieved the highest accuracy
Random Forest produced the lowest EOD
Logistic Regression showed the highest DI and EOD
Merged HR Dataset
Logistic Regression achieved the highest accuracy
Some fairness values appeared close to parity
Results should be interpreted cautiously because the HR dataset is small
Thesis

The full thesis PDF is stored in the Thesis/ folder.

Requirements

This project uses the following main Python libraries:

pandas
numpy
scikit-learn
matplotlib
openpyxl
Author

Parabhjyot Singh
Master’s Computer Science
GISMA University Potsdam
