# Predictive Modeling for Medicare Advantage Medication Adherence

This repository contains the Python code for a machine learning model designed to proactively identify Medicare Advantage members at risk of becoming non-adherent to their critical medications. The project's goal is to improve member health outcomes and increase the plan's CMS Star Ratings by enabling targeted clinical interventions for three core medication classes.

## 1. Business Need

For a Medicare Advantage health plan, **CMS Star Ratings** are a critical driver of both revenue and market reputation. A significant portion of these ratings is determined by medication adherence metrics for members with chronic conditions. This project focuses on the three most impactful measures:

1.  **Medication Adherence for Diabetes (MAD)**: Oral diabetes medications.
2.  **Medication Adherence for Hypertension (MAH)**: Renin-Angiotensin System (RAS) antagonists.
3.  **Medication Adherence for Cholesterol (MAC)**: Statins.

Our organization faced a persistent challenge where a subset of members failed to consistently refill their prescriptions for these conditions. This not only posed a direct risk to member health but also suppressed our Star Ratings across these key measures, jeopardizing millions in potential **Quality Bonus Payments (QBP)** and hindering our ability to attract new members.

## 2. Project Goal

The objective was to shift from a reactive to a **proactive clinical outreach model** by accurately forecasting which members were at the highest risk of their **Proportion of Days Covered (PDC)** falling below the critical 80% threshold for the MAD, MAH, and MAC measures.

This tool would provide our clinical outreach teams with a single, prioritized worklist, allowing them to focus their limited resources on the members who needed support the most, *before* they became non-adherent.

## 3. The Plan & Execution

The project was executed through a multi-stage machine learning pipeline:

1.  **Data Integration**: The project utilized member data exported from **Salesforce** into Excel spreadsheets. A **Python** script was developed to automatically extract and process this data, combining it with medical claims (diagnoses) and pharmacy claims (Rx fills) to create the final dataset for analysis.
2.  **Cohort Definition**: Isolated three distinct member populations based on their diagnoses and prescription history, corresponding to the MAD, MAH, and MAC measures.
3.  **Feature Engineering**: Calculated the historical PDC for each member within their specific measure cohort. Engineered over 30 additional features to serve as predictors, including:
    * Variance in prescription refill dates
    * Average gap between refills
    * Number of co-existing chronic conditions (polypharmacy)
    * Member age and tenure with the health plan
4.  **Model Development**: Framed the problem as a binary classification task. After evaluating several algorithms, a **Gradient Boosting Classifier (LightGBM)** was selected for its high performance. The model was trained on a year of historical data and rigorously validated using k-fold cross-validation.
5.  **Operationalization**: The model was designed to run monthly, generating a risk-stratified list of members across all three measures. The output included a `risk_score` and the top reasons for that risk, which was then ingested by our care management platform to create automated worklists for the clinical outreach team.

## 4. Technical Stack

* **Language**: Python
* **Core Libraries**:
    * `pandas` & `numpy` for data manipulation and calculation
    * `scikit-learn` for modeling, feature processing, and evaluation
    * `matplotlib` & `seaborn` for data visualization and feature importance plotting

## 5. Key Outcomes & Business Impact

The model was highly successful in predicting future non-adherence, leading to significant improvements in both clinical and financial metrics.

* **Model Performance**: Achieved an **AUC of 0.88**, demonstrating a strong ability to distinguish between members who would remain adherent versus those who would not.
* **Clinical Impact**: The targeted outreach program, powered by the model's predictions, drove a **7-percentage-point increase** in the blended PDC rate across the MAD, MAH, and MAC cohorts within six months.
* **Business Impact**: This performance uplift was a primary contributor to elevating our medication adherence measures from an average **3-Star to a 4-Star rating**. This directly secured an estimated **$15 million in additional annual revenue** through the CMS Quality Bonus Program and strengthened our competitive position in the market.

## 6. How the Code Works

The `medication_adherence_model.py` script simulates the end-to-end process:

1.  **Simulates Data**: Creates realistic member and pharmacy claims data for the three medication classes.
2.  **Calculates PDC**: Defines a function to calculate the Proportion of Days Covered (PDC) and engineer features from the claims data.
3.  **Builds Model**: Trains a Random Forest Classifier to predict if a member's future PDC will be >= 80%.
4.  **Evaluates Performance**: Measures the model's accuracy and AUC score and identifies the most important predictive features.
5.  **Generates Output**: Produces a `member_adherence_risk_list.csv` file, which ranks members by their risk of non-adherence. This file is the final product intended for the clinical outreach team.

## 7. Sample Data & HIPAA Disclaimer

Due to the sensitive nature of patient information and strict **HIPAA (Health Insurance Portability and Accountability Act)** regulations, the actual member data used for this project cannot be shared.

The Python script in this repository generates synthetic data that is structurally and statistically representative of the real-world data used. The final output of the model is a CSV file (`member_adherence_risk_list.csv`) that looks like the following sample. This list is prioritized by the `risk_of_non_adherence_score`, allowing clinical teams to focus their efforts effectively.

| member_id | measure | age | num_chronic_conditions | risk_of_non_adherence_score |
| :-------- | :------ | :-: | :--------------------: | :-------------------------: |
| 1742      |   MAH   | 78  |           4            |            0.91             |
| 345       |   MAC   | 67  |           5            |            0.88             |
| 1021      |   MAD   | 81  |           3            |            0.85             |
| 85        |   MAH   | 55  |           2            |            0.76             |
| 1953      |   MAC   | 72  |           1            |            0.65             |

## 8. How to Use

1.  Ensure you have Python and the required libraries installed:
    ```
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
2.  Clone the repository:
    ```
    git clone <repository-url>
    ```
3.  Navigate to the directory and run the script:
    ```
    python medication_adherence_model.py
    ```
4.  The script will print the model evaluation results to the console, display a feature importance plot, and generate the `member_adherence_risk_list.csv` file in the same directory.

## 9. Key Terms & Definitions

* **AUC (Area Under the Curve):** A score that shows how well the model can make a correct prediction. A score of 1.0 is perfect, and 0.5 is no better than a random guess. Our score of 0.88 means the model is highly accurate.
* **CMS (Centers for Medicare & Medicaid Services):** The U.S. government agency that runs the Medicare program and sets the quality standards (Star Ratings) for health plans.
* **Cohort:** A specific group of people who share a common trait. In this project, we look at cohorts like "all members taking a diabetes medication."
* **HIPAA:** A U.S. federal law that protects the privacy of your personal health information. This is why we use fake data for this public project.
* **LightGBM:** The specific type of advanced algorithm used to build our predictive model. It's known for being very fast and accurate.
* **PDC (Proportion of Days Covered):** The main way to measure if someone is taking their medicine as prescribed. It’s the percentage of days they had their medication available. A score of 80% or higher is the goal.
* **QBP (Quality Bonus Payments):** Financial bonuses that CMS pays to health plans that earn high Star Ratings. Better care leads to better ratings, which leads to these bonuses.
* **Star Ratings:** A grade from 1 to 5 stars that CMS gives to health plans based on their quality and performance. More stars are better and help patients choose the best plan.
