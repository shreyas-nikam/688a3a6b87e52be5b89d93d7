# QuLab: Model Risk Materiality and Impact Simulator

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

This repository hosts a multi-page Streamlit application designed as a hands-on lab to explore the critical concept of **model risk materiality** in financial institutions. Users can interactively simulate various model scenarios to understand how different characteristics – such as complexity, data quality, usage frequency, and business impact – contribute to overall model risk and influence the required rigor of model risk management.

The application draws inspiration from the principles outlined in regulatory guidance for model risk management, emphasizing that the level of oversight should be commensurate with a model's potential impact on a bank's financial condition.

## 🌟 Features

The application is structured into three main pages, each offering distinct functionalities:

### Page 1: Data Generation & Validation
*   **Synthetic Data Generation**: Generate a customizable number of synthetic financial models, each with randomly assigned attributes for complexity, data quality, usage frequency, and business impact.
*   **Data Overview**: View the head of the generated dataset and its `info()` summary.
*   **Robust Data Validation**: Perform automated checks for expected column names, data types, uniqueness of identifiers, and presence of missing values in critical fields.
*   **Summary Statistics**: Display descriptive statistics for numerical columns and value counts for categorical columns to understand data distribution.

### Page 2: Risk Score Calculation & Guidance
*   **Configurable Risk Weights**: Interactively adjust the weights for each model risk factor (complexity, data quality, usage frequency, business impact) via sidebar sliders.
*   **Model Risk Score Calculation**: Compute a composite `model_risk_score` for each synthetic model using a weighted sum methodology.
*   **Materiality-Based Guidance**: Assign a `management_guidance` level ('Standard Oversight', 'Enhanced Scrutiny', 'Rigorous Management') to each model based on its calculated risk score and predefined materiality thresholds.
*   **Display Results**: View the DataFrame with calculated scores and guidance, along with a distribution of the assigned guidance levels.

### Page 3: Visualizations & Sensitivity Analysis
*   **Model Risk Distribution (Bar Chart)**: Visualize the average model risk score across different `business_impact_category` levels to highlight materiality.
*   **Model Risk Heatmap**: Explore the interaction between `complexity_level` and `business_impact_category` by visualizing the average model risk score in a heatmap.
*   **Relationship Scatter Plot**: Analyze the relationship between a continuous parameter (e.g., `data_quality_index`) and `model_risk_score`, optionally colored by another categorical factor (`complexity_level`).
*   **Interactive Sensitivity Analysis**: Conduct sensitivity tests by varying a single model characteristic (e.g., `complexity_level` or `data_quality_index`) for a hypothetical base model, showing its impact on the `model_risk_score`.

## 🚀 Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+ (preferably 3.9 or 3.10)
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/qu-lab-model-risk.git
    cd qu-lab-model-risk
    ```
    (Replace `your-username/qu-lab-model-risk` with the actual repository path)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    streamlit
    pandas
    numpy
    plotly
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## 💡 Usage

1.  **Run the Streamlit application**:
    Ensure your virtual environment is activated and you are in the project's root directory (`qu-lab-model-risk`).
    ```bash
    streamlit run app.py
    ```

2.  **Access the application**:
    The command above will open the application in your default web browser (usually at `http://localhost:8501`).

3.  **Navigate and Interact**:
    *   Use the sidebar on the left to navigate between **"Page 1: Data Generation & Validation"**, **"Page 2: Risk Score & Guidance"**, and **"Page 3: Visualizations & Sensitivity Analysis"**.
    *   **Page 1**: Enter the desired number of synthetic models and click "Generate Synthetic Data". Observe the data validation outputs.
    *   **Page 2**: Adjust the weights for each model risk factor using the sliders in the sidebar. Click "Calculate Risk Scores" to see the impact.
    *   **Page 3**: Explore the pre-generated visualizations. For sensitivity analysis, select a parameter to vary and its values in the sidebar, then click "Run Sensitivity Analysis".

## 📁 Project Structure

```
qu-lab-model-risk/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # List of Python dependencies
├── application_pages/          # Directory containing individual page modules
│   ├── __init__.py             # Makes application_pages a Python package
│   ├── page1.py                # Data Generation & Validation logic
│   ├── page2.py                # Risk Score Calculation & Guidance logic
│   └── page3.py                # Visualizations & Sensitivity Analysis logic
└── README.md                   # This README file
```

## 🛠️ Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application and user interface.
*   **Pandas**: For data manipulation and analysis, especially with DataFrames.
*   **NumPy**: For numerical operations and array processing.
*   **Plotly**: For generating interactive and dynamic data visualizations.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/fix-bug-name`).
3.  Make your changes and ensure the code adheres to the existing style.
4.  Commit your changes (`git commit -m 'feat: Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request with a clear description of your changes.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (not included here, but would be a separate file in a real repo).

## 📧 Contact

For questions or feedback, please reach out to:
*   **QuantUniversity** - [www.quantuniversity.com](https://www.quantuniversity.com/)