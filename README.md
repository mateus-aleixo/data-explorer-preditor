# Data Explorer & Predictor

An interactive web application built with Streamlit that allows users to upload, explore, visualize, and build basic prediction models from their data. Perfect for quick data analysis and prototyping machine learning models.

## Installation

```bash
# Clone the repository
git clone https://github.com/mateus-aleixo/data-explorer-predictor.git
cd data-explorer-predictor

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`
3. Upload a CSV file to begin exploring your data
4. Use the interactive interface to:
   - Visualize data distributions and relationships
   - Filter data based on various criteria
   - Build and test prediction models
   - Export processed data
