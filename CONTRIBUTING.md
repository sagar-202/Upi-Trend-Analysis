# Contributing to UPI Transaction Analysis Dashboard

Thank you for your interest in contributing! We welcome pull requests, bug reports, and feature requests from the community.

## How to Contribute

### 1. Reporting Bugs
If you find a bug or experience an unexpected error while running the Streamlit dashboard or ML pipeline, please open an Issue. Provide:
- Your operating system and Python version
- Steps to reproduce the error
- A snippet of the traceback logs

### 2. Suggesting Features
Have an idea to improve the machine learning recall? Want to add a new visualization to the dashboard?
Open an Issue labeled `enhancement` and describe your idea.

### 3. Submitting Pull Requests
If you want to contribute code, follow these steps:
1. **Fork** the repository.
2. **Clone** your fork locally.
3. Create a **feature branch**: `git checkout -b feature/your-feature-name`
4. Make your changes in the appropriate directory (`src/` for ML logic, `dashboard/` for Streamlit UI).
5. **Test** your changes locally: `streamlit run dashboard/app.py`
6. **Commit** your code: `git commit -m "Add feature XYZ"`
7. **Push** to your fork: `git push origin feature/your-feature-name`
8. Open a **Pull Request (PR)** against our `main` branch.

## Development Setup
```bash
# Clone the repository
git clone https://github.com/sagar-202/Upi-Trend-Analysis.git

# Navigate to project directory
cd Upi-Trend-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Folder Structure Rules
- Do not put machine learning training scripts inside `dashboard/`. Keep UI and ML separate!
- All reusable core ML modules must go inside `src/`.
- All raw datasets go into `data/raw/` (do not commit large datasets to GitHub).
- Exported visual assets should be saved to `assets/images/`.

Thank you for making this project better!
