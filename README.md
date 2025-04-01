# MLOps Lab Repository

This repository contains lab exercises and projects for MLOps (Machine Learning Operations) coursework. The labs focus on implementing MLOps best practices, including experiment tracking, model versioning, and workflow automation.

## Repository Structure

```
mlops/lab/
├── lab1/                    # Lab 1 exercises
│   └── Lab_1_development.ipynb
├── lab2/                    # Lab 2 exercises
├── lab3/                    # Lab 3 exercises
├── ...                      # Other Lab exercises
├── requirements.txt         # Project dependencies
└── README.md               # Explanation of this Repository
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops/lab
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project uses the following main dependencies:
- MLflow (2.15.1) - For experiment tracking and model management
- NumPy (1.26.4) - For numerical computations
- Pandas (2.2.2) - For data manipulation and analysis
- scikit-learn (1.5.1) - For machine learning algorithms
- Hyperopt - For hyperparameter optimization

## Labs Overview

### Lab 1
- Focus on development environment setup
- Introduction to MLflow for experiment tracking
- Basic model development workflow

### Lab 2
- Advanced MLOps practices
- Model deployment and serving
- Workflow automation

## Usage

Each lab contains Jupyter notebooks with detailed instructions and exercises. Follow the instructions in each notebook to complete the lab exercises.

## License

This project is part of academic coursework. All rights reserved.
