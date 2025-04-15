# Emotion Analysis

## Prerequisites

Ensure you have the following installed on your system before proceeding:

- Python (>= 3.7)
- Anaconda or Miniconda
- A compatible IDE or text editor (optional but recommended, e.g., VS Code or PyCharm)

## Setup Instructions

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/Parth-RK/emotion_analysis.git
   cd emotion_analysis
   ```

2. **Create a Conda Environment** (Optional but recommended)

   ```bash
   conda create -n emotion_env python=3.10
   ```
   ```bash
   # Activate the conda environment
   conda activate emotion_env
   ```

3. **Install Dependencies**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install cupy
   pip install -r requirements.txt
   pip install spacy
   pip install torch torchtext pandas numpy spacy matplotlib seaborn tqdm pandarallel nltk scikit-learn
   python -m spacy download en_core_web_md
   ```

4. **Verify Installation**

   ```bash
   pip list
   ```

## Additional Notes

- To deactivate the conda environment, run:
  ```bash
  conda deactivate
  ```

## My Specs:
-Nvidia GeForce MX150 2GB
-Nvidia Graphics Driver 572.83
-Cuda toolkit 11.7
-Python 3.10.0
-PyTorch version: 2.6.0+cu118
-cupy-cuda11x
