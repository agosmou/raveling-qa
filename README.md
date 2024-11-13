
# Raveling App

A Streamlit application for predicting and QA'ing the severity of Raveling using pre-trained machine learning models.

## Application Features

- **Upload Ground Truth CSV (Optional)**: Upload an existing ground truth CSV file to compare with predictions.
- **Upload Images**: Upload multiple images to get severity predictions.
- **View Predictions**: Display each image with its predicted severity.
- **Input Ground Truth Values**: Input or modify ground truth severity values for each image.
- **Download Ground Truth CSV**: Download the CSV containing image titles and their corresponding severity values.
- **Compare with Ground Truth**: If ground truth data is provided, view a comparison table between predictions and actual values.

## Running Locally

1. **Set up a virtual environment**

   ```bash
   cd app/
   # For macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Running with Docker (Work In Progress)

1. **Build the Docker image** (optional)

   ```bash
   docker build -t streamlit .
   ```

2. **Run the Docker container**

   ```bash
   docker run -p 8501:8501 streamlit
   ```

For more details on deploying with Docker, see the [Streamlit Docker Documentation](https://docs.streamlit.io/deploy/tutorials/docker).
