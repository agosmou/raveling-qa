
# Raveling App

A Streamlit application for predicting and QA'ing the severity of Raveling using a pre-trained random forest machine learning model

## Application Features

- **Upload Ground Truth CSV (Optional)**: Upload an existing ground truth CSV file to compare with predictions.
- **Upload Images**: Upload multiple images to get severity predictions.
- **View Predictions**: Display each image with its predicted severity.
- **Input Ground Truth Values**: Input or modify ground truth severity values for each image.
- **Download Ground Truth CSV**: Download the CSV containing image titles and their corresponding severity values.
- **Compare with Ground Truth**: If ground truth data is provided, view a comparison table between predictions and actual values.

## Instructions

Please the `guide.md` file or the PDF for a comprehensive tutorial on using the application.

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

4. **Testing the app**
   Use the `test_files/` directory for some artifacts you can use to test the application

## Running on vm

1. simply make sure the `./run.sh` has executable permissions and run it within the vm


## Running with Docker Compose

1. ```bash
   docker compose up
   ```


**Essentially running Docker under the hood so you could do:**

1. **Build the Docker image** (optional)

   ```bash
   docker build -t streamlit .
   ```

2. **Run the Docker container**

   ```bash
   docker run -p 80:80 streamlit
   ```

## TODO

- [X] Add a "Run predictions button" to decouple expensive operations from image upload

- [X] Modify the feature extractor based on Xinan's script

- [X] Setup Docker to deploy app

- [X] Add a modifiable instructional document for DOT engineer

- [ ] Add auth to the application
