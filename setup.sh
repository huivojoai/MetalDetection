#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y libfreetype6-dev libjpeg-dev zlib1g-dev libtiff-dev liblcms2-dev libwebp-dev

# Install Python dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
