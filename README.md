# Personalised Property Recommendation System

## Project Overview

The Personalised Property Recommendation System is designed to assist potential homebuyers in the UK real estate market by providing tailored property recommendations based on individual preferences and financial situations. This project integrates historical transaction data from HM Land Registry with current property listings from OnTheMarket to deliver customised property suggestions.

The system employs deep learning models to determine the most effective approach for property recommendation, framed as a classification task.

## Project Structure

The project is organized into several components:

- Data Collection and Cleaning
- Data Standardization and Merging
- Synthetic User Generation
- Model Building and Training
- Web Server for User Interaction

## Setup and Installation

1. Clone the repository to your local machine.

2. Install the necessary packages: `pip install -r requirements.txt`

3.  Set up the PostgreSQL database using Docker: `docker-compose up -d`

4. Run Alembic migrations to set up the database schema: `alembic upgrade head`

## Data Pipeline

Follow these steps to run the data pipeline:

1. Place the HM Land Registry file (`pp-2023.csv`) in the directory: `data/historical-data/`

2. Run the data cleaner: `python src/data_cleanser/data_cleanser_service.py`

3. Collect data from OnTheMarket: `python src/data_collector/data_collector_service.py`

4. Generate synthetic user data: `python src/data_generator/synthethic_user_generator.py`

5. Merge and standardize the data: `python src/data_standardiser/main.py`

6. Post-process the merged data: `python src/data_preparation/data_post_processor.py`

7. Build and train the model: `python src/model/main.py`

## Running the Web Server

After completing the data pipeline and model training, start the Flask server: `python src/webserver/app.py`

The web interface will be available at `http://localhost:5001`.

## Project Components

- `src/data_cleanser/`: Scripts for cleaning the HM Land Registry data
- `src/data_collector/`: Web scraping scripts for OnTheMarket data
- `src/data_exploration/`: Data exploration utilities
- `src/data_generator/`: Synthetic user data generation
- `src/data_preparation/`: Data post-processing scripts
- `src/data_standardiser/`: Data standardization and merging utilities
- `src/database/`: Database models and migration scripts
- `src/model/`: Machine learning model building, training, and evaluation
- `src/webserver/`: Flask web server for user interaction
