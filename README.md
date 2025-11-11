<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Kairovia Weather Prediction System</h3>
  <p align="center">
    Real-time weather forecasting using ensemble machine learning models for prediction markets
    <br />
    <a href="https://github.com/PraneethO/kairovia"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/PraneethO/kairovia/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/PraneethO/kairovia/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

Kairovia is a sophisticated weather prediction system designed for real-time forecasting applications, particularly prediction markets. The system leverages an ensemble of machine learning models to predict daily high temperatures with high accuracy using historical weather data from NOAA stations.

### Key Features

- **Ensemble Learning**: Combines LightGBM gradient boosting and heteroscedastic neural networks for robust predictions
- **Advanced Feature Engineering**: Extracts 200+ temporal and geospatial features including FFTs, lag variables, and rolling statistics
- **Uncertainty Quantification**: Neural network model provides both point predictions and uncertainty estimates
- **Real-time Data Pipeline**: Automated data fetching from NOAA weather stations via Meteostat API
- **Production-Ready**: Trained models and scalers are saved for deployment in inference pipelines

The system processes hourly weather observations (temperature, dewpoint, pressure, wind direction, wind speed, precipitation) and predicts the maximum daily temperature, making it suitable for applications requiring accurate short-term weather forecasts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][Python.org]][Python-url]
- [![TensorFlow][TensorFlow.org]][TensorFlow-url]
- [![LightGBM][LightGBM.org]][LightGBM-url]
- [![scikit-learn][scikit-learn.org]][scikit-learn-url]
- [![Pandas][Pandas.org]][Pandas-url]
- [![NumPy][NumPy.org]][NumPy-url]
- [![Matplotlib][Matplotlib.org]][Matplotlib-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

This section will guide you through setting up the project locally.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository

   ```sh
   git clone https://github.com/github_username/kairovia.git
   cd kairovia
   ```

2. Install required packages

   ```sh
   pip install numpy pandas scikit-learn lightgbm tensorflow matplotlib joblib meteostat python-dateutil
   ```

3. Download weather data

   ```sh
   python download.py
   ```

   This will fetch historical weather data from the Austin Bergstrom International Airport (KAUS) station and save it to `data.csv`.

4. (Optional) Find alternative weather stations
   ```sh
   python find_stations.py
   ```
   Use this script to discover weather stations near specific coordinates if you want to use data from a different location.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

### Training the Model

To train the ensemble model on your weather data:

```sh
python model.py
```

This script will:

1. Load and preprocess the weather data from `data.csv`
2. Engineer temporal and geospatial features
3. Train a LightGBM model with quantile regression
4. Train a heteroscedastic neural network
5. Create an ensemble prediction combining both models
6. Evaluate performance and generate visualization plots
7. Save trained models (`hetero_nn.keras`, `lgb_model.txt`) and scalers (`scaler_X.joblib`, `scaler_y.joblib`)

### Model Configuration

Key parameters can be adjusted in `model.py`:

- `INPUT_HOURS`: Number of hours of historical data to use (default: 12)
- `SPLIT_DATE`: Date to split train/test sets (default: None, uses 80/20 split)
- `LGB_PARAMS`: LightGBM hyperparameters
- `EPOCHS`: Number of training epochs for neural network
- `BATCH_SIZE`: Training batch size

### Output

The script generates:

- Performance metrics (MAE, MSE) for each model component
- Error distribution plots and analysis
- Saved model files for inference deployment

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model Architecture

The system uses a two-model ensemble approach:

### 1. LightGBM Gradient Boosting

- Point prediction model for baseline forecasts
- Quantile regression models (10th, 50th, 90th percentiles) for uncertainty bounds
- Handles non-linear relationships and feature interactions efficiently

### 2. Heteroscedastic Neural Network

- Deep learning model that predicts both mean (μ) and variance (σ²)
- Architecture: 256 → 128 dense layers with batch normalization and dropout
- Uses negative log-likelihood loss for heteroscedastic regression
- Provides uncertainty estimates alongside point predictions

### Ensemble Strategy

- Final prediction: 50% LightGBM + 50% Neural Network mean prediction
- Quantile predictions combine LightGBM quantiles with neural network uncertainty

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

### Feature Engineering

The system extracts comprehensive features from raw weather observations:

**Temporal Features:**

- Cyclical encoding of hour and day-of-year (sine/cosine transformations)
- Rolling statistics (24-hour rolling means)
- Lag variables from previous days
- Linear trend detection in temperature sequences

**Derived Meteorological Features:**

- Relative humidity calculated from temperature and dewpoint
- Pressure normalization and interpolation
- Wind vector components

**Statistical Features:**

- Previous day statistics (max, min, mean temperature)
- Rolling window aggregations
- Temporal patterns and trends

All features are standardized using scikit-learn's `StandardScaler` before model training.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [ ] Add support for multi-location predictions
- [ ] Extend to multi-day ahead forecasts
- [ ] Implement automated retraining pipeline
- [ ] Add model monitoring and drift detection

See the [open issues](https://github.com/github_username/kairovia/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

PraneethO - potthi [at] berkeley [dot] edu

Project Link: [https://github.com/PraneethO/kairovia](https://github.com/PraneethO/kairovia)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- [Meteostat](https://meteostat.net/) - Weather data API and historical climate data
- [NOAA](https://www.noaa.gov/) - National Oceanic and Atmospheric Administration for weather station data
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/github_username/kairovia.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/kairovia/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/kairovia.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/kairovia/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/kairovia.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/kairovia/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/kairovia.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/kairovia/issues
[license-shield]: https://img.shields.io/github/license/github_username/kairovia.svg?style=for-the-badge
[license-url]: https://github.com/github_username/kairovia/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[TensorFlow.org]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
[LightGBM.org]: https://img.shields.io/badge/LightGBM-FF6F00?style=for-the-badge&logo=lightgbm&logoColor=white
[LightGBM-url]: https://lightgbm.readthedocs.io/
[scikit-learn.org]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/
[Pandas.org]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[NumPy.org]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib.org]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
