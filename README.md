# Project Machine Learning

The goal of the project is to develop a model capable of predicting expected returns on the basis of a ranking of scores assigned to all the evaluated stocks. 

## List of contents

- [Files](https://github.com/Agablue-red/Machine-Learning#files)
- [Description](https://github.com/Agablue-red/Machine-Learning#description)
	- [Data preparation](https://github.com/Agablue-red/Machine-Learning#data-preparation)
	- [Time Series analysis](https://github.com/Agablue-red/Machine-Learning#time-series-analysis) 
	- [Linear regression](https://github.com/Agablue-red/Machine-Learning#linear-regression)
	- Advanced modelling

- [Technologies](https://github.com/Agablue-red/Machine-Learning#technologies)
- [Authors](https://github.com/Agablue-red/Machine-Learning#authors)

## Files

### Code files

- [Data preparation](https://github.com/Agablue-red/Machine-Learning/blob/master/code/data_preparation.ipynb)
- [Time Series analysis](https://github.com/Agablue-red/Machine-Learning/blob/master/code/time-series.ipynb)
- [Linear regression](https://github.com/Agablue-red/Machine-Learning/blob/master/code/regression.ipynb)
- [LR using the logarithmic rate of return](https://github.com/Agablue-red/Machine-Learning/blob/master/code/regression2.ipynb)
- Advanced modelling 

### Specifications

- [Data preparation](https://github.com/Agablue-red/Machine-Learning/blob/master/PDF/data_preparation.pdf)
- [Time Series analysis](https://github.com/Agablue-red/Machine-Learning/blob/master/PDF/time-series.pdf)
- [Linear regression](https://github.com/Agablue-red/Machine-Learning/blob/master/PDF/regression.pdf)
- [LR using the logarithmic rate of return](https://github.com/Agablue-red/Machine-Learning/blob/master/PDF/regression2.ipynb)
- Advanced modelling 

## Description

### Data preparation

The dataset consists of date, stock index, sector, rating, closing price, and rate of return. 

```python
    | Date      | symbol | sector                | score    | close      | return_rate |
    |2022-02-09 | PEP    | Consumer Non-Durables | 0.701507 | 171.940002 | -0.003189   |
    |2022-02-09 | SSNC   | Technology Services   | 0.701123 | 82.419998  |  0.025890   |
    |2022-02-09 | GEF    | Process Industries    | 0.697954 | 56.930000  | -0.001753   |
    |2022-02-09 | DPZ    | Consumer Services     | 0.697741 | 444.760010 |  0.015272   |
    |2022-02-09 | LIFZF  | Non-Energy Minerals   | 0.695644 | 34.410000  |  0.069630   |
```

The main dataset is a combination of two datasets. The first set comes directly from the lecturer and includes expert assessment of the company. The second set was downloaded by the authors at Yahoo finance and used to calculate the rate of return.

During data preparation, `397` stock indices were removed because the symbols of companies in both sets hadn't match.

    Number of all unique symbols: 1804
    Number of missing symbols: 397
    Number of symbols in dataset: 1407

As a result of removing missing symbols and closing prices, the dataset has `30551` rows.

    Old data frame length: 37360
    New data frame length: 30551
    Number of rows deleted: 6809

The mean score for this dataset is `0.73`, while mean closing price is `101.3` and mean return rate is `0.004`.
 
```python
    #basic statistics
    data.describe()
    
    |      | score    | close       | return_rate |
    |count | 30551    | 30551       |  30551      |
    |mean  | 0.731206 | 101.353658  |  0.003849   |
    |std   | 0.117692 | 2627.016498 |  0.044643   |
    |min   | 0.413554 | 0.020000    | -0.951550   |
    |25%   | 0.653428 | 26.072500   | -0.016298   |
    |50%   | 0.741474 | 44.770000   |  0.002865   |
    |75%   | 0.813471 | 73.910004   |  0.023672   |
    |max   | 0.987225 | 453000      |  0.632911   |
```

### Time Series analysis

The time-series begins in `2004-02-10` and ends in `2022-02-10`.  The analysis involves data from `2006`. The parametric measures are the closing price and the rate of return.

#### Visualize the stock’s weekly closing price and rate of return

![close_price](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/close_price.png)

The process above is not stationary, because the mean is not constant through time.

![rate_of_return](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/rate_of_return.png)

It has a lot of deviations whereas seasonality is not observed. The highest deviance was observed in `2008` with a weekly return of `-17%`. In year `2020` the biggest fluctuations on return rates were noted with in between `-14%` and `17%`.

#### Dickey-Fuller test

Dickey-Fuller test can be used to determine whether or not a series has a unit root, and thus whether or not the series is stationary (H<sub>0</sub>).


```python
    Results od Dickey-Fuller Test
             Values                       Metric
    0    -20.962812              Test Statistics
    1      0.000000                      p-value
    2     38.000000             No. of lags used
    3  27883.000000  Number of observations used
    4     -3.430585          critical value (1%)
    5     -2.861644          critical value (5%)
    6     -2.566825         critical value (10%)
```
We can rule out the Null hypothesis because the p-value is smaller than 0.05. Additionally, the test statistics exceed the critical values. As a result, the data is **nonlinear**.

#### Estimating trend

To reduce the magnitude of the values and the growing trend in the series use log of the series. 

![log_close_price](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/log_close_price.png)

Visualization of logarithmic closing prices. The falls are the results of crises. The trend is growing.

#### Rolling statistics

![log_scale](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/log_scale.png)

The result of smoothing by the previous quarter can hardly see a trend, because it is too close to actual curve. In addition the increasing mean and standard deviation may be seen, indicating that our series isn’t stationary.

#### SARIMAX (3, 0, 3) model

    train_data = df_log['2010':'2020']
    test_data = df_log['2021':'2022']

![test_train](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/test_train.png)

Model

![ARMA](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/ARMA.png)

**Standardized residual**

The first chart shows the grouping of volatility. The residual errors appear to have a uniform variance and fluctuate between -2 and 2.

**Histogram plus estimated density**

The density plot suggests a normal distribution with a mean of zero. What is the excess kurtosis with long tails.

**Normal Q-Q**

Normal Q-Q shows deviations from the red line, both at the beginning and at the end, what would indicate a skewed distribution with long tails.

**Correlogram**

The fourth graph shows the linear relationships in the first lag. As a result, need to add more Xs (predictors) to the model.

![ARMA_model](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/ARMA_model.png)

The best model with the lowest `AIC = 50337.918` was selected.

Do each coefficient is statistically significant?

The tests are:

-   Null Hypothesis: each coefficient is NOT statistically significant.
-   Alternate Hypothesis: coefficient is statistically significant (p-value of less than 0.05).

**Each parameter is statistically significant.**

Do the residuals are independent (white noise)?

The Ljung Box tests that the errors are white noise.

The probability (`0.23`) is above 0.05, so  **we can’t reject the null that the errors are white noise**.

Do residuals show variance?

Heteroscedasticity tests that the error residuals are homoscedastic or have the same variance.

Test statistic of `1.82` and a p-value of 0.00, which means we reject the null hypothesis and  **residuals show variance**.

Did data is normally distributed?

Jarque-Bera tests for the normality of errors.

Test statistic of `104150.24` with a probability of `0`, which means we reject the null hypothesis, and  **the data is not normally distributed**.

In addition results show:

-   Negative skewness - left side asymmetry (long tail on the left side).
-   Excess kurtosis - results fluctuate around a mean

### Linear Regression

#### Training and test sets

  ![ARMA_model](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/rr_test_train.png)

Training set involves data `from 2010 to 2020` while testing set includes the year `2021`.

Training set consist of `19797` observations whereas test set has `2021` observations  


#### Dummy regression


    Coefficient of determination: 0.0

0% represents a model that does not explain any of the variation in the response variable around its mean.

    Coefficient of determination (Adjusted R2): -0.00140
    Mean absolute error (MAE): 0.00214
    Residual sum of squares (MSE): 0.00178
    Root mean squared error (RMSE): 0.04214

#### Linear Regression

$f(x) = - 0.027x + 0.024$

    Coefficient of determination: 0.005437296983874185

Model explains only `0.0054` of the variation in the response variable around its mean.

    Coefficient of determination (Adjusted R2): -0.00431
    Mean absolute error (MAE): 0.03100
    Residual sum of squares (MSE): 0.00178
    Root mean squared error (RMSE): 0.04220

#### Comparison between dummy regression and linear regression in combination with observations from test set.

![Comparison](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/comparision_dummy-linear.png)

Model does not explain any of the variation in the response variable around its mean.

Linear regression is marginally better than dummy regression.

Both models are not well fit.

So project group used **the logarithmic rate of return** but measures of fit a model are worst than using the simple rate of return.

    Coefficient of determination: 0.002064
    Mean absolute error (MAE): 0.01353
    Residual sum of squares (MSE): 0.00038
    Root mean squared error (RMSE): 0.01957

### Advanced modelling

> Not yet

 
## Technologies

Project is created in Python with:

* matplotlib version: 3.3.4
* numpy version: 1.20.1
* pandas version: 1.2.4
* pmdarima version: 1.8.5
* scikit-learn version: 0.24.1
* seaborn version: 0.11.1
* statsmodels version: 0.12.2
* yfinance version: 0.1.70

  

## Authors

  

Wiktoria Ekwińska

  

Bartek Gimzicki

  

Agnieszka Pijaczyńska