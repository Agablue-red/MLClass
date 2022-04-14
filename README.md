# Project Machine Learning

The goal of the project is to develop a model capable of predicting expected returns on the basis of a ranking of scores assigned to all the evaluated stocks.

## List of contents
 - [Files](https://github.com/Agablue-red/Machine-Learning#files)
 - [Description](https://github.com/Agablue-red/Machine-Learning#description)
	 - [Data preparation](https://github.com/Agablue-red/Machine-Learning#data-preparation)
	 - [Initial modelling](https://github.com/Agablue-red/Machine-Learning#initial-modelling)
	 - [Advanced modelling]()
 - [Technologies](https://github.com/Agablue-red/Machine-Learning#technologies)
 - [Authors](https://github.com/Agablue-red/Machine-Learning#authors)
.
## Files

 - [Data preparation](https://github.com/Agablue-red/Machine-Learning/blob/master/code/data_preparation.ipynb)
 - [Initial modelling](https://github.com/Agablue-red/Machine-Learning/blob/master/code/regression.ipynb)
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
    Number of symbols in dataset:  1407

As a result of removing missing symbols and closing prices, the dataset has `30551` rows.

    Old data frame length: 37360 
    New data frame length: 30551 
    Number of rows deleted:  6809

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
   
### Initial modelling

#### Plotting Weekly Return
The time-series begins in `2004-02-10` and ends in `2022-02-10`.

![Return rate](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/return_rate.png)

The figure shows weekly return rates from 2004 to 2022. It has a lot of deviations whereas seasonality is not observed. The highest deviance was observed in 2004 with a weekly return of `-95%`. In years from 2009 and 2010 the biggest fluctuations on return rates were noted with in between `-40%` and `60%`.

#### Training and test sets

Training set involves data from 2010 to 2020 while testing set includes the year 2021.
Test set has `2021` observations whereas training set consist of `19797` observations.

#### Linear Regression

    lm = LinearRegression().fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    lm.score(X_train, y_train)
    0.005437296983874185

> Note

    Mean squared error (linear model): 0.002
    Median absolute error (linear model): 0.023

#### Dummy regression

    reg_dummy = DummyRegressor(strategy = 'mean').fit(X_train, y_train) 
    y_pred_dum = reg_dummy.predict(X_test)
    reg_dummy.score(X_train, y_train)
    0.0

> Note

    Mean squared error (linear model): 0.002
    Median absolute error (linear model): 0.023

#### Comparison between dummy regression and linear regression in combination with observations from testing sets.
![Comparison](https://raw.githubusercontent.com/Agablue-red/Machine-Learning/master/image/comparision_dummy-linear.png)

> Note

### Advanced modelling
> Not yet

## Technologies
Project is created in Python with:
* yfinance version: 0.1.70
* pandas version: 1.2.4
* numpy version: 1.20.1
* scikit-learn version: 0.24.1

##  Authors

Wiktoria Ekwińska

Bartek Gimzicki

Agnieszka Pijaczyńska