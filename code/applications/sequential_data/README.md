# Sequential Data

## About
This part of the repository is concerned with __Sequential Data__, including Time Series.

Keywords: Sequential Data, Time Series, Linear Dynamical Systems, State Space Models, ARMA, ARIMA, SARIMAX

## Introduction

A Time Series can be either _univariate_ or _multivariate_.

__Time Series Forecasting__ can be: (i) _One-Step ahead_, (ii) _Multi-Step ahead_ or (iii) _Dynamic_.

Multi-Step ahead Time Series Forecasting can have three main strategies:

1. _Direct Strategy_*, when creating many separate models, one for each time step ahead.

1. _Recursive Strategy_, when using predicted (and possibly actual) values as input to the (single) model.

1. _Direct-Recursive Hybrid Strategies_*, when combining the two methods.

1. _Multiple Output Strategy_, when using only actual (and not predicted) values as input to the (single) model.

\* Strategies 1 and 3 are much more _computational intensive_ (than the rest ones), since a separate model needs to be trained during the Training Phase.

## Sub-tasks
There are various sub-tasks that fall under this specific domain and some of them are the following:

- [One-Step ahead Time Series Forecasting](one_step_time_series_forecasting)
- [Multi-Step ahead Time Series Forecasting checking, performance of only non-overlapping windows](multi_step_time_series_forecasting)
- [Multi-Step ahead Time Series Forecasting, checking performance of all overlapping windows](multi_step_time_series_forecasting_steps)
- [Dynamic Time Series Forecasting](dynamic_time_series_forecasting)

## Theory
Regarding theory of Sequential Data, you can check the following pages on my personal wiki:

- [Linear Dynamical Systems](https://wiki.kourouklides.com/wiki/Linear_Dynamical_System)
- [Statistical Signal Processing](https://wiki.kourouklides.com/wiki/Statistical_Signal_Processing)

The wiki contains curated lists of online and offline resources (e.g. books, papers, URL links) about these topics.

## Datasets
 - Monthly sunspots

## Blog posts

TODO: write them
