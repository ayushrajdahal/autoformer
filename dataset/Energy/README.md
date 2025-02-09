# Electricity Load Forecasting

[Short-term electricity load forecasting (Panama case study)](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting/data)

## Context

This is a useful dataset to train and test Machine Learning forecasting algorithms and compare results with the official forecast from weekly pre-dispatch reports. The following considerations should be kept to compare forecasting results with the weekly pre-dispatch forecast:

Saturday is the first day of each weekly forecast; for instance, Friday is the last day.
A 72 hours gap of unseen records should be considered before the first day to forecast. In other words, next week forecast should be done with records until each Tuesday last hour.
Data sources provide hourly records. The data composition is the following:

Historical electricity load, available on daily post-dispatch reports, from the grid operator (CND).
Historical weekly forecasts available on weekly pre-dispatch reports, both from CND.
Calendar information related to school periods, from Panama's Ministery of Education.
Calendar information related to holidays, from "When on Earth?" website.
Weather variables, such as temperature, relative humidity, precipitation, and wind speed, for three main cities in Panama, from Earthdata.

## Content

The original data sources provide the post-dispatch electricity load in individual Excel files on a daily basis and weekly pre-dispatch electricity load forecast data in individual Excel files on a weekly basis, both with hourly granularity. Holidays and school periods data is sparse, along with websites and PDF files. Weather data is available on daily NetCDF files.

For simplicity, the published datasets are already pre-processed by merging all data sources on the date-time index:

A CSV file containing all records in a single continuous dataset with all variables.
A CSV file containing the load forecast from weekly pre-dispatch reports.
Two Excel files containing suggested regressors and 14 training/testing datasets pairs as described in the PDF file.

## What do the variables mean?

`nat_demand`: National electricity load
`T2M`: Temperature at 2 meters
`QV2M`: Relative humidity at 2 meters
`TQL`: Liquid precipitation
`W2M`: Wind speed at 2 meters

And after the underscore is the city

`toc`: Tocumen, Panama city
`san`: Santiago city
`dav`: David city

The rest of variables:

`Holiday_ID`: Unique identification number integer
`holiday`: Holiday binary indicator (1=holiday, 0=regular day)
`school`: School period binary indicator (1=school, 0=vacations)

## Acknowledgements

Aguilar Madrid, Ernesto (2021), “Short-term electricity load forecasting (Panama case study)”, Mendeley Data, V1, doi: 10.17632/byx7sztj59.1