# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC For this project, I will be using the Prophet algorithm developed by meta. Read the docs [here](https://facebook.github.io/prophet/).
# MAGIC
# MAGIC The algorithm takes a bayesian approach to estimating the trend and seasonality of the time series to inference future values. 

# COMMAND ----------

# MAGIC %md
# MAGIC import packages for time series modeling

# COMMAND ----------

from prophet import Prophet

# COMMAND ----------

# MAGIC %md
# MAGIC Let's set up a dataframe with the time series for just one park for a simple example 

# COMMAND ----------

park = 'Arches'

df_arches = (
    df_visit.loc[(df_visit["park"] == park)]
    .rename(columns={"visitors": "y",
                     "ts": "ds"}) #rename for prophet
    .sort_values('ds')
)

df_arches

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base Model

# COMMAND ----------

# MAGIC %md
# MAGIC Let's do a base model with the standard parameters

# COMMAND ----------


model = Prophet() #instantiate a model
model.fit(df_arches) #fit onto the historical data

periods = 33 #forecast 33 months into the future (through 2026)
future = model.make_future_dataframe(periods=periods, freq = 'MS') #make a monthly future dataset
forecast = model.predict(future) #predict over future dates

# COMMAND ----------

# MAGIC %md
# MAGIC Let's explore the forecast

# COMMAND ----------

# MAGIC %md
# MAGIC Let's break down this chart:
# MAGIC - the solid blue line is the forecasted yhat value
# MAGIC - the light blue line is the uncertainty
# MAGIC - the black dots are the actual observed values
# MAGIC - time is on the x-axis and visitors is on the y-axis
# MAGIC
# MAGIC The model seems to do a poor job at the beginning and end of the series. This suggests a multiplicative growth instead of an additive growth could do better

# COMMAND ----------

plt = model.plot(forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we can see the model picked up the linerar trend of growing visitors. It also found the seasonlity we discovered during our eploratry analysis

# COMMAND ----------

plt = model.plot_components(forecast)

# COMMAND ----------

df_arches.count()
