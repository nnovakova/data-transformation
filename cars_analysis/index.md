+++
title="Car Market Trands - Data Exploration"
date=2020-12-18
draft = true

[taxonomies]
categories = ["Python", "Data Science", "Data Analysis","Data Preparation"]
tags = ["data analysis","data preparation","plotly","pandas"]

[extra]
toc = true
+++

In this article I wanted to show some basic data preparations and basic analytic. Also I want  to try plotly library for data visualisation

I looked for datasets I can easily manipulate to demonstrate the first prestep or ML  - data preparation and cleaning . That is why I started to surf for datasets I can somehow explain and easily interpret based on my experience. 
For this I used two Kaggle datasets:
- [UA Cars dataset](https://www.kaggle.com/dimakyn/vehicles-price-2020-ukraine)
- [US Cars dataset](https://www.kaggle.com/doaaalsenani/usa-cers-dataset)

I think it will be interesting to work with this set because of their visibility and simplicity .

The mission of the research to show the common trends in the total different car market and people preferences in different societies but not to estimate total sales and so on. I think that this research is representative enough for this, but not enough to assess this market.  

Whole source code you can find here : [GitHub](https://www.kaggle.com/doaaalsenani/usa-cers-dataset)

# Ukrainian Dataset

Let’s start with Ukrainian cars dataset. 

We will use pandas and numpy in this research.
```python
import pandas as pd
import numpy as np
ua_cars=pd.read_csv("cars/ua_vehicle_price.csv")
```
Look deeper into columns:

```python
ua_cars.columns
```
```python
Index(['brand', 'model', 'year', 'body', 'price$', 'car_mileage', 'fuel',
       'power', 'transmission'],
      dtype='object')
```
```python
ua_cars.head(10)
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 11.08.48.png", width=600, height=700, op="fit_width") }}


Calculate total amount of cars by brand.
```python
#Total count cars by brands
ua_cars["brand"].value_counts().head(10).plot(kind='bar')
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 11.10.12.png", width=600, height=700, op="fit_width") }}

There we see that Volksvagen is a top brand in UA. It make sense as for me, because some there are a lot of VW models are more available for majority of  car users in Ukraine.

Next let’s describe numeric and categorical variables in the dataset:
```python
#Statistic for numerical features
ua_cars.describe()
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 11.17.10.png", width=400, height=700, op="fit_width") }}

In this table we can see that data has some strange behavior and the reason for it can be outliers. for example the max power = 35, when mean = 2.125 and min = 0.1. So this dataset needs more transformations.
Car mileage looks better and we can accurately say that average car mileage is 215 176 km.


```python
#Statistic for categorical features
ua_cars.astype('object').describe().transpose()
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 11.17.19.png", width=400, height=700, op="fit_width") }}


Here we see again that top brand is VW, and top model is “Passat”. The top 

Very interesting observation is that the most  popular body type in Ukraine is a sedan. It is one more point to my assumption. Actually my guess is that in general Ukrainian drivers preferred cheap economical sedans, probably with manual transmission and diesel fuel. This is a perfect description of "People's car" (Volkswagen).

I would like to see all possible types of cars:

```python
# All possible transmission types in ua_cars DataFrame
ua_cars["transmission"].unique()
```
```python
array(['manual', 'automatic', 'other', 'typtronik', 'adaptive', nan],
      dtype=object)
```

There are some strange for me like 'typtronik'. But I am interesting more of manual and automatic types as more common.


It looks like that data in DF has incorrect types and should be transformed.

```python
# Colums info for ua_cars DataFrame
ua_cars.info()
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 11.50.09.png", width=400, height=700, op="fit_width") }}

Rename some columns for convinience:
```python
#Rename colums in ua_cars DataFrame
ua_cars.rename(columns={"price$":"price","car_mileage":"mileage"},inplace = True)
```
Here we see that price has object type, but it has to be numeric. For this we need to apply data type transformation.

```python
#As we see in info some colums should have another type as they have 
#Price is object/ should be integer or float
#Change pryce column type
ua_cars[["price"]] = ua_cars[["price"]].apply(pd.to_numeric,errors='coerce') 
ua_cars.head(5)
ua_cars.info()
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 12.00.16.png", width=300, height=700, op="fit_width") }}

So now it is time for some analytic. I would like to group data by some important from my side features and compare them. 

```python
#Group cars by transmission and years and calculate theirs count
ua_transmission = ua_cars.groupby(['transmission','year'])['year'].count().reset_index(name='count')
print(ua_transmission)
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 12.00.16.png", width=300, height=700, op="fit_width") }}

There I calculated the amount of cars sold each year and grouped by transmission type for each year.

It will be good to see how it looks in the graph. For visualisation I am prefere use plotly library.
```python
#import plotly lib for the further visua;lisation
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
```
Save grouped transmission type in the separate dataframes.

```python
#Extract main transmission types in separate dataframes 
ua_adaptive_transmission=ua_transmission[ua_transmission['transmission']=='adaptive']

ua_automatic_transmission=ua_transmission[ua_transmission['transmission']=='automatic']

ua_manual_transmission=ua_transmission[ua_transmission['transmission']=='manual']
```
Now it is time to draw:
```python
adaptive_trace = go.Bar(
    x = ua_adaptive_transmission.year,
    y = ua_adaptive_transmission['count'],
    name = "adaptive transmission"
)
automatic_trace = go.Bar(
    x = ua_automatic_transmission.year,
    y = ua_automatic_transmission['count'],
    name = "automatic transmission"
)
manual_trace = go.Bar(
    x = ua_manual_transmission.year,
    y = ua_manual_transmission['count'],
    name = "manual transmission"
)
layout = go.Layout(
    title='Amount of cars sold during a year by transmission types',\
    xaxis= {'title':"years"},yaxis = {'title':"total amount of cars"},)


fig = go.Figure(data = [adaptive_trace,automatic_trace,manual_trace], layout = layout)
iplot(fig)
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 12.06.20.png", width=1000, height=700, op="fit_width") }}

From this graph we can see:
- that manual transmission cars were super popular in Ukraine till 2012. 
- Last years we saw the trend that automatic transmission became more and more popular. We can affirm that automatic transmission will become an absolut  lead on Ukrainian market next years.
- Total amount of sales of both transmission types in UA growing down. Maybe it is an economical reason.
- The top sales years for manual transmission are 2007-2008 and for automatic are 2015-2016.

Further I would like to compare two of the most popular brands (Toyota nad Volkswagen) in time.

Group data by brand and year and compare the max price for each.

```python
#Total Toyota/VolksWagen sales by year
max_price = ua_cars.groupby(['brand', 'year'])['price'].max() \
  .reset_index(name='price').sort_values(['year'], ascending=False) 

toyota_price=max_price[max_price['brand']=='toyota']
vw_price=max_price[max_price['brand']=='volkswagen']
```

Use scatterplot to draw this info:
```
toyota_trace = go.Scatter(
    x = toyota_price.year,
    y = toyota_price['price'],
    name = 'Toyota'
)
vw_trace = go.Scatter(
    x = vw_price.year,
    y = vw_price['price'],
    name= 'VolksWagen'
)
layout = go.Layout(
    title='Total sales by year',\
    xaxis= {'title':"years"},yaxis = {'title':"total sales,K$"}
)

fig = go.Figure(data = [toyota_trace,vw_trace], layout = layout)
iplot(fig)
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 13.50.31.png", width=800, height=700, op="fit_width") }}

There we can see that both top brands have a good growth of sales. 
Volkswagen has more stable sales but Toyota has more tricky trend: some last years are more lucky for Toyota and the growth is extremely sharp (in  2015 and 2019). It is a point for additional research.

--------------------------------------------------------------------------------------------------------
# US Cars
Now it is time to analyse another dataset  - the dataset of US cars. It will be interesting to compare these two markets in the end.

```python
us_cars = pd.read_csv("cars/usa_vehicles_price.csv")
```

I’ll skip some code because it is similar as for UA cars  and just describe some results.
US cars dataset have next colums:
```python
Index(['Unnamed: 0', 'price', 'brand', 'model', 'year', 'title_status',
       'mileage', 'color', 'vin', 'lot', 'state', 'country', 'condition'],
      dtype='object')
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 14.08.34.png", width=900, height=700, op="fit_width") }}
This dataset is wider by features and it is great, so maybe we can extract more interesting info here.

```us_cars["country"].unique() ``` says to us that this dataset collects info about Canada cars sold in US. I’d like to delete them because now I’m not sure how to interpret them in the future. Additionally let’s delete vin numbers - official car id number: they can’t give us now some specific info.
```python

us_cars = us_cars.drop(['Unnamed: 0','vin',"condition","lot","title_status"],axis = 1)
us_cars= us_cars[us_cars['country']==' usa']
us_cars = us_cars.drop(["country"],axis = 1)
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 14.13.44.png", width=500, height=700, op="fit_width") }}

Of course the first thing I noticed when I downloaded the dataset, what units are used for US cars)))) Europeans “likes” american measures :) 

So let’s transform US miles into kilometers:
```python
us_cars['mileage']=(us_cars['mileage']*1.60934/1000).round(0).astype(int)
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 14.22.33.png", width=600, height=700, op="fit_width") }}



Let’s find top 10 popular brands in US:
```python
us_cars["brand"].value_counts().head(10).plot(kind='bar')
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 14.24.34.png", width=600, height=700, op="fit_width") }}
So in Us Ford is a leader of sales. Dodge, Nissan and Chevrolet are popular too, but Ford is absolut leader.

Also we can group data by brand and year and compare their maximal prices 
```python
max_price_us = us_cars.groupby(['brand', 'year'])['price'].max() \
  .reset_index(name='price').sort_values(['year'], ascending=False) 

ford_price=max_price[max_price['brand']=='ford']
dodge_price=max_price[max_price['brand']=='dodge']
nissan_price=max_price[max_price['brand']=='nissan']
chevrolet_price=max_price[max_price['brand']=='chevrolet']
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 15.14.13.png", width=800, height=700, op="fit_width") }}

There we see:
that there is no data about car prices for brands except For from the 40s till mid. the 80s. I can explain that three other brands appeared in the US market just in the 80s.
In the 70s Ford sharply grew down and started slightly growing up with other new brands only in the middle of the 2000s. It looks like some crises on the market. 
Dodge shows stable growth in the last 10 years. Three other brands have a few peaks. Especially significant peaks for Nisan and Chevrolet in 2011-1012. I can suggest that some new popular models appeared during this time.
Fords became a leader again on the market at 2018-2019.

Next I would like to groupe data by US states and popular brands in and review the trend:
```python
car_by_state = us_cars.groupby(['state','brand'])['price'].max().reset_index(name='price')
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 15.38.12.png", width=800, height=700, op="fit_width") }}





Ok, now it is time for the most “cutie” side of the dataset. This is a car color feature. I lived and visited a lot of countries and I see that each country has its own “character” also in car preferences. So I have some hipotesis about it and I want to get the approval. 

```python
car_by_colors = us_cars.groupby('color')['model'].count().reset_index().sort_values('model',ascending = False).head(5)
fig= pl.bar(car_by_colors, x='color',y='model')
fig.show()
```

{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 15.11.46.png", width=800, height=700, op="fit_width") }}

Ok, for US the color number one is white. Great! Less popular are blach, gray/silver and red. (A brief digression: in Germany the most popular car colors are black and dark gray. You don’t need to be an analytic to understand that. People explain that prices of insurance and subsequently such cars will be easy to sell. Other reasons are: practicality and road safety).


My next goal to understand is some relationship between color preference and state. I have an advice that in southern states white is more preferable than in northern and vice versa.
```python
car_by_colors_state = us_cars.groupby(['color','state'])['state'].count().reset_index(name='count').sort_values('count',ascending = False)
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 21.58.45.png", width=400, height=700, op="fit_width") }}
```python
white_color_in_state=car_by_colors_state[car_by_colors_state['color']=='white'].head(5)
black_color_in_state=car_by_colors_state[car_by_colors_state['color']=='black'].head(5)
gray_color_in_state=car_by_colors_state[car_by_colors_state['color']=='gray'].head(5)
silver_color_in_state=car_by_colors_state[car_by_colors_state['color']=='silver'].head(5)
red_color_in_state=car_by_colors_state[car_by_colors_state['color']=='red'].head(5)

color = {'White':'white',
          'Black':'black',
         'Gray':'gray',
         'Silver':'silver',
         'Red':'red'}

white_trace = go.Bar(
    x = white_color_in_state.state,
    y = white_color_in_state['count'],
    name = 'White',
    marker_color=color['White']
)
black_trace = go.Bar(
    x = black_color_in_state.state,
    y = black_color_in_state['count'],
    name = 'Black',
    marker_color=color['Black']
)
gray_trace = go.Bar(
    x = gray_color_in_state.state,
    y = gray_color_in_state['count'],
    name = 'Gray',
    marker_color=color['Gray']
)
silver_trace = go.Bar(
    x = silver_color_in_state.state,
    y = silver_color_in_state['count'],
    name = 'Silver',
    marker_color=color['Silver']
)
red_trace = go.Bar(
    x = red_color_in_state.state,
    y = red_color_in_state['count'],
    name = 'Red',
    marker_color=color['Red']
)
layout = go.Layout(
    title='',\
    xaxis= {'title':" "},yaxis = {'title':" "}
)

fig = go.Figure(data = [white_trace,black_trace,gray_trace,silver_trace,red_trace], layout = layout)
iplot(fig)
```
{{ resize_image(path="cars_analysis/images/Screen Shot 2020-12-16 at 22.00.10.png", width=800, height=700, op="fit_width") }}
As I previously thought white cars are selling better in southern states such as Texas, California, Michigan. Black cars are popular in northern Pennsylvania. What is strange, that in southern Florida white cars are inferior to black. At the same time other colors are presented enough.

In conclusion I would say that US car drivers choose domestic cars (Fords and Dodges) with practical colors depending on the state.

#### Summary:
###### - UA and US markets are very different. People in Ukraine preferred small Europaens sedans such as Volkswagen or Korean Toyota.  Simultaneously in the US it is domestic cars.
###### - Ukrainian market transforming and automatic transmission gained popularity.
###### - My observations of car colors were approved. White color is the most popular in this country.Color preferences are  changing from state to state. 
