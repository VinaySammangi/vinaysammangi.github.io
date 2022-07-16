---
title: Content-based Movie Recommendation System
excerpt: Using TF-IDF to recommend netflix movies
collection: portfolio
permalink: /portfolio/Recommendation Engine/
---

# 1. Netflix Movies: Recommendation Engine #
## 1.1 Setting the Context

<div style="text-align: justify"> 
Netflix is one of the most popular media and video streaming platforms. They have over 8000 movies or tv shows available on their platform. As of Oct-2021, they have over 215M subscribers globally. Once you start logging into Netflix regularly, you will realize that Netflix is usually spot on about what you'd like to see. This is done with the help of something known as a recommender system. A recommender system is capable of predicting a person's future preference given a fixed amount of limited data. One primary reason Netflix uses a recommender system is that a lot of content is present on its platform, which can be entirely irrelevant to people based on their language or genres of interest.
</div>

In this blog, we will build a straightforward content-based recommendation system on Netflix data. But before getting to that point, we need to preprocess the data and understand the variables. The workflow is 
as follows:
- A 3-step missing value imputation process
- Building a Content-based Recommender System
**Maximum runtime of the notebook - 5-6 mins**

## 1.2 Setup


```python
# Data handling
import numpy as np
import pandas as pd
from collections import Counter
import time, math

# Parallel Tasking
from joblib import Parallel, delayed

# Web Crawling
from bs4 import BeautifulSoup
import requests

import matplotlib.pyplot as plt

# For the recommender system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer 
```

## 1.3 Dataset : Kaggle

The tabular dataset that we will use in this notebook consists of listings of all the movies and tv shows available on Netflix, along with details such as - cast, directors, ratings, release year, duration, etc. The data is downloaded from [Kaggle](https://www.kaggle.com/shivamb/netflix-shows/).


```python
netflix_data = pd.read_csv("Data/netflix_titles.csv")
netflix_data.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Kirsten Johnson</td>
      <td>NaN</td>
      <td>United States</td>
      <td>September 25, 2021</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>NaN</td>
      <td>Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban...</td>
      <td>South Africa</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, TV Dramas, TV Mysteries</td>
      <td>After crossing paths at a party, a Cape Town t...</td>
    </tr>
  </tbody>
</table>
</div>



Having a glance at the first two rows of the dataset tells us there are some missing values in the data. But we will deal with it later. First, we will understand what these variables represent.


|Variable      |Description                           |
|:------------:|:------------------------------------:|
|show_id       |Unique ID for every Movie / Tv Show   |
|type          |Identifier - A Movie or TV Show       |
|title         |Title of the Movie / Tv Show          |
|director      |Director of the Movie                 |
|cast          |Actors involved in the movie / show   |
|country       |Country where the movie / show was produced|
|date_added    |Date it was added on Netflix          |
|release_year  |Actual Release year of the move / show|
|rating        |TV Rating of the movie / show         |
|duration      |Total Duration - in minutes or number of seasons|
|listed_in     |Genere                                |
|description   |The summary description               |

## 1.4 Data Exploration

### 1.4.1 Filter only movies data and remove duplicate rows


```python
netflix_data.describe(include='all').head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8807</td>
      <td>8807</td>
      <td>8807</td>
      <td>6173</td>
      <td>7982</td>
      <td>7976</td>
      <td>8797</td>
      <td>8807.0</td>
      <td>8803</td>
      <td>8804</td>
      <td>8807</td>
      <td>8807</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>8807</td>
      <td>2</td>
      <td>8807</td>
      <td>4528</td>
      <td>7692</td>
      <td>748</td>
      <td>1767</td>
      <td>NaN</td>
      <td>17</td>
      <td>220</td>
      <td>514</td>
      <td>8775</td>
    </tr>
    <tr>
      <th>top</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Rajiv Chilaka</td>
      <td>David Attenborough</td>
      <td>United States</td>
      <td>January 1, 2020</td>
      <td>NaN</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Dramas, International Movies</td>
      <td>Paranormal activity at a lush, abandoned prope...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>6131</td>
      <td>1</td>
      <td>19</td>
      <td>19</td>
      <td>2818</td>
      <td>109</td>
      <td>NaN</td>
      <td>3207</td>
      <td>1793</td>
      <td>362</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



There are 8807 rows in the dataset. We will focus only on movies data (not TV shows) and build a recommendation system on it.


```python
netflix_data.groupby('type').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Movie</th>
      <td>6131</td>
      <td>6131</td>
      <td>5943</td>
      <td>5656</td>
      <td>5691</td>
      <td>6131</td>
      <td>6131</td>
      <td>6129</td>
      <td>6128</td>
      <td>6131</td>
      <td>6131</td>
    </tr>
    <tr>
      <th>TV Show</th>
      <td>2676</td>
      <td>2676</td>
      <td>230</td>
      <td>2326</td>
      <td>2285</td>
      <td>2666</td>
      <td>2676</td>
      <td>2674</td>
      <td>2676</td>
      <td>2676</td>
      <td>2676</td>
    </tr>
  </tbody>
</table>
</div>



We can observe 6131 movies and 2676 tv shows on Netflix. So, we will only filter the movies data from the original dataset.


```python
movies_data = netflix_data.loc[netflix_data["type"]=="Movie",].copy()
```


```python
movies_data["title"] = movies_data['title'].str.strip().str.lower()
temp = movies_data['title'].value_counts()
movies_data.loc[movies_data["title"].isin(list(temp.index[temp>1])),]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>s160</td>
      <td>Movie</td>
      <td>love in a puff</td>
      <td>Pang Ho-cheung</td>
      <td>Miriam Chin Wah Yeung, Shawn Yue, Singh Hartih...</td>
      <td>Hong Kong</td>
      <td>September 1, 2021</td>
      <td>2010</td>
      <td>TV-MA</td>
      <td>103 min</td>
      <td>Comedies, Dramas, International Movies</td>
      <td>When the Hong Kong government enacts a ban on ...</td>
    </tr>
    <tr>
      <th>303</th>
      <td>s304</td>
      <td>Movie</td>
      <td>esperando la carroza</td>
      <td>Alejandro Doria</td>
      <td>Luis Brandoni, China Zorrilla, Antonio Gasalla...</td>
      <td>Argentina</td>
      <td>August 5, 2021</td>
      <td>1985</td>
      <td>TV-MA</td>
      <td>95 min</td>
      <td>Comedies, Cult Movies, International Movies</td>
      <td>Cora has three sons and a daughter and she´s a...</td>
    </tr>
    <tr>
      <th>3371</th>
      <td>s3372</td>
      <td>Movie</td>
      <td>consequences</td>
      <td>Ozan Açıktan</td>
      <td>Nehir Erdoğan, Tardu Flordun, İlker Kaleli, Se...</td>
      <td>Turkey</td>
      <td>October 25, 2019</td>
      <td>2014</td>
      <td>TV-MA</td>
      <td>106 min</td>
      <td>Dramas, International Movies, Thrillers</td>
      <td>Secrets bubble to the surface after a sensual ...</td>
    </tr>
    <tr>
      <th>6529</th>
      <td>s6530</td>
      <td>Movie</td>
      <td>consequences</td>
      <td>Ozan Açıktan</td>
      <td>Nehir Erdoğan, Tardu Flordun, İlker Kaleli, Se...</td>
      <td>Turkey</td>
      <td>October 25, 2019</td>
      <td>2014</td>
      <td>TV-MA</td>
      <td>106 min</td>
      <td>Dramas, International Movies, Thrillers</td>
      <td>Secrets bubble to the surface after a sensual ...</td>
    </tr>
    <tr>
      <th>6705</th>
      <td>s6706</td>
      <td>Movie</td>
      <td>esperando la carroza</td>
      <td>Alejandro Doria</td>
      <td>Luis Brandoni, China Zorrilla, Antonio Gasalla...</td>
      <td>Argentina</td>
      <td>July 15, 2018</td>
      <td>1985</td>
      <td>NR</td>
      <td>95 min</td>
      <td>Comedies, Cult Movies, International Movies</td>
      <td>Cora has three sons and a daughter and she´s a...</td>
    </tr>
    <tr>
      <th>7345</th>
      <td>s7346</td>
      <td>Movie</td>
      <td>love in a puff</td>
      <td>Pang Ho-cheung</td>
      <td>Miriam Chin Wah Yeung, Shawn Yue, Singh Hartih...</td>
      <td>Hong Kong</td>
      <td>August 1, 2018</td>
      <td>2010</td>
      <td>TV-MA</td>
      <td>103 min</td>
      <td>Comedies, Dramas, International Movies</td>
      <td>When the Hong Kong government enacts a ban on ...</td>
    </tr>
  </tbody>
</table>
</div>



It looks like these are surely duplicate rows. So, we can remove either of the rows for each movie.


```python
movies_data = movies_data.drop([6529,6705,7345])
```

### 1.4.2 Missing Value Handling/ Imputation
Instead of directly removing rows with missing values, we try to impute as much data as possible with high accuracy.
This process involves three steps:
- Stage 1: Remove rows for columns with very, very few missing values
- Stage 2: Web crawling based imputation to achieve high accuracy
- Stage 3: Replace the remaining NaN values with an empty string to preserve information in other columns

#### 1.4.2.1 Handling Missing Values - Stage 1 (Drop Rows)


```python
print("Rows with missing values in the data: "+
      str(round(100*sum(movies_data.isnull().any(axis=1))/movies_data.shape[0],2))+"%")
movies_data.isna().sum()
```

    Rows with missing values in the data: 15.44%





    show_id           0
    type              0
    title             0
    director        188
    cast            475
    country         440
    date_added        0
    release_year      0
    rating            2
    duration          3
    listed_in         0
    description       0
    dtype: int64



We can see several missing values in the `director`, `cast`, `country` columns and a very few missing values in the `rating` and `duration` columns. Let's remove the rows with missing values in the `rating` and `duration` columns.


```python
movies_data.dropna(subset=["rating","duration"], how='any', inplace=True)
print("Rows with missing values in the data: "+str(round(100*sum(movies_data.isnull().any(axis=1))/movies_data.shape[0],2))+"%")
movies_data.isna().sum()
```

    Rows with missing values in the data: 15.37%





    show_id           0
    type              0
    title             0
    director        187
    cast            475
    country         439
    date_added        0
    release_year      0
    rating            0
    duration          0
    listed_in         0
    description       0
    dtype: int64




```python
nan_rows_df = movies_data[movies_data.isnull().any(axis=1)]
nan_rows_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>dick johnson is dead</td>
      <td>Kirsten Johnson</td>
      <td>NaN</td>
      <td>United States</td>
      <td>September 25, 2021</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>s7</td>
      <td>Movie</td>
      <td>my little pony: a new generation</td>
      <td>Robert Cullen, José Luis Ucha</td>
      <td>Vanessa Hudgens, Kimiko Glenn, James Marsden, ...</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>PG</td>
      <td>91 min</td>
      <td>Children &amp; Family Movies</td>
      <td>Equestria's divided. But a bright-eyed hero be...</td>
    </tr>
  </tbody>
</table>
</div>



There are still more than 15% rows with missing values. So instead of removing those rows, we will try to impute the data with high accuracy.

#### 1.4.2.2 Handling Missing Values - Stage 2 (Web Crawling)

The concept behind this is simple. For example, let's look at the first row in the above table where the movie title is "Dick Johnson Is Dead" and the cast has NaN value. First of all, why is this value missing? There could be two potential reasons.
- Netflix might not log this information on their platform. Hence the value was missing
- This data file is being maintained on Kaggle and constantly updated by only a single person. So, there might be some manual errors involved while copying the data into a CSV file.

However, we can't attribute each row with missing value to a specific reason. In either case, we will look up the movie on [IMDb](https://www.imdb.com/) and get the director, cast, and country of origin data.

You can also look at this sample [URL](https://www.imdb.com/title/tt11394180/?ref_=fn_al_tt_1) and see how we can extract the director, cast, and country of origin variables from it.


```python
# Functions to extract the director, cast, country of origin data from a page source

def get_director(soup):
    """
    Extract the director information from the HTML source data
    Args:
        soup: object (page source) obtained from scraping the website using BeautifulSoup() function
    Returns:
        director: returns a string containing directors of a movie separated by a comma
    """ 
    try:
        director = ""
        temp = soup.find("section",{"data-testid":"title-cast"}).find_all("li",{"class","ipc-metadata-list__item"})
        if len(temp)==4: #if the section on the page in found
            director_soups = temp[0].find_all("a")
            for director_soup in director_soups:
                name = director_soup.get_text().strip()
                director = director + name + ", "
            director = director[:-2]
            return director
        else:
            return director
    except:
        return director

def get_cast(soup):
    """
    Extract the cast information from the HTML source data
    Args:
        soup: object (page source) obtained from scraping the website using BeautifulSoup() function
    Returns:
        cast: returns a string containing all the cast members of a movie separated by a comma
    """ 
    try:
        cast = ""
        cast_soups = soup.find("section",{"data-testid":"title-cast"}).find_all("a",{"data-testid":"title-cast-item__actor"})
        for cast_soup in cast_soups:
            name = cast_soup.get_text().strip()
            cast = cast + name + ", "
        cast = cast[:-2]
        return cast
    except:
        return cast
    
def get_country(soup):
    """
    Extract the country information from the HTML source data
    Args:
        soup: object (page source) obtained from scraping the website using BeautifulSoup() function
    Returns:
        country: returns a string containing all the countries of origin of a movie separated by a comma
    """ 
    try:
        country = ""
        countries_soups = soup.find("div",{"data-testid":"title-details-section"}).find("li",{"data-testid":"title-details-origin"}).find_all("a")
        for countries_soup in countries_soups:
            name = countries_soup.get_text().strip()
            country = country + name + ", "
        country = country[:-2]
        return country
    except:
        return country

```


```python
%%time

def imdb_requests(row):
    """
    Main Function that extracts the director, cast, and countries of origin information for each row with NaN value
    Args:
        row: dataframe row containing atleast one NaN value
    Returns:
        main_dict: returns a dictionary containing title, show_id, director, cast, country as keys and their
        corresponding values as dictionary values
    """ 
    main_dict = {}
    main_dict["title"] = row["title"]
    main_dict["show_id"] = row["show_id"]
    
    try:
        source = requests.get("https://www.imdb.com/find?ref_=nv_sr_fn&q="+str(main_dict["title"]))
        source.raise_for_status()
        soup = BeautifulSoup(source.text,'html.parser')

        #take the first URL on the results page and extract information from it
        title = soup.find("td",{"class":"result_text"}).find('a').get("href")
        
        new_url = "https://www.imdb.com"+title
        source = requests.get(new_url)
        source.raise_for_status()

        soup = BeautifulSoup(source.text,'html.parser')
        main_dict["director"] = get_director(soup)
        main_dict["cast"] = get_cast(soup)
        main_dict["country"] = get_country(soup)        
    except:
        pass
    
    return main_dict

"""
The below four rows calls several URL requests using a parallel function, convert an array of dictionaries 
to a data frame, and replace empty string with NaN values (to impute them later).
This code takes close to 5 minutes to run. I ran this already and generated the intermediate file, 
which I will use in the subsequent sections.
"""
# nan_rows_search_results = Parallel(n_jobs=-1)(delayed(imdb_requests)(row) for index, row in nan_rows_df.iterrows())
# nan_rows_search_results_df = pd.DataFrame(nan_rows_search_results)
# nan_rows_search_results_df = nan_rows_search_results_df.replace('',np.nan,regex=True)
# nan_rows_search_results_df.to_csv("IMDB_intermediate_data.csv",index=False)
nan_rows_search_results_df = pd.read_csv("IMDB_intermediate_data.csv")
```

    CPU times: user 4.45 ms, sys: 1.67 ms, total: 6.12 ms
    Wall time: 5.25 ms


**Single Core vs Multi Core Computations:**

<img src="/assets/images/netflix_recommender/parallel.png" width="1600" height="800">

We observe that using a parallel function helps us reduce the run time to 15%.


```python
movies_data.cast = np.where(movies_data.cast.isnull(),movies_data.show_id.map(nan_rows_search_results_df.set_index('show_id').cast),movies_data.cast)
movies_data.country = np.where(movies_data.country.isnull(),movies_data.show_id.map(nan_rows_search_results_df.set_index('show_id').country),movies_data.country)
movies_data.director = np.where(movies_data.director.isnull(),movies_data.show_id.map(nan_rows_search_results_df.set_index('show_id').director),movies_data.director)

```


```python
print("Rows with missing values in the data: "+str(round(100*sum(movies_data.isnull().any(axis=1))/movies_data.shape[0],2))+"%")
movies_data.isna().sum()
```

    Rows with missing values in the data: 5.42%





    show_id           0
    type              0
    title             0
    director        109
    cast            104
    country         183
    date_added        0
    release_year      0
    rating            0
    duration          0
    listed_in         0
    description       0
    dtype: int64



We now only have about 5.4% of the missing rows in the data. Unfortunately, we could not find the rest of them from the IMDB data. So, we replace them with an empty string.

#### 1.4.2.3 Handling Missing Values - Stage 3 (Replace with Empty String)


```python
movies_data = movies_data.replace(np.nan,'',regex=True)
movies_data.reset_index(drop=True,inplace=True)
print("Rows with missing values in the data: "+str(round(100*sum(movies_data.isnull().any(axis=1))/movies_data.shape[0],2))+"%")
movies_data.isna().sum()
```

    Rows with missing values in the data: 0.0%





    show_id         0
    type            0
    title           0
    director        0
    cast            0
    country         0
    date_added      0
    release_year    0
    rating          0
    duration        0
    listed_in       0
    description     0
    dtype: int64



### 1.4.3 Parsing the `date_added`, `duration` columns

We will extract the year, month, and day of the week data from the `date_added` column and analyze them separately to generate more insights later. Also, we parse the `duration` column into a numeric column.


```python
# Year added column
movies_data['year_added'] = movies_data['date_added'].apply(lambda x: x.split(" ")[-1])
movies_data['year_added'] = movies_data["year_added"].astype("int")
# Month added column
movies_data['month_added'] = movies_data['date_added'].apply(lambda x: x.split(" ")[0])
movies_data['date_added'] = pd.to_datetime(movies_data['date_added'])
movies_data['day_of_week'] = movies_data['date_added'].dt.day_name()
movies_data[['month_added','year_added','day_of_week']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month_added</th>
      <th>year_added</th>
      <th>day_of_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>September</td>
      <td>2021</td>
      <td>Saturday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>September</td>
      <td>2021</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>September</td>
      <td>2021</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>September</td>
      <td>2021</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>September</td>
      <td>2021</td>
      <td>Thursday</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_data['duration']=movies_data['duration'].str.replace(' min','')
movies_data['duration']=movies_data['duration'].astype(str).astype(int)
```

## 1.5 Content-based recommendation engine on multiple metrics

Now that we have a fair understanding of the variables, we will build the recommendation engine using a few of them. There are [two](https://en.wikipedia.org/wiki/Recommender_system) main types of recommendation engines: content-based filtering and collaborative filtering. We will try to build the former one in this notebook.

Content-based filtering works on the principle that you will also like another item if you like a particular item. For example, to provide movie recommendations, algorithms use several movie attributes like `title`, `genre`, `director`, `cast` to compare movies using cosine or euclidean distances. One of the major downsides of this approach is that this system limits recommending movies similar to what the person has already watched. However, we will not address this in this notebook.


```python
features = ['title','director','cast','listed_in']

def clean_data(df,features):
    df_subset = df[features].copy()
    df_subset['main_column'] = ""
    for feature in features:
        if feature!="description":
            df_subset[feature] = df_subset[feature].apply(lambda x: str.lower(x.replace(" ", "")))
        df_subset["main_column"] = df_subset["main_column"] + ' ' + df_subset[feature]
    return df_subset
```

We need to remove the spaces from the data before combining the features to a new column. This is required because, for example, there are 84 directors with Michael as part of their name, but none of them have a common full name. So it doesn't make sense to recommend a director's movies only because they have a part of their name common to another director. The same logic applies to the other columns.


```python
movies_data_subset = clean_data(movies_data,features)
movies_data_subset.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>listed_in</th>
      <th>main_column</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dickjohnsonisdead</td>
      <td>kirstenjohnson</td>
      <td>michaelhilow,anahoffman,dickjohnson,kirstenjoh...</td>
      <td>documentaries</td>
      <td>dickjohnsonisdead kirstenjohnson michaelhilow...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mylittlepony:anewgeneration</td>
      <td>robertcullen,joséluisucha</td>
      <td>vanessahudgens,kimikoglenn,jamesmarsden,sofiac...</td>
      <td>children&amp;familymovies</td>
      <td>mylittlepony:anewgeneration robertcullen,josé...</td>
    </tr>
  </tbody>
</table>
</div>



We use the TF-IDF (term frequency–inverse document frequency) matrix to process the new combined column `main_column` that was created in the previous step. You can also read about TF-IDF [here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). We then use cosine-similarity to create a score between each pair of movies.


```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_data_subset['main_column'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

### 1.5.1 Which movies are the most similar to each other?


```python
movie_titles_df = pd.DataFrame(movies_data['title']).reset_index()
movie_titles_df.columns = ["row_id","Title"]

cosine_sim_df = pd.DataFrame(cosine_sim).reset_index()
cosine_sim_df_melted = pd.melt(cosine_sim_df, id_vars=['index'], value_vars=list(cosine_sim_df.columns[1:]))
cosine_sim_df_melted.columns = ["row_id1","row_id2","similarity"]
cosine_sim_df_melted = cosine_sim_df_melted.sort_values("similarity",ascending=False)
cosine_sim_df_melted = cosine_sim_df_melted.loc[cosine_sim_df_melted["row_id1"]<cosine_sim_df_melted["row_id2"],].reset_index(drop=True)
```

**Filter movies with very high similarity**


```python
thres = 0.9
filtered_df = cosine_sim_df_melted.loc[cosine_sim_df_melted["similarity"]>thres,].copy()
filtered_df = filtered_df.merge(movie_titles_df,left_on="row_id1",right_on="row_id")
filtered_df = filtered_df.merge(movie_titles_df,left_on="row_id2",right_on="row_id")
filtered_df = filtered_df[["Title_x","Title_y","similarity"]].copy()
filtered_df.columns = ["Movie1","Movie2","Similarity"]
filtered_df["Similarity"] = round(filtered_df["Similarity"],2)
filtered_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie1</th>
      <th>Movie2</th>
      <th>Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>oh! baby (tamil)</td>
      <td>oh! baby</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>oh! baby (malayalam)</td>
      <td>oh! baby</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>oh! baby (malayalam)</td>
      <td>oh! baby (tamil)</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>solo: a star wars story</td>
      <td>solo: a star wars story (spanish version)</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rogue warfare: death of a nation</td>
      <td>rogue warfare</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rogue warfare: the hunt</td>
      <td>rogue warfare</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rogue warfare: death of a nation</td>
      <td>rogue warfare: the hunt</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>7</th>
      <td>boomika</td>
      <td>boomika (hindi)</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>8</th>
      <td>boomika</td>
      <td>boomika (telugu)</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>9</th>
      <td>boomika</td>
      <td>boomika (malayalam)</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>10</th>
      <td>petta (telugu version)</td>
      <td>petta</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bo burnham: what.</td>
      <td>bo burnham: make happy</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>12</th>
      <td>godzilla the planet eater</td>
      <td>godzilla city on the edge of battle</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>13</th>
      <td>osuofia in london</td>
      <td>osuofia in london ii</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tughlaq durbar</td>
      <td>tughlaq durbar (telugu)</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>15</th>
      <td>naruto shippuden the movie: blood prison</td>
      <td>naruto shippuden : blood prison</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sarvam thaala mayam (telugu version)</td>
      <td>sarvam thaala mayam (tamil version)</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>17</th>
      <td>chris d'elia: man on fire</td>
      <td>chris d'elia: incorrigible</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>18</th>
      <td>octonauts &amp; the ring of fire</td>
      <td>octonauts &amp; the great barrier reef</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>19</th>
      <td>the twilight saga: breaking dawn: part 1</td>
      <td>the twilight saga: breaking dawn: part 2</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>20</th>
      <td>baahubali 2: the conclusion (hindi version)</td>
      <td>baahubali 2: the conclusion (tamil version)</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>21</th>
      <td>baahubali 2: the conclusion (malayalam version)</td>
      <td>baahubali 2: the conclusion (tamil version)</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>22</th>
      <td>baahubali 2: the conclusion (hindi version)</td>
      <td>baahubali 2: the conclusion (malayalam version)</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>23</th>
      <td>baahubali: the beginning (hindi version)</td>
      <td>baahubali: the beginning (tamil version)</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>24</th>
      <td>baahubali: the beginning (malayalam version)</td>
      <td>baahubali: the beginning (tamil version)</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>25</th>
      <td>baahubali: the beginning (hindi version)</td>
      <td>baahubali: the beginning (malayalam version)</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>26</th>
      <td>the magic school bus rides again the frizz con...</td>
      <td>the magic school bus rides again kids in space</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>27</th>
      <td>game over (hindi version)</td>
      <td>game over (tamil version)</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>28</th>
      <td>game over (hindi version)</td>
      <td>game over (telugu version)</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>29</th>
      <td>game over (tamil version)</td>
      <td>game over (telugu version)</td>
      <td>0.90</td>
    </tr>
  </tbody>
</table>
</div>



We observe that the same movie with different versions in multiple languages has the highest score based on the results. If we do not want them as part of our recommendations, we can remove the duplicate entries in the preprocessing step. For now, we will keep them as part of our model.


```python
movies_data_subset=movies_data_subset.reset_index()
indices = pd.Series(movies_data_subset.index, index=movies_data_subset['title'])
```

### 1.5.2 Let's get some recommendations for a movie


```python
def get_recommendations_new(title, cosine_sim, n):
    """
    Find the similar movies to a given movie
    Args:
        title: movie title to which we find recommendations
        cosine_sim: cosine similarity matrix for finding similar movies
        n: number of movies to recommend
    Returns:
        results_df: returns a dataframe containing the list of recommended movies with rowids
        and their similarity score
    """ 
    title = title.replace(' ','').lower()
    idx = indices[title]

    #pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    #sort the movies based on cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top n most similar movies
    sim_scores = sim_scores[1:(n+1)]
    # Get their movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    results_df = pd.DataFrame(movies_data['title'].iloc[movie_indices])
    results_df["score"] = np.round(np.array(sim_scores)[:,1],2)
    results_df = results_df.reset_index(drop=False)
    results_df.columns = ["RowID","Recommended Movie","Similarity Score"]
    return results_df
```


```python
movie_title = "pk"
recommendations_df = get_recommendations_new(movie_title,cosine_sim,5)
```


```python
temp_df = movies_data.loc[movies_data.title.isin([movie_title]+list(recommendations_df["Recommended Movie"]))]
temp_df = temp_df[features].reset_index(drop=True)
temp_df = temp_df.merge(recommendations_df,left_on="title",right_on = "Recommended Movie",how="outer")
temp_df = temp_df.sort_values("Similarity Score",ascending=False)
temp_df = temp_df[["title","director","cast","listed_in","Similarity Score"]]
temp_df["new"] = range(1,len(temp_df)+1)
temp_df.loc[temp_df.title==movie_title,'new'] = 0
temp_df = temp_df.sort_values("new").drop('new', axis=1)
temp_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>listed_in</th>
      <th>Similarity Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>pk</td>
      <td>Rajkumar Hirani</td>
      <td>Aamir Khan, Anuskha Sharma, Sanjay Dutt, Saura...</td>
      <td>Comedies, Dramas, International Movies</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3 idiots</td>
      <td>Rajkumar Hirani</td>
      <td>Aamir Khan, Kareena Kapoor, Madhavan, Sharman ...</td>
      <td>Comedies, Dramas, International Movies</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sanju</td>
      <td>Rajkumar Hirani</td>
      <td>Ranbir Kapoor, Vicky Kaushal, Paresh Rawal, So...</td>
      <td>Dramas, International Movies</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>drive</td>
      <td>Tarun Mansukhani</td>
      <td>Jacqueline Fernandez, Sushant Singh Rajput, Bo...</td>
      <td>Action &amp; Adventure, International Movies</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>taare zameen par</td>
      <td>Aamir Khan</td>
      <td>Aamir Khan, Darsheel Safary, Tanay Chheda, Tis...</td>
      <td>Dramas, International Movies</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>0</th>
      <td>madness in the desert</td>
      <td>Satyajit Bhatkal</td>
      <td>Aamir Khan, Ashutosh Gowariker</td>
      <td>Documentaries, International Movies</td>
      <td>0.12</td>
    </tr>
  </tbody>
</table>
</div>



<font color='blue' size = 4>
<center>The above recommendations look pretty good for a starting point.</center>
</font>

## 1.6 Summary and Scope for Improvement

### 1.6.1 Summary
We started with data preprocessing steps that involved removing duplicate entries, missing value imputation stages, and feature extractions. Using web crawling, we used a unique approach to imputing missing data with high accuracy. Finally, we converted the preprocessed text into a TF-IDF matrix and calculated the scores using the cosine similarity function to create the final recommendation system.

### 1.6.2 Scope for Improvement
The below pointers mention a few ways to improve the workflow of this notebook:
- We did not analyze the `description` column, which contains a movie summary, but it can also be added to the existing system to generate more accurate recommendations.
- Word clouds can also be plotted when analyzing the `description` column.

