#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import csv
import random
import time


# In[2]:


pip install flair


# In[3]:


def scrape_reviews(movie_id):
    base_url = f'https://www.imdb.com/title/{movie_id}/reviews'
    scraped_data = []
    max_pages = 150
    current_page = 1

    while current_page <= max_pages:
        url = f'{base_url}?page={current_page}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            break  # Stop scraping if an error occurs
        
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = soup.find_all('div', class_='lister-item-content')
        
        # If no reviews are found, break the loop
        if not reviews:
            break

        # Scrape data from each review on the page
        for review in reviews:
            title_element = review.find('a', class_='title')
            title = title_element.text.strip() if title_element else 'N/A'

            rating_element = review.find('span', class_='rating-other-user-rating')
            rating = rating_element.text.strip() if rating_element else 'N/A'

            text_element = review.find('div', class_='text show-more__control')
            text = text_element.text.strip() if text_element else 'N/A'

            scraped_data.append({'Title': title, 'Rating': rating, 'Text': text})
        
        # Move to the next page
        current_page += 1

        # Add a random delay between requests
        time.sleep(random.uniform(1, 3))  # Adjust delay as needed

    return scraped_data


# In[4]:


import pandas as pd
movie_id = 'tt15398776'
reviews = scrape_reviews(movie_id)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(reviews)

# Save the DataFrame to a CSV file
df.to_csv(f'{movie_id}_reviews.csv', index=False)

print("CSV file has been created successfully.")


# In[5]:


import torch


# In[6]:


import flair


# In[7]:


import sentence_transformers


# In[8]:


import chardet


# In[9]:


data = pd.read_csv(r"tt15398776_reviews.csv", encoding="latin1")


# In[10]:


data


# In[11]:


data.drop(columns=['Title', 'Rating'], inplace=True)


# In[12]:


data


# In[13]:


from flair.models import TextClassifier


# In[14]:


from flair.data import Sentence


# In[15]:


flair_sentiment = TextClassifier.load('en-sentiment')


# In[16]:


def predict_sentiment(comment):
  if isinstance(comment, str):
    sentence = Sentence(comment)
    flair_sentiment.predict(sentence)
    total_sentiment = sentence.labels[0]
    return total_sentiment.value
  else:
    return None


# In[17]:


data['review sentiment'] = data['Text'].apply(predict_sentiment)
data.head()


# In[18]:


data.to_csv('scrapped data sentiment.csv',index=False)


# In[19]:


data=pd.read_csv("scrapped data sentiment.csv")


# In[20]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re


# In[21]:


# Define function for text preprocessing
def preprocess_text(text):
  # Remove special characters and digits
  text = re.sub(r'[^a-zA-Z\s]', '', text)

  # Tokenization
  tokens = word_tokenize(text.lower())

  # Remove stopwords
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]

  # Stemming
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(word) for word in tokens]

  # Join tokens back into a string
  preprocessed_text = ' '.join(tokens)
  
  return preprocessed_text


# In[22]:


# Apply preprocessing function to 'review' column
data['clean_review'] = data['Text'].apply(preprocess_text)

# Display the preprocessed data
print(data[['Text', 'clean_review']].head())


# In[23]:


import matplotlib.pyplot as plt

# Plot the distribution of sentiment labels in the original dataset
plt.figure(figsize=(8, 6))
data['review sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Sentiment Labels in Original Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[24]:


import pandas as pd
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# In[25]:


genre_keywords = {
    'Action': ['fast-paced', 'fight scenes', 'chase scenes', 'superheroes', 'martial arts', 'stunts', 'cop movies', 'disaster films', 'spy films'],
    'Adventure': ['exotic locale', 'far away', 'swashbuckler films', 'survival films'],
    'Comedy': ['funny', 'amusing', 'humorous', 'real-life story', 'mockumentary', 'dark comedy', 'romantic comedy', 'parody', 'slapstick comedy'],
    'Drama': ['high stakes', 'conflicts', 'plot-driven', 'emotionally-driven characters', 'historical drama', 'romantic drama', 'teen drama', 'medical drama'],
    'Fantasy': ['magical', 'supernatural', 'imaginary universes', 'fantastical elements', 'high fantasy', 'fairy tales', 'magical realism'],
    'Horror': ['fear', 'dread', 'serial killers', 'monsters', 'ghosts', 'gore', 'jump scares', 'macabre', 'gothic horror', 'supernatural'],
    'Musical': ['songs', 'musical numbers', 'big stage-like productions'],
    'Mystery': ['detective', 'amateur sleuth', 'puzzle', 'suspense', 'clues', 'evidence', 'murder mystery'],
    'Romance': ['love stories', 'relationships', 'sacrifice', 'romantic comedy', 'gothic romance', 'romantic action'],
    'Science Fiction': ['alternate realities', 'time travel', 'space travel', 'future', 'technological advances'],
    'Sports': ['team', 'individual player', 'sport', 'emotional arcs'],
    'Thriller': ['mystery', 'tension', 'plot twists', 'crime', 'political thrillers', 'psychological thrillers'],
    'Western': ['cowboy', 'gunslinger', 'outlaw', 'duel', 'shootout', 'American West', 'spaghetti westerns'],
    'Crime': ['crime', 'criminal', 'underworld', 'heist', 'caper', 'investigation', 'detective']
}


# In[26]:


def assign_genre(review):
    for genre, keywords in genre_keywords.items():
        for keyword in keywords:
            if keyword in review:
                return genre
    return 'Other'  # If no genre keyword is found, assign 'Other'


# In[27]:


# Assign genre labels to reviews
df['genre'] = df['Text'].apply(assign_genre)

# Visualize the distribution of genres
genre_counts = df['genre'].value_counts()


# In[28]:


# Plotting the distribution of genres using a pie chart
plt.figure(figsize=(10, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Genres in Movie Reviews')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[29]:


movie_review_text = "This movie was full of action-packed scenes and thrilling chase sequences."
predicted_genre = assign_genre(movie_review_text)
print("Predicted genre:", predicted_genre)


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit TF-IDF vectorizer on preprocessed text data
X = tfidf_vectorizer.fit_transform(data['clean_review'])

# Get the target variable
y = data['review sentiment']

# Print the shape of the TF-IDF matrix
print("Shape of TF-IDF matrix:", X.shape)

# Optionally, you can also print the vocabulary
# print("Vocabulary:", tfidf_vectorizer.get_feature_names())


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Step 5: Model Selection
# Choose a machine learning model (e.g., SVM)
model = SVC(kernel='linear')

# Step 6: Model Training
model.fit(X_train, y_train)

# Step 7: Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)


# In[33]:


# Calculate accuracy and other evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[34]:


import joblib

# Step 8: Save the trained model
joblib.dump(model, 'movie_sentiment_model.pkl')


# In[35]:


loaded_model = joblib.load('movie_sentiment_model.pkl')


# In[36]:


def scrape_reviews(movie_id):
    base_url = f'https://www.imdb.com/title/{movie_id}/reviews'
    scraped_data = []
    max_pages = 150
    current_page = 1

    while current_page <= max_pages:
        url = f'{base_url}?page={current_page}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            break  # Stop scraping if an error occurs
        
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = soup.find_all('div', class_='lister-item-content')
        
        # If no reviews are found, break the loop
        if not reviews:
            break

        # Scrape data from each review on the page
        for review in reviews:
            title_element = review.find('a', class_='title')
            title = title_element.text.strip() if title_element else 'N/A'

            rating_element = review.find('span', class_='rating-other-user-rating')
            rating = rating_element.text.strip() if rating_element else 'N/A'

            text_element = review.find('div', class_='text show-more__control')
            text = text_element.text.strip() if text_element else 'N/A'

            scraped_data.append({'Title': title, 'Rating': rating, 'Text': text})
        
        # Move to the next page
        current_page += 1

        # Add a random delay between requests
        time.sleep(random.uniform(1, 3))  # Adjust delay as needed

    return scraped_data


# In[37]:


import pandas as pd
movie_id = 'tt7286456'
reviews = scrape_reviews(movie_id)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(reviews)

# Save the DataFrame to a CSV file
df.to_csv(f'{movie_id}_reviews.csv', index=False)

print("CSV file has been created successfully.")


# In[38]:


revv=pd.read_csv('tt7286456_reviews.csv')


# In[39]:


revv.drop(columns=['Title','Rating'],inplace = True)


# In[40]:


revv


# In[41]:


import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('movie_sentiment_model.pkl')

# Load new movie reviews from a CSV file
new_reviews_df = revv

# Assuming 'review' column contains the movie reviews
new_reviews = new_reviews_df['Text'].tolist()

# Preprocess the new reviews (assuming you have a preprocess_text function)
preprocessed_reviews = [preprocess_text(review) for review in new_reviews]

# Transform the preprocessed text into TF-IDF vectors
X_new = tfidf_vectorizer.transform(preprocessed_reviews)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_new)

# Add predicted sentiment to the DataFrame
new_reviews_df['predicted_sentiment'] = ['Positive' if prediction == 'POSITIVE' else 'Negative' for prediction in predictions]

print(new_reviews_df[['Text', 'predicted_sentiment']])


# In[42]:


import matplotlib.pyplot as plt

# Plot the distribution of sentiment labels in the original dataset
plt.figure(figsize=(8, 6))
new_reviews_df['predicted_sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Sentiment Labels in Original Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[43]:


revv['genre'] = revv['Text'].apply(assign_genre)

# Visualize the distribution of genres
genre_counts = revv['genre'].value_counts()

# Plotting the distribution of genres using a pie chart
plt.figure(figsize=(10, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Genres in Movie Reviews')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'revv' is your DataFrame and 'Text' is the column with text data
# 'assign_genre' is a function that assigns a genre based on text

revv['genre'] = revv['Text'].apply(assign_genre)

# Visualize the distribution of genres
genre_counts = revv['genre'].value_counts()

# Exclude the 'other' genre from the data before finding the top 2
filtered_genre_counts = genre_counts[genre_counts.index != 'Other']

# Get the top 2 genres
top_genres = filtered_genre_counts.nlargest(2)

# Print top 2 genres
print("Top 2 genres:")
print(top_genres)

# Plotting the distribution of genres using a pie chart
plt.figure(figsize=(10, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Genres in Movie Reviews')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[45]:


netflix=pd.read_excel("C:/Users/whitebeard/Downloads/netflix.xlsx")


# # 

# In[46]:


netflix


# In[47]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'data1' is your DataFrame and it has columns 'genre' and 'sub_genre'

# Create a new column combining 'genre' and 'sub_genre' with a separator for better readability
netflix['genre_combo'] = netflix['genre'] + ' - ' + netflix['genre 2']

# Group by the new combined column and count the occurrences
genre_combinations = netflix.groupby('genre_combo').size()

# Reset the index to turn it back into a DataFrame for easier plotting
genre_combinations = genre_combinations.reset_index(name='Count')

# Sort the combinations for better visualization, if desired
genre_combinations = genre_combinations.sort_values(by='Count', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))  # You can adjust the size to better fit your number of categories
plt.bar(genre_combinations['genre_combo'], genre_combinations['Count'])
plt.title('Distribution of Genre and Sub-Genre Combinations in Original Dataset')
plt.xlabel('Genre and Sub-Genre')
plt.ylabel('Count')plt.xticks(rotation=90)  # Adjust rotation based on the actual data for better label visibility
plt.show()


# In[ ]:





# In[48]:


amazon=pd.read_excel("C:/Users/whitebeard/Downloads/amazon prime.xlsx")


# In[49]:


amazon


# In[50]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'data1' is your DataFrame and it has columns 'genre' and 'sub_genre'

# Create a new column combining 'genre' and 'sub_genre' with a separator for better readability
amazon['genre_combo'] = amazon['genre 1'] + ' - ' + amazon['genre 2']

# Group by the new combined column and count the occurrences
genre_combinations = amazon.groupby('genre_combo').size()

# Reset the index to turn it back into a DataFrame for easier plotting
genre_combinations = genre_combinations.reset_index(name='Count')

# Sort the combinations for better visualization, if desired
genre_combinations = genre_combinations.sort_values(by='Count', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))  # You can adjust the size to better fit your number of categories
plt.bar(genre_combinations['genre_combo'], genre_combinations['Count'])
plt.title('Distribution of Genre and Sub-Genre Combinations in Original Dataset')
plt.xlabel('Genre and Sub-Genre')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Adjust rotation based on the actual data for better label visibility
plt.show()


# In[51]:


hotstar=pd.read_excel("C:/Users/whitebeard/Downloads/disney+hotstar.xlsx")


# In[52]:


hotstar


# In[53]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'data1' is your DataFrame and it has columns 'genre' and 'sub_genre'

# Create a new column combining 'genre' and 'sub_genre' with a separator for better readability
hotstar['genre_combo'] = hotstar['genre 1'] + ' - ' + hotstar['genre 2']

# Group by the new combined column and count the occurrences
genre_combinations = hotstar.groupby('genre_combo').size()

# Reset the index to turn it back into a DataFrame for easier plotting
genre_combinations = genre_combinations.reset_index(name='Count')

# Sort the combinations for better visualization, if desired
genre_combinations = genre_combinations.sort_values(by='Count', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))  # You can adjust the size to better fit your number of categories
plt.bar(genre_combinations['genre_combo'], genre_combinations['Count'])
plt.title('Distribution of Genre and Sub-Genre Combinations in Original Dataset')
plt.xlabel('Genre and Sub-Genre')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Adjust rotation based on the actual data for better label visibility
plt.show()


# In[54]:


import pandas as pd

# Function to split genre combinations and count each genre separately
def expand_genre_counts(data):
    # Split the 'genre' column on the delimiter and expand into new rows
    all_genres = data['genre_combo'].str.split('-', expand=True).stack().str.strip()
    # Count occurrences of each genre
    return all_genres.value_counts()

# Apply the function to each DataFrame
netflix_genres = expand_genre_counts(netflix)
amazon_genres = expand_genre_counts(amazon)
hotstar_genres = expand_genre_counts(hotstar)

# Print the most popular genre from each OTT platform
print(f"Most popular genre on Netflix: {netflix_genres.idxmax()}")
print(f"Most popular genre on Amazon Prime: {amazon_genres.idxmax()}")
print(f"Most popular genre on Hotstar: {hotstar_genres.idxmax()}")

# Assuming 'top_genres' contains the top 2 genres from the movie reviews data
print("\nComparison of top genres from movie reviews with OTT platforms:")
for genre in top_genres.index:
    print(f"\nGenre: {genre}")
    print(f"Popularity in Netflix: {netflix_genres.get(genre, 0)}")
    print(f"Popularity in Amazon Prime: {amazon_genres.get(genre, 0)}")
    print(f"Popularity in Hotstar: {hotstar_genres.get(genre, 0)}")

# Suggesting which OTT platform is best for each top genre
for genre in top_genres.index:
    netflix_count = netflix_genres.get(genre, 0)
    amazon_count = amazon_genres.get(genre, 0)
    hotstar_count = hotstar_genres.get(genre, 0)

    max_count = max(netflix_count, amazon_count, hotstar_count)
    print(f"\nBest OTT platform for {genre}:")
    if max_count == 0:
        print("None of the platforms show significant preference for this genre.")
    else:
        if netflix_count == max_count:
            print("Netflix")
        if amazon_count == max_count:
            print("Amazon Prime")
        if hotstar_count == max_count:
            print("Hotstar")


# In[55]:


import pandas as pd

# Function to split genre combinations and count each combination separately
def expand_genre_combinations(data, genres):
    # Create an empty dictionary to store counts of combinations
    genre_combinations = {}
    
    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        # Split the genre combination string and strip whitespace
        genres_in_row = [genre.strip() for genre in row['genre_combo'].split('-')]
        
        # Generate all combinations of genres from the given set
        for i in range(len(genres)):
            for j in range(i + 1, len(genres)):
                genre_combination = tuple(sorted([genres[i], genres[j]]))
                
                # Increment the count of this combination in the dictionary
                genre_combinations[genre_combination] = genre_combinations.get(genre_combination, 0) + 1
    
    return genre_combinations

# Define the list of genres
given_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Thriller']

# Assuming 'netflix', 'amazon', and 'hotstar' DataFrames are defined
netflix_combinations = expand_genre_combinations(netflix, given_genres)
amazon_combinations = expand_genre_combinations(amazon, given_genres)
hotstar_combinations = expand_genre_combinations(hotstar, given_genres)

# Print the most popular genre combination from each OTT platform
print(f"Most popular genre combination on Netflix: {max(netflix_combinations, key=netflix_combinations.get)}")
print(f"Most popular genre combination on Amazon Prime: {max(amazon_combinations, key=amazon_combinations.get)}")
print(f"Most popular genre combination on Hotstar: {max(hotstar_combinations, key=hotstar_combinations.get)}")

# Suggesting which OTT platform is best for each genre combination
for combination in given_genres:
    netflix_count = netflix_combinations.get(combination, 0)
    amazon_count = amazon_combinations.get(combination, 0)
    hotstar_count = hotstar_combinations.get(combination, 0)

    max_count = max(netflix_count, amazon_count, hotstar_count)
    print(f"\nBest OTT platform for {combination}:")
    if max_count == 0:
        print("None of the platforms show significant preference for this combination.")
    else:
        if netflix_count == max_count:
            print("Netflix")
        if amazon_count == max_count:
            print("Amazon Prime")
        if hotstar_count == max_count:
            print("Hotstar")


# In[56]:



import pandas as pd

# Function to split genre combinations and count each combination separately
def expand_genre_combinations(data, genres):
    # Create an empty dictionary to store counts of combinations
    genre_combinations = {}
    
    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        # Split the genre combination string and strip whitespace
        genres_in_row = [genre.strip() for genre in row['genre_combo'].split('-')]
        
        # Check if the given combination of genres is present in the row
        if all(genre in genres_in_row for genre in genres):
            # Increment the count for this combination
            genre_combinations[tuple(genres)] = genre_combinations.get(tuple(genres), 0) + 1
    
    return genre_combinations

# Define the genres of interest (in this case, "Comedy" and "Horror")
target_genres = ["Comedy", "Horror"]

# Assuming 'netflix', 'amazon', and 'hotstar' DataFrames are defined
netflix_combinations = expand_genre_combinations(netflix, target_genres)
amazon_combinations = expand_genre_combinations(amazon, target_genres)
hotstar_combinations = expand_genre_combinations(hotstar, target_genres)

# Count occurrences of the target genre combination on each platform
netflix_count = netflix_combinations.get(tuple(target_genres), 0)
amazon_count = amazon_combinations.get(tuple(target_genres), 0)
hotstar_count = hotstar_combinations.get(tuple(target_genres), 0)

# Print the platform with the highest count for the target genre combination
if netflix_count == max(netflix_count, amazon_count, hotstar_count):
    print("Netflix is suitable for Comedy-Horror genre.")
elif amazon_count == max(netflix_count, amazon_count, hotstar_count):
    print("Amazon Prime is suitable for Comedy-Horror genre.")
elif hotstar_count == max(netflix_count, amazon_count, hotstar_count):
    print("Hotstar is suitable for Comedy-Horror genre.")
else:
    print("None of the platforms show significant preference for Comedy-Horror genre.")


# In[ ]:




