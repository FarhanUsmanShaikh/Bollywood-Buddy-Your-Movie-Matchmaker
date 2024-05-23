from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
with open('bollywood_recommendation_model.pkl', 'rb') as f:
    sig = pickle.load(f)

# Load the movies DataFrame
movies_df = pd.read_csv("indian movies.csv")

# Preprocessing to match the loaded model requirements
# Create a Series with movie names in lower case as index and movie indices as values
indices = pd.Series(movies_df.index, index=movies_df['Movie Name'].str.lower()).drop_duplicates()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        if movie_name.strip() == "":
            return render_template('index.html', recommendation_text="Please enter a movie name.")
        
        # Convert input to lower case
        movie_name_lower = movie_name.lower()  
        
        if movie_name_lower not in indices:
            return render_template('index.html', recommendation_text="Sorry, the movie is not in our database.")
        
        idx = indices[movie_name_lower]
        
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        
        if idx >= len(sig) or idx < 0:
            return render_template('index.html', recommendation_text="Sorry, the movie index is out of bounds.")
        
        # Get the similarity scores for the selected movie
        sig_scores = list(enumerate(sig[idx]))
        
        # Sort the movies based on similarity scores in descending order
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top 10 similar movies
        sig_scores = sig_scores[1:11]
        
        # Get the indices of the top 10 movies
        movie_indices = [i[0] for i in sig_scores]
        
        # Get the names of the top 10 movies
        recommended_movies = movies_df['Movie Name'].iloc[movie_indices].tolist()
        
        # Return the recommended movies as a string with each movie name on a new line
        return render_template('index.html', recommendation_text='<br>'.join(recommended_movies))
    else:
        # If the request method is GET, return the home page
        return render_template('index.html')

if __name__ == "__main__":
    # Run the Flask application in debug mode on specific port 
    app.run(debug=True, port=5006)