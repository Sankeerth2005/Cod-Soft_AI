import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    'User': ['Harsh', 'Harsh', 'Dipanshu', 'Dipanshu', 'Mrunal', 'Mrunal', 'Rahul', 'Rahul', 'Gaurav', 'Gaurav'],
    'Item': ['KGF', 'DDLJ', 'KGF', 'RA1', 'DDLJ', 'RA1', 'RA1', 'Interstellar', 'Ironman', 'Interstellar'],
    'Rating': [5, 4, 3, 2, 4, 5, 1, 3, 4, 5],
}

df = pd.DataFrame(data)

# Create a user-item matrix
user_item_matrix = df.pivot_table(index='User', columns='Item', values='Rating', fill_value=0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Function to get recommendations for a given user
def get_recommendations(user):
    user_ratings = user_item_matrix.loc[user]
    user_index = user_item_matrix.index.get_loc(user)
    similar_users = np.argsort(user_similarity[user_index])[::-1][1:]  # Exclude the user itself
    
    recommendations = []

    for item in user_item_matrix.columns:
        if user_ratings[item] == 0:  # User hasn't rated the item
            weighted_sum = sum(user_similarity[user_index][similar_user] * user_item_matrix.iloc[similar_user][item]
                               for similar_user in similar_users)
            total_similarity = sum(user_similarity[user_index][similar_user] for similar_user in similar_users)
            
            if total_similarity > 0:
                predicted_rating = weighted_sum / total_similarity
                recommendations.append((item, predicted_rating))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations

# Function to handle button click event
def recommend():
    user_input = entry.get().capitalize()  # Capitalize the input to match the user names
    if user_input in user_item_matrix.index:
        recommendations = get_recommendations(user_input)
        print(recommendations)  # Debugging line
        recommendation_text.set(f"Top recommendations for {user_input}:\n" + "\n".join([f"{item}: {rating:.2f}" for item, rating in recommendations[:5]]))
    else:
        messagebox.showinfo("User Not Found", f"User {user_input} not found in the dataset.")

# GUI setup
window = tk.Tk()
window.title("Movie Recommendation System")

label = tk.Label(window, text="Enter User (Harsh, Dipanshu, Mrunal, Rahul, Gaurav):")
label.pack(pady=10)

entry = tk.Entry(window)
entry.pack(pady=10)

button = tk.Button(window, text="Get Recommendations", command=recommend)
button.pack(pady=10)

recommendation_text = tk.StringVar()
result_label = tk.Label(window, textvariable=recommendation_text)
result_label.pack(pady=10)

window.mainloop()