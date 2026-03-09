from langchain_community.document_loaders.dataframe import DataFrameLoader
import pandas as pd


def load_movies():
    movies_raw = pd.read_csv("IMDB.csv")
    movies_raw.head()
    
    # Rename primaryTitle, Description columns. Assign to movies.
    movies = movies_raw.rename(columns = {
        "primaryTitle": "movie_title",
        "Description": "movie_description",
    })

    # Add source column from tconst
    movies["source"] = "https://www.imdb.com/title/" + movies["tconst"]

    # Subset for titleType equal to "movie"
    movies = movies.loc[movies["titleType"] == "movie"]

    # Remove N/A values for movie_description
    movies["movie_description"] = movies["movie_description"].fillna("No description")

    # Select movie_title, movie_description, source, genres columns
    movies = movies[["movie_title", "movie_description", "source", "genres"]]

    # Show the head of movies
    movies.head()
    #create page content column
    movies["page_content"] = "Title: " + movies["movie_title"] + "\n" + \
                            "Genre: " + movies["genres"] + "\n" + \
                            "Description: " + movies["movie_description"]

    # select the page_content and source columns
    movies = movies[["page_content", "source"]]
    
    # load the documents from the dataframe into docs
    docs = DataFrameLoader(movies, page_content_column="page_content").load()
    
    return docs

def calculate_estimated_cost(docs):
    # Import tiktoken
    import tiktoken
    
    # Create the encoder
    encoder = tiktoken.encoding_for_model("text-embedding-3-large")
    
    # Create a list containing the number of tokens for each document
    tokens_per_doc = [len(encoder.encode(doc.page_content)) for doc in docs]
    
    # Show the estimated cost, which is the sum of the amount of tokens divided by 1000, times $0.0001
    total_tokens = sum(tokens_per_doc)
    cost_per_1M_tokens = 0.13
    cost =  (total_tokens / 1_000_000) * cost_per_1M_tokens
    return cost
    

def main():
    print("Hello from searchmovieopenai!")
    docs = load_movies()
    print(docs)
    cost = calculate_estimated_cost(docs)
    print(f"Estimated cost: ${cost:.6f}")


if __name__ == "__main__":
    main()
