import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_movie(description):
    movie_descriptions = [
        "When Hiccup discovers Toothless isn't the only Night Fury, he must seek 'The Hidden World', a secret Dragon Utopia before a hired tyrant named Grimmel finds it first.",
        "After the death of Superman, several new people present themselves as possible successors.",
        "A darkness swirls at the center of a world-renowned dance company, one that will engulf the artistic director, an ambitious young dancer, and a grieving psychotherapist. Some will succumb to the nightmare. Others will finally wake up.",
        "A humorous take on Sir Arthur Conan Doyle's classic mysteries featuring Sherlock Holmes and Doctor Watson.",
        "A 16-year-old girl and her extended family are left reeling after her calculating grandmother unveils an array of secrets on her deathbed.",
        "In the last moments of World War II, a young German soldier fighting for survival finds a Nazi captain's uniform. Impersonating an officer, the man quickly takes on the monstrous identity of the perpetrators he is trying to escape from.",
        "The world at an end, a dying mother sends her young son on a quest to find the place that grants wishes.",
        "A musician helps a young singer and actress find fame, even as age and alcoholism send his own career into a downward spiral.",
        "Corporate analyst and single mom, Jen, tackles Christmas with a business-like approach until her uncle arrives with a handsome stranger in tow.",
        "Adapted from the bestselling novel by Madeleine St John, Ladies in Black is an alluring and tender-hearted comedy drama about the lives of a group of department store employees in 1959 Sydney."
    ]

    # Use TfidfVectorizer to convert the descriptions into vectors
    vectorizer = TfidfVectorizer()
    movie_vectors = vectorizer.fit_transform(movie_descriptions)

    # Convert the input description into a vector
    input_vector = vectorizer.transform([description])

    # Calculate the cosine similarity between the input vector and each movie vector
    similarities = cosine_similarity(input_vector, movie_vectors)[0]

    # Get the index of the most similar movie
    most_similar_index = np.argmax(similarities)

    # Return the title of the most similar movie
    titles = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E", "Movie F", "Movie G", "Movie H", "Movie I", "Movie J"]
    return titles[most_similar_index]

description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator."
print(find_similar_movie(description))  # Output: Movie G
