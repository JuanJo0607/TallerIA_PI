import os
import numpy as np
from django.shortcuts import render
from openai import OpenAI
from dotenv import load_dotenv
from movie.models import Movie

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend(request):
    result = None
    prompt = ""

    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')
        load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        response = client.embeddings.create(input=[prompt], model="text-embedding-3-small")
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        best_movie, max_sim = None, -1
        for movie in Movie.objects.exclude(emb=None):
            sim = cosine_similarity(prompt_emb, np.frombuffer(movie.emb, dtype=np.float32))
            if sim > max_sim:
                max_sim, best_movie = sim, movie

        result = {'movie': best_movie, 'similarity': round(float(max_sim), 4)}

    return render(request, 'recommend.html', {'result': result, 'prompt': prompt})