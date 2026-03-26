from django.shortcuts import render
from django.http import HttpResponse

from .models import Movie

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

def home(request):
    import os, numpy as np
    from openai import OpenAI
    from dotenv import load_dotenv

    searchTerm = request.GET.get('searchMovie')
    movies = Movie.objects.all()
    recommendation = None

    if searchTerm:
        # Búsqueda tradicional por título
        movies = Movie.objects.filter(title__icontains=searchTerm)

        # Recomendación semántica con embeddings
        load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        response = client.embeddings.create(input=[searchTerm], model="text-embedding-3-small")
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        best_movie, max_sim = None, -1
        for movie in Movie.objects.exclude(emb=None):
            sim = cosine_similarity(prompt_emb, np.frombuffer(movie.emb, dtype=np.float32))
            if sim > max_sim:
                max_sim, best_movie = sim, movie

        recommendation = {'movie': best_movie, 'similarity': round(float(max_sim), 4)}

    return render(request, 'home.html', {
        'searchTerm': searchTerm,
        'movies': movies,
        'recommendation': recommendation,
    })


def about(request):
    #return HttpResponse('<h1>Welcome to About Page</h1>')
    return render(request, 'about.html')

def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email':email})


def statistics_view0(request):
    matplotlib.use('Agg')
    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # Crear un diccionario para almacenar la cantidad de películas por año
    movie_counts_by_year = {}

    # Filtrar las películas por año y contar la cantidad de películas por año
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    # Ancho de las barras
    bar_width = 0.5
    # Posiciones de las barras
    bar_positions = range(len(movie_counts_by_year))

    # Crear la gráfica de barras
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')

    # Personalizar la gráfica
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)

    # Ajustar el espaciado entre las barras
    plt.subplots_adjust(bottom=0.3)

    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    # Renderizar la plantilla statistics.html con la gráfica
    return render(request, 'statistics.html', {'graphic': graphic})

def statistics_view(request):
    matplotlib.use('Agg')
    # Gráfica de películas por año
    all_movies = Movie.objects.all()
    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    # Gráfica de películas por género
    movie_counts_by_genre = {}
    for movie in all_movies:
        # Obtener el primer género
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic



def recommend(request):
    result = None
    prompt = ""

    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')

        load_dotenv('../openAI.env')
        client = OpenAI(api_key=os.environ.get('openai_apikey'))

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        response = client.embeddings.create(input=[prompt], model="text-embedding-3-small")
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        best_movie = None
        max_similarity = -1

        for movie in Movie.objects.exclude(emb=None):
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = cosine_similarity(prompt_emb, movie_emb)
            if similarity > max_similarity:
                max_similarity = similarity
                best_movie = movie

        result = {'movie': best_movie, 'similarity': round(float(max_similarity), 4)}

    return render(request, 'recommend.html', {'result': result, 'prompt': prompt})