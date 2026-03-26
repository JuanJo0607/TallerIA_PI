import os
import random
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


class Command(BaseCommand):
    help = 'Selecciona una película al azar y muestra su embedding (binary emb o generarlo si no existe).'

    def add_arguments(self, parser):
        parser.add_argument('--use-db-emb', action='store_true', help='Usar emb guardado en la BD si existe')

    def handle(self, *args, **options):
        load_dotenv('../openAI.env')
        api_key = os.environ.get('openai_apikey')
        if not api_key:
            self.stderr.write(self.style.ERROR('openai_apikey no encontrada en openAI.env'))
            return

        client = OpenAI(api_key=api_key)

        movies = list(Movie.objects.all())
        if not movies:
            self.stderr.write(self.style.ERROR('No hay películas en la base de datos.'))
            return

        movie = random.choice(movies)
        self.stdout.write(self.style.NOTICE(f"Película aleatoria seleccionada: {movie.title}"))
        self.stdout.write(f"Descripción: {movie.description}")

        embedding = None
        if options['use_db_emb'] and movie.emb:
            try:
                embedding = np.frombuffer(movie.emb, dtype=np.float32)
                self.stdout.write(self.style.SUCCESS('🚀 Embedding cargado desde la base de datos.'))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error al leer el emb almacenado: {e}"))

        if embedding is None:
            self.stdout.write('Generando embedding con OpenAI')
            response = client.embeddings.create(input=[movie.description], model='text-embedding-3-small')
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            if movie.emb is None:
                movie.emb = embedding.tobytes()
                movie.save(update_fields=['emb'])
                self.stdout.write(self.style.SUCCESS('Guardado embedding en la base de datos.'))

        self.stdout.write(self.style.SUCCESS(f'Embedding length: {len(embedding)}'))
        first_values = ', '.join([f"{x:.6f}" for x in embedding[:10]])
        self.stdout.write(f'Primeros 10 valores: [{first_values}]')

        # Si hay campo emb en DB, opcional mostrar similitud self con prompt de ejemplo
        prompt = 'película sobre la Segunda Guerra Mundial'
        prompt_emb = np.array(client.embeddings.create(input=[prompt], model='text-embedding-3-small').data[0].embedding, dtype=np.float32)
        similarity = cosine_similarity(prompt_emb, embedding)
        self.stdout.write(self.style.NOTICE(f"Similitud con '{prompt}': {similarity:.4f}"))