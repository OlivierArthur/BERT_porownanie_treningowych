Konkretne parametry użyte podczas treningu można zobaczyć na https://huggingface.co/OliverArt5500/klasyfikatorspamu1 

W projekcie wykorzystano środowisko google colab z "!pip install evaluate".

Obraz na dockerze znajduje się tutaj: https://hub.docker.com/repository/docker/oliverart5500/klasyfikatorspamu/general

Jak korzystać ( używając dockera ):
wpisać w terminalu
1. docker run -d -p 8080:8000 --name spam-api oliverart5500/klasyfikatorspamu
2. na http://localhost:8080/docs będzie interaktyne GUI, do którego można wkleić maila








