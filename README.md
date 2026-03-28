Konkretne parametry użyte podczas treningu modelu można zobaczyć tutaj: 
https://huggingface.co/OliverArt5500/klasyfikatorspamu1

Model został wytrenowany z wykorzystaniem środowiska Google Colab (z użyciem `evaluate`, wymagane !pip install evaluate).

Gotowy obraz środowiska znajduje się na Docker Hub: 
https://hub.docker.com/repository/docker/oliverart5500/klasyfikatorspamu/general


Aby uruchomić gotowy serwer z modelem, wpisz w terminalu poniższą komendę:

docker run -d -p 8080:8000 --name spam-api oliverart5500/klasyfikatorspamu

Na http://localhost:8080/docs będzie interkatywne GUI z włączonym klasyfikatorem.



