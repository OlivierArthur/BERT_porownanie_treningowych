Uwaga: kod jest jedynie poglądowy, nie ma wersji użytej do pierwszego eksperymentu w repozytorium - ale jest opisane co trzeba zmienić żeby go odtworzyć.

Porównanie trzech różnych zbiorów danych z emailami, dwa z bardziej realistyczną proporcją spamu do prawdziwych wiadomości, jeden 50/50.
W danych pojawiają się "Subject: " (w enron i ling) i ciągi mailów, w pierwszej iteracji eksperymentu chcemy je zostawić i zobaczyć co się dzieje ( czy zepsują coś? )

W pierwszej iteracji porównania training setów w TrainingArguments nie było weight decay, były 3 epoki i learning rate 3e-5. Na wykresach można było zaobserwować overtraining w zbiorach treningowych enron i lingspam. W przypadku SpamAssassin trenowanie przez 3 epoki z większym learning rate było efektywne, co można zobaczyć na wykresach w folderach.

W eksperymencie dwa zmniejszamy learning rate, robimy 2 epoki i dodajemy weight decay - wszystko żeby zapobiegać overfitting.
We wszystkich zbiorach danych było to efektywne.



