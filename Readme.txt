Classification Dog vs Cat

Python 3.6.5
Modules :
- Tensorflow 1.4.0
- Keras 2.0.6
- Numpy 1.13.3
- Matplotlib 2.1.0
- opencv

Ce code entraine des mod�les de r�seaux de neurones convolutifs � classifier des images de chats et de chiens.

Nombre d'images pour l'apprentissage, dossier "/train" :
- 10 000 images de chats
- 10 000 images de chiens

Nombre d'image pour le test, dossier "/test":
- 2 500 images de chats
- 2 500 images de chiens

Param�tres :

Redimensionnement de l'image :
- Height
- Width

Nombre de donn�es d'apprentissage :
- train_data_size : nombre d'image d'entrainements
- test_data_size : nombre d'image pour le test

Mod�les impl�ment�s  :
- R�seaux de neurones convolutifs � 6, 11, 16 et 19 couches

Classes :
- la classe data dans le fichier preprocess.py, permet de pr�-traiter les images pour quelles soient exploitable pour les r�seaux de neurones
elle prend en param�tre, le nombre d'images pour l'apprentissage, le nombre d'image pour le test et les dimensions des images

- la classe model dans le fichier model.py, permet d'entrainer des mod�les et de les tester
elle prend en param�tre, les matrices d'apprentissage et de tests, le nombre d'it�ration (epoch), la taille du batch