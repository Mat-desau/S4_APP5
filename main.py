#Libraries utiliser
import numpy as np
import wave
import matplotlib.pyplot as plt

#Boolean pour les fonctions
Afficher_Graphique = True

#Valeur utiles
pi = np.pi


def LectureToArray(NomFichier):
    # Read file to get buffer
    Fichier = wave.open(NomFichier)
    Echantillon = Fichier.getnframes()
    Audio = Fichier.readframes(Echantillon)

    # Convert buffer to float32 using NumPy
    Audio_int16 = np.frombuffer(Audio, dtype=np.int16)
    Audio_float32 = Audio_int16.astype(np.float32)

    return Audio_float32

def Trouver32Sinus(Array):
    return Array

def main():
    #Lecture des audio et mise en array
    Audio_Guitare = LectureToArray("note_guitare_LAd.wav")
    Audio_Basson = LectureToArray("note_basson_plus_sinus_1000_Hz.wav")

    #Redressement des audios
    Audio_Guitare_Redresser = abs(Audio_Guitare)
    Audio_Basson_Redresser = abs(Audio_Basson)

    #Afficher sur les graphique au besoin
    if(Afficher_Graphique):
        Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
        SUB1.plot(Audio_Guitare)
        SUB1.set_title('Audio Guitare Normal')
        SUB2.plot(Audio_Guitare_Redresser)
        SUB2.set_title('Audio Guitare Redresser')

        Figure2, (SUB3, SUB4) = plt.subplots(2, 1)
        SUB3.plot(Audio_Basson)
        SUB3.set_title('Audio Basson Normal')
        SUB4.plot(Audio_Basson_Redresser)
        SUB4.set_title('Audio Basson Redresser')

        plt.show()




if __name__ == '__main__':
    main()

