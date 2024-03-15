#Libraries utiliser
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import librosa as librosa

#Boolean pour les fonctions
Afficher_Graphique = True
Afficher_Filtres = False
Mise_A_Base_1 = True
Mise_En_Log = True

#Valeur utiles
pi = np.pi

def LectureToArray(NomFichier):
    # Read file to get buffer
    _, Sample_Rate = librosa.load(NomFichier, sr=None)
    Fichier = wave.open(NomFichier)
    Echantillon = Fichier.getnframes()
    Audio = Fichier.readframes(Echantillon)

    # Convert buffer to float32 using NumPy
    Audio_int16 = np.frombuffer(Audio, dtype=np.int16)
    Audio_float32 = Audio_int16.astype(np.float32)

    if(Mise_A_Base_1):
        Audio_float32 = Audio_float32 / max(Audio_float32)

    return Audio_float32, Sample_Rate

def Trouver32Sinus(Array, Sample_Rate):
    #Pour cree les harmoniques
    Array_Hanning = Array * np.hanning(len(Array))
    Signal_FFT = fft(Array_Hanning)   #Array FFT

    #Garder juste coté positif
    Signal_FFT_New = Signal_FFT[0:int(len(Signal_FFT)/2)]   #Juste coté positif
    Signal_FFT_New_Not_db = Signal_FFT  # Juste coté positif

    #Mise des données en LOG
    if(Mise_En_Log):
        Signal_FFT_New = 20 * np.log10(np.abs(Signal_FFT_New)) #Mise en log

    #Normaliser les frequences
    Frequence_New = np.linspace(0, 0.5, len(Signal_FFT_New)) * Sample_Rate  #Creation de la fréquence qui est en fonction du sample rate
    Frequence_Full = np.linspace(0, 1, len(Signal_FFT_New_Not_db)) * Sample_Rate

    #find peaks en utilisant la valeur maximale comme distance
    Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, distance=np.argmax(Signal_FFT_New)) #Find peaks en fonction du plus haut (Position en X)
    Position_Maximum2, _ = signal.find_peaks(Signal_FFT_New, height=(0.85 * np.max(Signal_FFT_New)), distance=40)   #Trouver Max en fonction de 10% (Position en X)

    #Ajustement pour Basson
    if(Position_Maximum2[0] < Position_Maximum[0]):
        Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, distance = (Position_Maximum2[1] - Position_Maximum2[0]))   #Si jamais il a un plus petit avant le plus haut

    #Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, distance=np.argmax(Signal_FFT_New))
    #Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, )

    #Garder juste les 32 premiers
    Position_Maximum = Position_Maximum[:32]    #Juste garder les 32 premiers index des max

    Freq_Max = Frequence_New[Position_Maximum]  #Trouver les fréquences maximum et non juste les indexs

    return Frequence_New, Frequence_Full, Signal_FFT_New, Signal_FFT_New_Not_db, Position_Maximum, Freq_Max

def Trouver_Enveloppe(Signal, Attente, Fstop, Fpass):
    N = Calcul_N(Attente, Fstop, Fpass)   #Calcul du N avec la fonction du manuel
    Passe_Bas_Valeur = Passe_Bas(2*len(Signal), int(N))   #Filtre passe Bas
    Enveloppe = Passe_Bas_Valeur * Signal   #Trouver l'enveloppe
    Enveloppe_Temps = np.fft.ifft(Enveloppe)    #Inverse pour la mettre en fontion du temps au lieu d'en fontion des fréquences

    return Enveloppe, Enveloppe_Temps, Passe_Bas_Valeur

def Calcul_N(Atten, Fstop_db, Fpass_db):
    return Atten / (22 * (Fstop_db - Fpass_db)) #Formule du manuel

def Passe_Bas(Len_Array, N):
    h = np.ones(N) * (1/N)
    h = fft(h, n=Len_Array)
    h = h[0:int(len(h)/2)]
    return h

def Coupe_Bande(Array, Sample_Rate, N):
    w0 = 2 * pi * 1000 #en rad
    w1 = 20 # en Hz
    K = (((w1/Sample_Rate) * 2) * 2) + (1/N)
    k = np.linspace(-(N-1)/2, (N-1)/2, N)

    h = (1 / N) * (np.sin(pi * k * K) / np.sin((pi * k) / N))
    #h = (1 / N) * (np.sin((pi * k * K) / N)/np.sin((pi * k) / N))            # Voici la vrai formule mais avec cette formule ça marche pas

    diract = np.zeros(N)
    diract[int(N / 2)] = 1

    #F = diract - 2 * h * np.cos(w0 * k)
    F = - 2 * h * np.cos(w0 * k)

    Valeur_FFT = np.fft.fftshift(np.fft.fft(h))
    Valeur_FFT_Freq = (np.fft.fftshift(np.fft.fftfreq(N))) * Sample_Rate

    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(Valeur_FFT_Freq, np.abs(Valeur_FFT))
    SUB2.plot(k, h)
    plt.show()

def plot1(X1, Y1, Titre1):
    Figure1, SUB1 = plt.subplots(1, 1)
    SUB1.plot(X1, Y1)
    SUB1.set_title(Titre1)
    plt.show()

def plot1_2(X1, Y1, X2, Y2, Titre1):
    Figure1, SUB1 = plt.subplots(1, 1)
    SUB1.plot(X1, Y1)
    SUB1.plot(X2, "X", color='red')
    SUB1.set_title(Titre1)
    plt.show()

def plot2(X1, X2, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1)
    SUB1.set_title(Titre1)
    SUB2.plot(X2)
    SUB2.set_title(Titre2)
    plt.show()

def plot2_2(X1, Y1, X2, Y2, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1, Y1)
    SUB1.set_title(Titre1)
    SUB2.plot(X2, Y2)
    SUB2.set_title(Titre2)
    plt.show()

def plot3(X1, Y1, X2, Y2, X3, Y3, X4, Y4, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1, Y1)
    SUB1.plot(X1[X3], Y3[X3], "X", color='red')
    SUB1.set_title(Titre1)
    SUB2.plot(X2, Y2)
    SUB2.plot(X2[X4], Y4[X4], "X", color='red')
    SUB2.set_title(Titre2)
    plt.show()

def main():
    #Lecture des audio et mise en array
    Audio_Guitare, Sample_Rate_Guitare = LectureToArray("note_guitare_LAd.wav")
    Audio_Basson, Sample_Rate_Basson = LectureToArray("note_basson_plus_sinus_1000_Hz.wav")

    #Redressement des audios
    Audio_Guitare_Redresser = abs(Audio_Guitare)
    Audio_Basson_Redresser = abs(Audio_Basson)

    #Filtres
    #Coupe_Bande(Audio_Basson, Sample_Rate_Basson, 4096)

    #Trouver les 32 frequences
    Frequence_Guitare, Frequence_FFT_Full_Guitare, Signal_FFT_Guitare, Signal_FFT_Not_db_Guitare, Positon_Maximum_Guitare, Frequence_Maximum_Guitare = Trouver32Sinus(Audio_Guitare, Sample_Rate_Guitare)
    Frequence_Basson, Frequence_FFT_Full_Basson, Signal_FFT_Basson, Signal_FFT_Not_db_Basson, Positon_Maximum_Basson, Frequence_Maximum_Basson = Trouver32Sinus(Audio_Basson, Sample_Rate_Basson)

    #Enveloppe
    Enveloppe_Guitare, Enveloppe_Temps_Guitare, Passe_Bas_Valeur_Guitare = Trouver_Enveloppe(Signal_FFT_Not_db_Guitare, 3, (pi/1000)/(2*pi), 0) #fois 2pi pour le mettre en Hz, puisque le reste est en Hertz
    Enveloppe_Basson, Enveloppe_Temps_Basson, Passe_Bas_Valeur_Basson = Trouver_Enveloppe(Signal_FFT_Not_db_Basson, 3, (pi/1000)/(2*pi), 0)

    # Affichage
    if (Afficher_Filtres):
        plot1(Frequence_Guitare, np.abs(Passe_Bas_Valeur_Guitare), 'Filte Passe-Bas Guitare')
        plot1(Frequence_Basson, np.abs(Passe_Bas_Valeur_Basson), 'Filte Passe-Bas Basson')

    #Afficher sur les graphique au besoin
    if(Afficher_Graphique):
        plot2(Audio_Guitare, Audio_Guitare_Redresser, 'Audio Guitare Normal', 'Audio Guitare Redresser')
        plot2(Audio_Basson, Audio_Basson_Redresser, 'Audio Basson Normal', 'Audio Basson Redresser')
        #plot2_2(Frequence_Guitare, Signal_FFT_Guitare, Frequence_Basson, Signal_FFT_Basson, 'Harmoniques Guitare', 'Harmoniques Basson')
        plot3(Frequence_Guitare, Signal_FFT_Guitare, Frequence_Basson, Signal_FFT_Basson, Positon_Maximum_Guitare, Signal_FFT_Guitare, Positon_Maximum_Basson, Signal_FFT_Basson, 'Harmoniques Guitare', 'Harmoniques Basson')
        plot2_2(Frequence_FFT_Full_Guitare, np.abs(Enveloppe_Guitare), Frequence_FFT_Full_Guitare, np.abs(Enveloppe_Temps_Guitare), 'Enveloppe Fréquence Guitare', 'Enveloppe Temps Guitare')
        plot2_2(Frequence_FFT_Full_Basson, np.abs(Enveloppe_Basson), Frequence_FFT_Full_Basson, np.abs(Enveloppe_Temps_Basson), 'Enveloppe Fréquence Basson', 'Enveloppe Temps Basson')

if __name__ == '__main__':
    main()

