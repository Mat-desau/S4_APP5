#Libraries utiliser
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import librosa as librosa

#Boolean pour les fonctions
Afficher_Graphique = False
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
    Signal_FFT = fft(Array_Hanning)

    Signal_FFT_New = Signal_FFT[0:int(len(Signal_FFT)/2)]

    if(Mise_En_Log):
        Signal_FFT_New = 20 * np.log10(np.abs(Signal_FFT_New))

    #Normaliser les frequences
    Frequence_New = np.linspace(0, 0.5, len(Signal_FFT_New)) * Sample_Rate

    #find peaks en utilisant la valeur maximale comme distance
    Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, distance=np.argmax(Signal_FFT_New))
    Position_Maximum2, _ = signal.find_peaks(Signal_FFT_New, height=(0.85 * np.max(Signal_FFT_New)), distance=40)

    if(Position_Maximum2[0] < Position_Maximum[0]):
        Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, distance = (Position_Maximum2[1] - Position_Maximum2[0]))

    #Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, distance=np.argmax(Signal_FFT_New))
    #Position_Maximum, _ = signal.find_peaks(Signal_FFT_New, )
    Position_Maximum = Position_Maximum[:32]

    Freq_Max = Frequence_New[Position_Maximum]

    return Frequence_New, Signal_FFT_New, Position_Maximum, Freq_Max

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

def plot(X1, X2, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1)
    SUB1.set_title(Titre1)
    SUB2.plot(X2)
    SUB2.set_title(Titre2)
    plt.show()

def plot2(X1, Y1, X2, Y2, Titre1, Titre2):
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

    Coupe_Bande(Audio_Basson, Sample_Rate_Basson, 4096)

    Frequence_Guitare, Signal_FFT_Guitare, Positon_Maximum_Guitare, Frequence_Maximum_Guitare = Trouver32Sinus(Audio_Guitare, Sample_Rate_Guitare)

    Frequence_Basson, Signal_FFT_Basson, Positon_Maximum_Basson, Frequence_Maximum_Basson = Trouver32Sinus(Audio_Basson, Sample_Rate_Basson)

    #Afficher sur les graphique au besoin
    if(Afficher_Graphique):

        plot(Audio_Guitare, Audio_Guitare_Redresser, 'Audio Guitare Normal', 'Audio Guitare Redresser')
        plot(Audio_Basson, Audio_Basson_Redresser, 'Audio Basson Normal', 'Audio Basson Redresser')
        #plot2(Frequence_Guitare, Signal_FFT_Guitare, Frequence_Basson, Signal_FFT_Basson, 'Harmoniques Guitare', 'Harmoniques Basson')
        plot3(Frequence_Guitare, Signal_FFT_Guitare, Frequence_Basson, Signal_FFT_Basson, Positon_Maximum_Guitare, Signal_FFT_Guitare, Positon_Maximum_Basson, Signal_FFT_Basson, 'Harmoniques Guitare', 'Harmoniques Basson')

if __name__ == '__main__':
    main()

