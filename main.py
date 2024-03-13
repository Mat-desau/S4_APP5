#Libraries utiliser
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import librosa as librosa

#Boolean pour les fonctions
Afficher_Graphique = True
Mise_A_Base_1 = True
Mise_En_Log = True

#Valeur utiles
pi = np.pi

def LectureToArray(NomFichier):
    # Read file to get buffer
    _, Sample_Rate = librosa.load(NomFichier)
    Fichier = wave.open(NomFichier)
    Echantillon = Fichier.getnframes()
    Audio = Fichier.readframes(Echantillon)

    # Convert buffer to float32 using NumPy
    Audio_int16 = np.frombuffer(Audio, dtype=np.int16)
    Audio_float32 = Audio_int16.astype(np.float32)

    if(Mise_A_Base_1):
        Audio_float32 = Audio_float32 / max(Audio_float32)

    Sample_Rate = Sample_Rate * 2
    return Audio_float32, Sample_Rate

def Trouver32Sinus(Array, Sample_Rate):
    #Pour cree les harmoniques
    Signal_FFT = fft(Array)
    Frequence = np.fft.fftfreq(Array.shape[-1], d=Array[1] - Array[0])

    #Faire en sorte que tout ce qui est négatif n'est pas tenu en compte
    Signal_FFT_New = []
    Frequence_New = []

    for i in range(len(Frequence)):
        if(Frequence[i] >= 0):
            Signal_FFT_New = np.append(Signal_FFT_New, Signal_FFT[i])
            Frequence_New = np.append(Frequence_New, Frequence[i])

    if(Mise_En_Log):
        Signal_FFT_New = np.log10(Signal_FFT_New)

    #Normaliser les frequences
    Frequence_New = (Frequence_New / max(Frequence_New)) * len(Signal_FFT_New)
    #Frequence_New = (Frequence_New / len(Frequence_New)) * Sample_Rate

    # find peaks en utilisant la valeur maximale comme distance
    Maximum, _ = signal.find_peaks(Signal_FFT_New, distance=np.argmax(Signal_FFT_New))

    return Frequence_New, Signal_FFT_New, Maximum

def Coupe_Bande(Array):
    


def main():
    #Lecture des audio et mise en array
    Audio_Guitare, Sample_Rate_Guitare = LectureToArray("note_guitare_LAd.wav")
    Audio_Basson, Sample_Rate_Basson = LectureToArray("note_basson_plus_sinus_1000_Hz.wav")

    #Redressement des audios
    Audio_Guitare_Redresser = abs(Audio_Guitare)
    Audio_Basson_Redresser = abs(Audio_Basson)

    Frequence_Guitare, Signal_FFT_Guitare, Maximum_Guitare = Trouver32Sinus(Audio_Guitare, Sample_Rate_Guitare)

    Frequence_Basson, Signal_FFT_Basson, Maximum_Basson = Trouver32Sinus(Audio_Basson, Sample_Rate_Basson)

    #Afficher sur les graphique au besoin
    if(Afficher_Graphique):
        #Audio Guitare
        Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
        SUB1.plot(Audio_Guitare)
        SUB1.set_title('Audio Guitare Normal')
        SUB2.plot(Audio_Guitare_Redresser)
        SUB2.set_title('Audio Guitare Redresser')

        #Audio Basson
        Figure2, (SUB3, SUB4) = plt.subplots(2, 1)
        SUB3.plot(Audio_Basson)
        SUB3.set_title('Audio Basson Normal')
        SUB4.plot(Audio_Basson_Redresser)
        SUB4.set_title('Audio Basson Redresser')

        #Harmoniques
        Figure3, (SUB5, SUB6) = plt.subplots(2, 1)
        SUB5.plot(Frequence_Guitare, Signal_FFT_Guitare)
        SUB5.plot(Maximum_Guitare, Signal_FFT_Guitare[Maximum_Guitare], "X", color='red')
        SUB5.set_title('Harmoniques Guitare')
        if(Mise_En_Log):
            SUB5.set_ylabel('Amplitude (log)')
        else:
            SUB5.set_ylabel('Amplitude')
        SUB5.set_xlabel('Fréquence')
        SUB6.plot(Frequence_Basson, Signal_FFT_Basson)
        SUB6.plot(Maximum_Basson, Signal_FFT_Basson[Maximum_Basson], "X", color='red')
        SUB6.set_title('Harmoniques Basson')
        if (Mise_En_Log):
            SUB6.set_ylabel('Amplitude (log)')
        else:
            SUB6.set_ylabel('Amplitude')
        SUB6.set_xlabel('Fréquence')
        plt.show()




if __name__ == '__main__':
    main()

