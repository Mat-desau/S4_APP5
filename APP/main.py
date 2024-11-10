#Libraries utiliser
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy import signal
import numpy.fft as fft
#from scipy.fft import fft
import librosa as librosa

#Boolean pour les fonctions
Afficher_Graphique = True
Afficher_Graphique_Changement_Frequence = True
Afficher_Graphique_Coupe_bande = True
Afficher_Filtres = False
Print_Valeurs = False
Print_Tableau_Freq_Notes = False
Print_Tableau_Ampliude_Notes = False
Mise_A_Base_1 = True
Mise_En_Log = True
Facteur_De_Grosseur = 20

#Valeur utiles
pi = np.pi

def LectureToArray(NomFichier):
    # Read file to get buffer
    _, Sample_Rate = librosa.load(NomFichier, sr=None)                                                                  #Lecture du sample Rate
    Fichier = wave.open(NomFichier)
    Echantillon = Fichier.getnframes()
    Audio = Fichier.readframes(Echantillon)

    # Convert buffer to float32 using NumPy
    Audio_int16 = np.frombuffer(Audio, dtype=np.int16)
    Audio_float32 = Audio_int16.astype(np.float32)

    Max = max(Audio_float32)                                                                                            #Garder en mémoire le Max

    if(Mise_A_Base_1):
        Audio_float32 = Audio_float32 / max(Audio_float32)                                                              #Normaliser à 1

    return Audio_float32, Sample_Rate, Max

def Trouver32Sinus(Array, Sample_Rate):
    #Pour cree les harmoniques
    Array_Hanning = Array * np.hanning(len(Array))
    Signal_FFT = fft.fft(Array_Hanning)                                                                                 #Array FFT

    #Garder juste coté positif
    Signal_FFT_Pos_db = Signal_FFT[0:int(len(Signal_FFT)/2)]                                                            #Juste coté positif
    Signal_FFT_Pos_Not_db = Signal_FFT[0:int(len(Signal_FFT)/2)]                                                        #Juste coté positif

    #Mise des données en LOG
    if(Mise_En_Log):
        Signal_FFT_Pos_db = 20 * np.log10(Signal_FFT_Pos_db)                                                            #Mise en log

    #Normaliser les frequences
    Frequence_Pos = np.linspace(0, 0.5, len(Signal_FFT_Pos_db)) * Sample_Rate                                           #Creation de la fréquence qui est en fonction du sample rate
    Frequence_Full = np.linspace(0, 1, len(Signal_FFT_Pos_Not_db)) * Sample_Rate                                        #Garder tout les fréquences

    #find peaks en utilisant la valeur maximale comme distance
    Position_Maximum, _ = signal.find_peaks(Signal_FFT_Pos_db, distance=int(np.argmax(Signal_FFT_Pos_db)))              #Find peaks en fonction du plus haut (Position en X)
    Position_Maximum2, _ = signal.find_peaks(Signal_FFT_Pos_db, height=(0.80 * np.max(Signal_FFT_Pos_db)), distance=10) #Trouver Max en fonction de 10% (Position en X)

    #Ajustement pour Basson
    if(Position_Maximum2[0] < Position_Maximum[0]):
        Position_Maximum, _ = signal.find_peaks(Signal_FFT_Pos_db, distance=int(np.argmax(Signal_FFT_Pos_db)/3))        #Si jamais il a un plus petit avant le plus haut

    #Garder juste les 32 premiers
    Position_Maximum = Position_Maximum[:32]                                                                            #Juste garder les 32 premiers index des max
    Amplitude_Max = Signal_FFT_Pos_db[Position_Maximum]

    Freq_Max = Frequence_Pos[Position_Maximum]                                                                          #Trouver les fréquences maximum et non juste les indexs

    #print(Freq_Max.astype(int))                                                                                        #Pour Rapport
    #print(np.abs(Amplitude_Max))
    #print(np.angle(Amplitude_Max, deg=True))

    return Frequence_Pos, Frequence_Full, Signal_FFT_Pos_db, Signal_FFT_Pos_Not_db, Position_Maximum, Freq_Max

def Trouver_Enveloppe(Signal, Sample_Rate):
    Val_Passe_Bas = Calcul_N(100000, Sample_Rate)                                                                       #Calcul du N avec la fonction du manuel

    Enveloppe_Temps = np.convolve(np.abs(Signal), Val_Passe_Bas, mode='same')                                           #Creation enveloppe temporelle en utilisant la convolution
    Enveloppe_Temps = Enveloppe_Temps / max(Enveloppe_Temps)                                                            #Normaliser à 1 l'enveloppe temporelle
    return Enveloppe_Temps

def Calcul_N(Longeur_Echantillon, Sample_Rate):
    DB3 = 10**(-3/20)                                                                                                   #Valeur de DB = 3
    pos = int(((pi/1000)/(2*pi)) * Longeur_Echantillon)                                                                 #Trouver la position à pi/1000

    N = 1
    gain = 1

    while (gain > DB3):
        h = np.ones(N) * (1/N)

        H = fft.fft(h, n=Longeur_Echantillon)

        gain = np.abs(H[pos])
        N += 1

    if(Afficher_Filtres):
        plot1(np.linspace(0, Sample_Rate, Longeur_Echantillon), H, 'Passe-Bas')                                         #Affichage des graphiques

    return h

def Passe_Bas(Len_Array, N):
    h = np.ones(N) * (1/N)
    h = fft(h, n=Len_Array)
    h = h[0:int(len(h)/2)]
    return h

def Coupe_Bande(Audio_Basson, Max, Freq_Coupure, Freq_Coupure2, Sample_Rate, N):
    w0 = (2 * pi * Freq_Coupure2) / Sample_Rate #en rad
    k = np.linspace(-(N-1)/2, (N-1)/2, N)
    #w1 = pi * ((k-1)/N)
    w1 = Freq_Coupure         # en Hz
    K = (((w1 / Sample_Rate) * 2) * 2) + (1 / N)

    h = (1 / N) * (np.sin((pi * k * K)) / np.sin((pi * k) / N))

    #Mettre a 0
    zero, = np.nonzero(k == 0)
    if len(zero):
        h[zero] = K

    h /= np.sum(h)

    diract = np.zeros(len(h))
    diract[int(len(h)/2)] = 1

    F = diract - 2 * h * np.cos(w0 * k)

    #Double Caller
    Audio_Filtrer = np.convolve(Audio_Basson, F, mode='same')
    Audio_Filtrer2 = np.convolve(Audio_Filtrer, F, mode='same')
    Audio_Filtrer3 = np.convolve(Audio_Filtrer2, F, mode='same')
    Audio_Filtrer3 *= Max
    Audio_Basson *= Max

    Valeur_FFT = fft.fftshift(np.fft.fft(F))
    Valeur_FFT_Freq = (fft.fftshift(np.fft.fftfreq(N))) * Sample_Rate

    if(Afficher_Graphique_Coupe_bande):
        Figure1, (SUB1, SUB2, SUB3, SUB4) = plt.subplots(4, 1)
        SUB1.plot(Valeur_FFT_Freq, np.abs(Valeur_FFT))
        SUB1.set_title('FFT du Coupe-Bande')
        SUB1.set_xlabel('Fréquence (Hz)')
        SUB1.set_ylabel('Amplitude (Gain)')
        SUB2.plot(k, h)
        SUB2.set_title('Réponse impulsionnelle (h[k]')
        SUB2.set_xlabel('Echantillons')
        SUB2.set_ylabel('Amplitude')
        SUB3.plot(F)
        SUB3.set_title('Filtre Coupe-Bande')
        SUB4.plot(Audio_Basson, color='red', label='Ancien')
        SUB4.plot(Audio_Filtrer3, label='Filtrer')
        SUB4.legend()
        SUB4.set_title('Audio Filtrer')

    return Audio_Filtrer2, Valeur_FFT, Valeur_FFT_Freq

def ifft(Signal_FFT, Maximum, Enveloppe, Longueur):
    Signal_IRFFT = fft.irfft(Signal_FFT, n=Longueur)                                                                    #Creation de irfft avec uniquement les reels

    Signal_IRFFT = Signal_IRFFT * Enveloppe * Maximum * Facteur_De_Grosseur                                             #Ajutement du signal avec le son

    return Signal_IRFFT

def Changer_Son(Frequence_Max, Position_Frequence, Frequence, Signal):
    Frequence_Differentes = [262, 277, 294, 311, 330, 350, 370, 392, 415, 440, 466, 494]                                #Frequence de chaque type
    Valeurs_K = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]                                                                  #Tableau des valeur non changer

    #Trouver c'est quoi la fréquence d'entrée pour ajuster le k
    for i in range(len(Frequence_Differentes)):
        if(Frequence_Differentes[i] > Frequence_Max[0] - 2 and Frequence_Differentes[i] < Frequence_Max[0] + 2):        #Trouver c'est quoi avec +- 2 Hz
            for ii in range(len(Valeurs_K)):
                Valeurs_K[ii] = Valeurs_K[ii] - i                                                                       #Ajuster les Valeur_K en fonction de la fréquence trouver
            break

    #initialisaion des tableau
    DO = np.zeros(len(Signal))
    DO_D = np.zeros(len(Signal))
    RE = np.zeros(len(Signal))
    RE_D = np.zeros(len(Signal))
    MI = np.zeros(len(Signal))
    FA = np.zeros(len(Signal))
    FA_D = np.zeros(len(Signal))
    SOL = np.zeros(len(Signal))
    SOL_D = np.zeros(len(Signal))
    LA = np.zeros(len(Signal))
    LA_D = np.zeros(len(Signal))
    SI = np.zeros(len(Signal))

    #Toute trouver les fréquences et les changer de places
    for i in range(len(Frequence_Max)):
        DO[int(Position_Frequence[i] * 2**(Valeurs_K[0]/12))] = np.abs(Signal[Position_Frequence[i]])
        DO_Freq = Frequence_Max * 2**(Valeurs_K[0]/12)

        DO_D[int(Position_Frequence[i] * 2**(Valeurs_K[1]/12))] = np.abs(Signal[Position_Frequence[i]])
        DO_D_Freq = Frequence_Max * 2 ** (Valeurs_K[1] / 12)

        RE[int(Position_Frequence[i] * 2**(Valeurs_K[2]/12))] = np.abs(Signal[Position_Frequence[i]])
        RE_Freq = Frequence_Max * 2 ** (Valeurs_K[2] / 12)

        RE_D[int(Position_Frequence[i] * 2**(Valeurs_K[3]/12))] = np.abs(Signal[Position_Frequence[i]])
        RE_D_Freq = Frequence_Max * 2 ** (Valeurs_K[3] / 12)

        MI[int(Position_Frequence[i] * 2**(Valeurs_K[4]/12))] = np.abs(Signal[Position_Frequence[i]])
        MI_Freq = Frequence_Max * 2 ** (Valeurs_K[4] / 12)

        FA[int(Position_Frequence[i] * 2**(Valeurs_K[5]/12))] = np.abs(Signal[Position_Frequence[i]])
        FA_Freq = Frequence_Max * 2 ** (Valeurs_K[5] / 12)

        FA_D[int(Position_Frequence[i] * 2**(Valeurs_K[6]/12))] = np.abs(Signal[Position_Frequence[i]])
        FA_D_Freq = Frequence_Max * 2 ** (Valeurs_K[6] / 12)

        SOL[int(Position_Frequence[i] * 2**(Valeurs_K[7]/12))] = np.abs(Signal[Position_Frequence[i]])
        SOL_Freq = Frequence_Max * 2 ** (Valeurs_K[7] / 12)

        SOL_D[int(Position_Frequence[i] * 2**(Valeurs_K[8]/12))] = np.abs(Signal[Position_Frequence[i]])
        SOL_D_Freq = Frequence_Max * 2 ** (Valeurs_K[8] / 12)

        LA[int(Position_Frequence[i] * 2**(Valeurs_K[9]/12))] = np.abs(Signal[Position_Frequence[i]])
        LA_Freq = Frequence_Max * 2 ** (Valeurs_K[9] / 12)

        LA_D[int(Position_Frequence[i] * 2**(Valeurs_K[10]/12))] = np.abs(Signal[Position_Frequence[i]])
        LA_D_Freq = Frequence_Max * 2 ** (Valeurs_K[10] / 12)

        SI[int(Position_Frequence[i] * 2**(Valeurs_K[11]/12))] = np.abs(Signal[Position_Frequence[i]])
        SI_Freq = Frequence_Max * 2 ** (Valeurs_K[11] / 12)

    #Affichage
    if(Print_Tableau_Freq_Notes):
        print('DO Freq', DO_Freq)
        print('DO_D Freq', DO_D_Freq)
        print('RE Freq', RE_Freq)
        print('RE_D Freq', RE_D_Freq)
        print('MI Freq', MI_Freq)
        print('FA Freq', FA_Freq)
        print('FA_D Freq', FA_D_Freq)
        print('SOL Freq', SOL_Freq)
        print('SOL_D Freq', SOL_D_Freq)
        print('LA Freq', LA_Freq)
        print('LA_D Freq', LA_D_Freq)
        print('SI Freq', SI_Freq)

    if (Print_Tableau_Ampliude_Notes):
        print('DO', DO)
        print('DO_D', DO_D)
        print('RE', RE)
        print('RE_D', RE_D)
        print('MI', MI)
        print('FA', FA)
        print('FA_D', FA_D)
        print('SOL', SOL)
        print('SOL_D', SOL_D)
        print('LA', LA)
        print('LA_D', LA_D)
        print('SI', SI)

    if(Afficher_Graphique_Changement_Frequence):
        Figure, SUB = plt.subplots(3, 4)
        SUB[0, 0].plot(Frequence, DO)
        SUB[0, 1].plot(Frequence, DO_D)
        SUB[0, 2].plot(Frequence, RE)
        SUB[0, 3].plot(Frequence, RE_D)
        SUB[1, 0].plot(Frequence, MI)
        SUB[1, 1].plot(Frequence, FA)
        SUB[1, 2].plot(Frequence, FA_D)
        SUB[1, 3].plot(Frequence, SOL)
        SUB[2, 0].plot(Frequence, SOL_D)
        SUB[2, 1].plot(Frequence, LA)
        SUB[2, 2].plot(Frequence, LA_D)
        SUB[2, 3].plot(Frequence, SI)

        SUB[0, 0].set_title('DO')
        SUB[0, 1].set_title('DO_D')
        SUB[0, 2].set_title('RE')
        SUB[0, 3].set_title('RE_D')
        SUB[1, 0].set_title('MI')
        SUB[1, 1].set_title('FA')
        SUB[1, 2].set_title('FA_D')
        SUB[1, 3].set_title('SOL')
        SUB[2, 0].set_title('SOL_D')
        SUB[2, 1].set_title('LA')
        SUB[2, 2].set_title('LA_D')
        SUB[2, 3].set_title('SI')

    return DO, DO_D, RE, RE_D, MI, FA, FA_D, SOL, SOL_D, LA, LA_D, SI

def Full_IFFT(Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur, DO, DO_D, RE, RE_D, MI, FA, FA_D, SOL, SOL_D, LA, LA_D, SI):
    #Toute faire les Sythèse de sans les printer
    Synth_DO = ifft(DO, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_DO_D = ifft(DO_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_RE = ifft(RE, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_RE_D = ifft(RE_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_MI = ifft(MI, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_FA = ifft(FA, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_FA_D = ifft(FA_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_SOL = ifft(SOL, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_SOL_D = ifft(SOL_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_LA = ifft(LA, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_LA_D = ifft(LA_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_SI = ifft(SI, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)

    return Synth_DO, Synth_DO_D, Synth_RE, Synth_RE_D, Synth_MI, Synth_FA, Synth_FA_D, Synth_SOL, Synth_SOL_D, Synth_LA, Synth_LA_D, Synth_SI

def Make_Single_Wave(K, Frequence_Max, Position_Frequence, Signal, Valeur_Max_Guitare, Enveloppe_Temps, Longueur):
    Note = np.zeros(len(Signal))

    for i in range(len(Frequence_Max)):
        Note[int(Position_Frequence[i] * 2**(K/12))] = np.abs(Signal[Position_Frequence[i]])
        Note_Freq = Frequence_Max * 2**(K/12)

    Finale = ifft(Note, Valeur_Max_Guitare, Enveloppe_Temps, Longueur)

    return Finale, Note

def Make_All_Waves(Sample_Rate_Guitare, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur, DO, DO_D, RE, RE_D, MI, FA, FA_D, SOL, SOL_D, LA, LA_D, SI):
    # Synthese du son
    Synth_DO = ifft(DO, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_DO_D = ifft(DO_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_RE = ifft(RE, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_RE_D = ifft(RE_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_MI = ifft(MI, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_FA = ifft(FA, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_FA_D = ifft(FA_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_SOL = ifft(SOL, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_SOL_D = ifft(SOL_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_LA = ifft(LA, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_LA_D = ifft(LA_D, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)
    Synth_SI = ifft(SI, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, Longueur)

    if(Print_Valeurs):
        print('Synth DO', Synth_DO)
        print('Synth DO_D', Synth_DO_D)
        print('Synth RE', Synth_RE)
        print('Synth RE_D', Synth_RE_D)
        print('Synth MI', Synth_MI)
        print('Synth FA', Synth_FA)
        print('Synth FA_D', Synth_FA_D)
        print('Synth SOL', Synth_SOL)
        print('Synth SOL_D', Synth_SOL_D)
        print('Synth LA', Synth_LA)
        print('Synth LA_D', Synth_LA_D)
        print('Synth SI', Synth_SI)

    write("DO.wav", Sample_Rate_Guitare, Synth_DO.astype(np.int16))
    write("DO_D.wav", Sample_Rate_Guitare, Synth_DO_D.astype(np.int16))
    write("RE.wav", Sample_Rate_Guitare, Synth_RE.astype(np.int16))
    write("RE_D.wav", Sample_Rate_Guitare, Synth_RE_D.astype(np.int16))
    write("MI.wav", Sample_Rate_Guitare, Synth_MI.astype(np.int16))
    write("FA.wav", Sample_Rate_Guitare, Synth_FA.astype(np.int16))
    write("FA_D.wav", Sample_Rate_Guitare, Synth_FA_D.astype(np.int16))
    write("SOL.wav", Sample_Rate_Guitare, Synth_SOL.astype(np.int16))
    write("SOL_D.wav", Sample_Rate_Guitare, Synth_SOL_D.astype(np.int16))
    write("LA.wav", Sample_Rate_Guitare, Synth_LA.astype(np.int16))
    write("LA_D.wav", Sample_Rate_Guitare, Synth_LA_D.astype(np.int16))
    write("SI.wav", Sample_Rate_Guitare, Synth_SI.astype(np.int16))

def Write_Single(Nom_Fichier, Sample_Rate, Valeur):
    write(Nom_Fichier, Sample_Rate, Valeur.astype(np.int16))

def Beethoven(SOL, RE_D, FA, RE):
    #Creation de la méliodie
    SOL = np.resize(SOL, int(len(SOL)/12))
    MI = np.resize(RE_D, int(len(RE_D)/4))
    FA = np.resize(FA, int(len(FA)/12))
    RE = np.resize(RE, int(len(RE)/4))

    Melodie = np.concatenate((SOL, SOL, SOL, MI, FA, FA, FA, RE))

    return Melodie

def plot1(X1, Y1, Titre1):
    Figure1, SUB1 = plt.subplots(1, 1)
    SUB1.plot(X1, Y1)
    SUB1.set_title(Titre1)
    #plt.show()

def plot2_4(X1, Y1, X2, Y2, X3, Y3, X4, Y4, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1, Y1)
    SUB1.plot(X2, Y2, color='red')
    SUB1.set_title(Titre1)
    SUB2.plot(X3, Y3)
    SUB2.plot(X4, Y4, color='red')
    SUB2.set_title(Titre2)
    #plt.show()

def plot2(X1, X2, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1)
    SUB1.set_title(Titre1)
    SUB2.plot(X2)
    SUB2.set_title(Titre2)
    #plt.show()

def plot2_2(X1, Y1, X2, Y2, Titre1, Titre2):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1, Y1)
    SUB1.set_title(Titre1)
    SUB2.plot(X2, Y2)
    SUB2.set_title(Titre2)
    #plt.show()

def plot3(X1, Y1, X2, Y2, X3, Y3, X4, Y4, Titre1, Titre2, AxeY, AxeX):
    Figure1, (SUB1, SUB2) = plt.subplots(2, 1)
    SUB1.plot(X1, Y1)
    SUB1.plot(X1[X3], Y3[X3], "X", color='red')
    SUB1.set_title(Titre1)
    SUB1.set_xlabel(AxeX)
    SUB1.set_ylabel(AxeY)
    SUB2.plot(X2, Y2)
    SUB2.plot(X2[X4], Y4[X4], "X", color='red')
    SUB2.set_title(Titre2)
    SUB2.set_xlabel(AxeX)
    SUB2.set_ylabel(AxeY)
    #plt.show()

def main():
    #Lecture des audio et mise en array
    Audio_Guitare, Sample_Rate_Guitare, Valeur_Max_Guitare = LectureToArray("note_guitare_LAd.wav")
    Audio_Basson, Sample_Rate_Basson, Valeur_Max_Basson = LectureToArray("note_basson_plus_sinus_1000_Hz.wav")

    #Redressement des audios
    Audio_Guitare_Redresser = abs(Audio_Guitare)
    Audio_Basson_Redresser = abs(Audio_Basson)

    #Filtres
    Audio_Basson_Filtrer, Valeur_FFT_Basson_Couper, Valeur_FFT_Freq_Couper = Coupe_Bande(Audio_Basson, Valeur_Max_Basson, 20, 1000, Sample_Rate_Basson, 4096)

    # Enveloppe
    Enveloppe_Temps_Guitare = Trouver_Enveloppe(Audio_Guitare, Sample_Rate_Guitare)  # fois 2pi pour le mettre en Hz, puisque le reste est en Hertz
    Enveloppe_Temps_Basson = Trouver_Enveloppe(Audio_Basson_Filtrer, Sample_Rate_Basson)

    #Trouver les 32 frequences
    Frequence_Pos_Guitare, Frequence_Full_Guitare, Signal_FFT_Guitare, Signal_FFT_Not_db_Guitare, Positon_Maximum_Guitare, Frequence_Maximum_Guitare = Trouver32Sinus(Audio_Guitare, Sample_Rate_Guitare)
    Frequence_Pos_Basson, Frequence_Full_Basson, Signal_FFT_Basson, Signal_FFT_Not_db_Basson, Positon_Maximum_Basson, Frequence_Maximum_Basson = Trouver32Sinus(Audio_Basson_Filtrer, Sample_Rate_Basson)

    #Changement de notes
    DO, DO_D, RE, RE_D, MI, FA, FA_D, SOL, SOL_D, LA, LA_D, SI = Changer_Son(Frequence_Maximum_Guitare, Positon_Maximum_Guitare, Frequence_Pos_Guitare, Signal_FFT_Not_db_Guitare)
    Synth_DO, Synth_DO_D, Synth_RE, Synth_RE_D, Synth_MI, Synth_FA, Synth_FA_D, Synth_SOL, Synth_SOL_D, Synth_LA, Synth_LA_D, Synth_SI = Full_IFFT(Valeur_Max_Guitare, Enveloppe_Temps_Guitare, len(Audio_Guitare), DO, DO_D, RE, RE_D, MI, FA, FA_D, SOL, SOL_D, LA, LA_D, SI)

    Synth_Basson, Valeur_Basson = Make_Single_Wave(0, Frequence_Maximum_Basson, Positon_Maximum_Basson, Signal_FFT_Not_db_Basson, Valeur_Max_Basson, Enveloppe_Temps_Basson, len(Audio_Basson))

    #Make_All_Waves(Sample_Rate_Guitare, Valeur_Max_Guitare, Enveloppe_Temps_Guitare, len(Audio_Guitare), DO, DO_D, RE, RE_D, MI, FA, FA_D, SOL, SOL_D, LA, LA_D, SI)

    Beethoven_Wave = Beethoven(Synth_SOL, Synth_RE_D, Synth_FA, Synth_RE)

    #Ajustement
    Audio_Basson_Filtrer = Audio_Basson_Filtrer * 20000
    Synth_Basson = Synth_Basson * 0.43

    Write_Single("Beethoven.wav", Sample_Rate_Guitare, Beethoven_Wave)
    Write_Single("Basson_Filtrer.wav", Sample_Rate_Basson, Audio_Basson_Filtrer)
    Write_Single("Basson_Synthetiser.wav", Sample_Rate_Basson, Synth_Basson)

    #Afficher sur les graphique au besoin
    if(Afficher_Graphique):
        plot2(Audio_Guitare, Audio_Guitare_Redresser, 'Audio Guitare Normal', 'Audio Guitare Redresser')    #Audio de base
        plot2(Audio_Basson, Audio_Basson_Redresser, 'Audio Basson Normal', 'Audio Basson Redresser')        #Audio de base
        plot2_2(Frequence_Pos_Guitare, Signal_FFT_Guitare,  Frequence_Pos_Basson, Signal_FFT_Basson, 'Harmoniques Guitare', 'Harmoniques Basson')  #Harmoniques sans le points
        plot3(Frequence_Pos_Guitare, Signal_FFT_Guitare, Frequence_Pos_Basson, Signal_FFT_Basson, Positon_Maximum_Guitare, Signal_FFT_Guitare, Positon_Maximum_Basson, Signal_FFT_Basson, 'Harmoniques Guitare',  'Harmoniques Basson', 'Signal (dB)', 'Frequence (Hz)',)  #Harmoniques avec les points
        plot1(np.arange(len(Enveloppe_Temps_Guitare)), Enveloppe_Temps_Guitare, 'Enveloppe Guitare')   #Enveloppe seul
        plot1(np.arange(len(Enveloppe_Temps_Basson)), Enveloppe_Temps_Basson, 'Enveloppe Basson')      #Enveloppe seul
        plot2_4(np.arange(len(Audio_Guitare)), Audio_Guitare, np.arange(len(Enveloppe_Temps_Guitare)), Enveloppe_Temps_Guitare, np.arange(len(Audio_Basson)), Audio_Basson, np.arange(len(Enveloppe_Temps_Basson)), Enveloppe_Temps_Basson, 'Guitare', 'Basson' ) #Son avec l'enveloppe par dessus
    plt.show()

if __name__ == '__main__':
    main()

