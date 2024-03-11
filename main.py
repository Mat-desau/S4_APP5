#Libraries utiliser
import numpy as np
import wave

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
    

def main():
    Audio_Guitare = LectureToArray("note_guitare_LAd.wav")
    Audio_Basson = LectureToArray("note_basson_plus_sinus_1000_Hz.wav")




if __name__ == '__main__':
    main()

