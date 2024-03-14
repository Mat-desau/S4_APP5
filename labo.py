import numpy, pylab, scipy.io.wavfile, scipy.signal

def x1(n):
    return numpy.sin(0.1*numpy.pi*n + numpy.pi/4)

def x2(n):
    return (-1)**n

def x3(n):
    return (n==10).astype(float)

def x4(n):
    f1 = 200.0/16000
    f2 = 3000.0/16000
    A1 = 1.0
    A2 = 0.25
    return A1*numpy.sin(2*numpy.pi*f1*n) + A2*numpy.sin(2*numpy.pi*f2*n)

def plotfft(signal,n, window, normfreq=False):
    """
    Tracer la fft de la fonction signal
    signal: fonction qui accepter un array d'entiers comme argument
    n: array de points auxquelles signal sera evalue
    window: fenetre
    normfreq: normaliser ren rad/ech si vrai
    """
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(n))
    if normfreq:
        freq *= 2*numpy.pi
    else:
        freq *= n
    fft_input = signal(numpy.arange(n))*window(n)
    spectrum = numpy.fft.fftshift(numpy.fft.fft(fft_input))
    ax_s = pylab.subplot(311)
    ax_s.stem(numpy.arange(n),signal(numpy.arange(n)))
    ax_a = pylab.subplot(312)
    ax_a.stem(freq, abs(spectrum))
    ax_p = pylab.subplot(313,sharex=ax_a)
    ax_p.stem(freq, numpy.angle(spectrum))
    ax_s.set_ylabel("signal")
    ax_s.set_xlabel("n")
    ax_a.set_ylabel("amplitude")
    ax_p.set_ylabel("phase")
    pylab.tight_layout()
    if normfreq:
        ax_a.set_xlabel("omega")
        ax_p.set_xlabel("omega")
    else:
        ax_a.set_xlabel("m")
        ax_p.set_xlabel("m")

def plot_fir(coeffs, axes, lt, label, unwrap=True, freqnorm=1.0):
    if unwrap:
        unwrap = numpy.unwrap
    else:
        unwrap = lambda x: x
        
    
    t = numpy.arange(coeffs.size)
    axes[0].plot(t, coeffs, lt)
    axes[0].plot(t, coeffs, lt[0]+'o', label=label)
    fft_taps = numpy.fft.fftshift(numpy.fft.fft(coeffs))
    fft_1024 = numpy.fft.fftshift(numpy.fft.fft(coeffs, n=1024))
    f_taps = freqnorm * numpy.fft.fftshift(numpy.fft.fftfreq(coeffs.size))
    f_1024 = freqnorm * numpy.fft.fftshift(numpy.fft.fftfreq(1024))

    axes[1].plot(f_1024, numpy.abs(fft_1024), lt)
    axes[1].plot(f_taps, numpy.abs(fft_taps), lt[0] + 'o', label=label)
    axes[2].plot(f_1024, unwrap(numpy.angle(fft_1024)), lt)
    axes[2].plot(f_taps, unwrap(numpy.angle(fft_taps)), lt[0] + 'o', label=label)

def normwin(coeffs, window):
    result = coeffs * window
    result /= numpy.sum(result)
    return result

def fir_win_comp(N, bw, window='rectangular', unwrap=True, freqnorm=2*numpy.pi):
    """
    Compare differentes manieres de calculer des filtres FIR passe bas par la methode des fenetres.
    :param N: int, nombre de N
    :param bw: float, bande passante (fs/Fs, devrait etre < 0.5)
    :param n: nombre de points de la FFT pour l'analyse frequentielle
    :return:
    """
    bw = 2.0 * bw + 1.0/N
    
    k = numpy.arange(N)
    k_sym = numpy.linspace(-(N-1)/2,(N-1)/2, N)
    # calcul analytique
    # avec repliement
    analytique_fin = 1.0/N * numpy.sin(numpy.pi*k_sym*bw) / numpy.sin(numpy.pi*k_sym / N)

    # sans repliement
    analytique_inf = numpy.sin(numpy.pi * k_sym * bw)/numpy.pi/k_sym

    # lever divisions par 0 pour k == 0
    zero, = numpy.nonzero(k_sym == 0)
    if len(zero):
        analytique_fin[zero] = bw
        analytique_inf[zero] = bw

    # noramliser le gain DC a 1
    analytique_inf /= numpy.sum(analytique_inf)


    dft = numpy.zeros(N)
    maxbin = int(round((bw * N-1.0)/2))
    # K dans UDSP est 2*maxbin + 1
    print("k = %d" % (2*maxbin+1))
    dft[:maxbin + 1] = 1
    dft[-maxbin:] = 1
    dft = numpy.fft.fftshift(numpy.fft.ifft(dft))

    ax = [pylab.subplot(3,1,i) for i in range(1,4)]

    # appliquer fenetre (symmetrique)
    w = scipy.signal.get_window(window, N, False)
    

    scipyfirwin = scipy.signal.firwin(N, bw, window=window)
    plot_fir(normwin(analytique_fin,w), ax, 'r-', label='analytique (avec repliement)', unwrap=unwrap, freqnorm=freqnorm)
    plot_fir(normwin(analytique_inf,w), ax, 'g-', label=r'analytique (sans repliement)', unwrap=unwrap, freqnorm=freqnorm)
    plot_fir(scipyfirwin, ax, 'k:', label='scipy.firwin', unwrap=unwrap, freqnorm=freqnorm)
    plot_fir(normwin(dft,w), ax, 'b:', label='dft', unwrap=unwrap, freqnorm=freqnorm)
    # nous enlevons le dernier point pour rendre le filtre symmetrique

    if freqnorm == 1:
        freqlabel = r"$f/F_s$"
    elif freqnorm == 2*numpy.pi:
        freqlabel = r"$f$ (rad/éch)"
    else:
        freqlabel = r"$f$ (Hz)"
    


    ax[0].set_title(r'coeffcients (réponse impulsionnelle)')
    ax[1].set_title(r'réponse frequencielle')
    ax[0].set_xlabel(r'$n$')
    ax[0].set_ylabel(r'$h$')
    ax[1].set_xlabel(freqlabel)
    ax[1].set_ylabel(r'$|H|$')
    ax[2].set_ylabel(r'$\angle(H)$')
    ax[2].set_xlabel(freqlabel)
    ax[0].legend()
    return normwin(analytique_fin,w)

def apply_fir_t(fir, signal):
    pylab.plot(fir)
    pylab.plot(signal)
    pylab.plot(numpy.convolve(signal,fir))

def apply_fir_f(fir, signal, n, fs=16000.0):
    fir_f = numpy.fft.fft(fir,n)
    signal_f = numpy.fft.fft(signal,n)
    resultat_f = signal_f*fir_f
    freq = fs * numpy.fft.fftshift(numpy.fft.fftfreq(n))
    ax = pylab.subplot(211)
    ax.plot(freq,abs(numpy.fft.fftshift(resultat_f)))
    ax.set_xlabel('f')
    ax = pylab.subplot(212)
    resultat_t = numpy.fft.ifft(resultat_f,n)
    ax.plot(resultat_t.real,label='re')
    ax.plot(resultat_t.imag,label='im')
    ax.legend()
    ax.set_xlabel('t')

def exercice1(N, normfreq=True):
    funcs = {'x1': x1, 'x2': x2, 'x3': x3}
    windows = {'rectangle': numpy.ones, 'Hanning': scipy.signal.windows.hann}
    for window in windows:
        for func in funcs:
            pylab.figure('%s pour N=%d ech, fenêtre %s' % (func, N, window));
            plotfft(funcs[func], N, windows[window], normfreq=normfreq)

def exercice2():
    results = {}
    signal = x4(numpy.arange(128))

    for n in (16,32,64):
        results[n] = {}
        for window in ['rectangular','hamming']:
            pylab.figure('%s avec %d N' % (window, n))
            results[n][window] = fir_win_comp(n,2*2.0/16,window)
    for window in results[64]:
        fir = results[64][window]
        pylab.figure(window+' domaine temporel')
        apply_fir_t(fir, signal)
        pylab.figure(window+' domaine frequentiel')
        apply_fir_f(fir, signal,256)

if __name__ == "__main__":
    #exercice1(64);
    exercice2();
    pylab.show()
    



