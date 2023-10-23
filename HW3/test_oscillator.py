import numpy
import scipy.optimize

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp, 2.*numpy.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * numpy.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}

# Example usage:
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate or load your data (replace with your data)
    time = numpy.linspace(0, 1, 1000)  # Time values
    data = 2.0 * numpy.sin(2 * numpy.pi * 1.5 * time + numpy.pi/4) + 0.5 * numpy.random.randn(1000)  # Simulated data

    # Call the fit_sin function
    results = fit_sin(time, data)

    # Extract the fitted parameters
    amplitude = results['amp']
    frequency = results['freq']
    phase_shift = results['phase']
    offset = results['offset']

    print("Amplitude:", amplitude)
    print("Frequency:", frequency)
    print("Phase Shift:", phase_shift)
    print("Offset:", offset)

    # Plot the data and the fitted sinusoidal function
    plt.plot(time, data, label='Data')
    plt.plot(time, results['fitfunc'](time), label='Fitted Sinusoid', linestyle='--')
    plt.legend()
    plt.show()
