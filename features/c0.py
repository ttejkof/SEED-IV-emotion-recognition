import numpy as np

def c0(x):
    X = np.fft.fft(x, axis=-1)
    M = np.mean(np.abs(X)**2, axis=-1)
    Y = np.where(X > M, X, 0)
    y = np.fft.ifft(Y, axis=-1)
    A1 = np.sum((x - y)**2, axis=-1)
    A0 = np.sum(x**2, axis=-1)
    return A1 / A0


# TODO: Ovo je samo prepisane formule iz rada mozda ne radi
if __name__ == '__main__':
    x = np.random.rand(10, 10, 10)
    c0(x)