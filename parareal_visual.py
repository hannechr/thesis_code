import numpy as np
import matplotlib.pyplot as plt

def fun(t, y):
    return (-(t)**2 + 5)/2

def fwe(t, y, f, dt):
    return y + dt*f(t, y)

init = 0.1
tend = 5
dtslice = 1
nslices = int(tend/dtslice)
ncoarse = 1
dtcoarse = dtslice/ncoarse
nfine = 50
dtfine = dtslice/nfine
tfine = np.linspace(0, tend, nslices*nfine + 1)
tcoarse = np.linspace(0, tend, nslices*ncoarse + 1)
tslice = np.linspace(0, tend, nslices + 1)

coarseall = np.zeros(nslices*ncoarse +1)
coarseall[0] = init
fineall = np.zeros(nslices*nfine +1)
fineall[0] = init

for p in range(nslices):
    t = p*dtslice
    for n in range(1,ncoarse+1):
        coarseall[p*ncoarse + n] = fwe(t, coarseall[p*ncoarse + n-1], fun, dtcoarse)
        t += dtcoarse
    t = p*dtslice
    for n in range(1,nfine+1):
        fineall[p*nfine + n] = fwe(t, fineall[p*nfine + n-1], fun, dtfine)
        t += dtfine

coarse = coarseall[np.arange(0,nslices*ncoarse +1, ncoarse)]
fine = fineall[np.arange(0,nslices*nfine +1, nfine)]

c = np.zeros((nslices, ncoarse+1))
f = np.zeros((nslices, nfine+1))

for p in range(nslices):
    init = coarse[p]
    c[p, 0] = init
    f[p, 0] = init
    t = p*dtslice
    for n in range(1,ncoarse+1):
        c[p,n] = fwe(t, c[p,n-1], fun, dtcoarse)
        t += dtcoarse
    t = p*dtslice
    for n in range(1,nfine+1):
        f[p,n] = fwe(t, f[p,n-1], fun, dtfine)
        t += dtfine

limmax = max(max(coarseall), max(fineall))
limmin = min(min(coarseall), min(fineall))

plt.figure()

for p in range(nslices+1):
    plt.plot([p*dtslice, p*dtslice], [limmin*1.1,limmax*1.1], 'k--', alpha=0.5)

plt.plot(tfine, fineall, 'b')
plt.plot(tslice, fine, 'bs')
plt.plot(tcoarse, coarseall, 'r')
plt.plot(tslice, coarse, 'rs')

for p in range(nslices):
    plt.plot(tfine[p*nfine:(p+1)*nfine+1], f[p,:], color='gold')

plt.show()