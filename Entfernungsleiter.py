import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

### Welche Korrekturterme sollen verwendet werden? ###

ExtKorr = 'ja' #['ja', 'nein'] ### Korrektur für Interstellare Extinktion ####
FormKorr = 'ja' #['ja', 'nein'] ### Korrektur für die Form von SN Ia-Lichtkurven ###
KKorr = 'ja' #['ja', 'nein'] ### rotverschiebungsabhängige K-Korrektur ###


# %% Cepheiden Kallibration
##############################################################################
#-----------------------------------------------------------------------------
#-------------------Perioden-Leuchtkraft-Beziehung----------------------------
#-----------------------------------------------------------------------------
##############################################################################

### Daten Milchstraßen-Cepheiden aus Riess et al. (2021) ###
# URL: http://dx.doi.org/10.3847/2041-8213/abdbaf
data = pd.read_csv('MW_Cepheids_Riess2021.csv')
data.columns

### Helligkeiten in verschiedenen Lichtbändern ###
Vmag = data['F555W']
Imag = data['F814W']
Hmag = data['F160W']

### Logarithmus der Periode ###
logP = data['logP']
elogP = 0.00016*logP

### Parallaxe ###
para = data['para']

### Entfernung ###
d=1000/para

### Entfernungsmodul ###
mu = 5 * np.log10(d) - 5

### Extinktionskorrektur ###
if ExtKorr == 'ja': ### Wesenheit-Magnitude für Cepheiden ###
    M = Hmag - 0.386*(Vmag - Imag) - mu
elif ExtKorr == 'nein':
    M = Hmag - mu


### Fit-Funktion PL-Beziehung ###
def fitFunc(x,a,b):
    return a*(x-1)+b

popt, pcov = curve_fit(fitFunc, logP, M)

perr = np.sqrt(np.diag(pcov))

y = popt[0]*(logP-1)+popt[1]

#Parameter der PL-Beziehung
a=popt[0]
ea=perr[0]

b=popt[1]
eb=perr[1]

### Standardabweichung aus Residuen ###
resid = (M -(a*(logP-1)+b))
sigmaCeph = np.std(resid)

### Plot darstellen ###
plt.figure(1)
plt.title('PL-Beziehung')
plt.gca().invert_yaxis()
plt.scatter(logP, M, s = 0.5, c='k')
plt.plot(logP, y, c='r')

plt.ylim(0, -10)
plt.xlim(-0.25, 2)

plt.xlabel('log$_{10}(P)$')
plt.ylabel('$M_H^W$ [mag]')


#### Standardabweichung darstellen ###
x = np.linspace(min(logP), max(logP), 50)

#1sigma
plt.fill_between(x,a*(x-1)+b +sigmaCeph, a*(x-1)+b -sigmaCeph,linewidth = 0.8,alpha=0.5, color='r')
#2sigma unten
plt.fill_between(x,a*(x-1)+b -sigmaCeph, a*(x-1)+b -2*sigmaCeph,linewidth = 0.5,alpha=0.3, color='r')
#2sigma oben
plt.fill_between(x,a*(x-1)+b +sigmaCeph, a*(x-1)+b +2*sigmaCeph,linewidth = 0.5,alpha=0.3, color='r')
#3sigma unten
plt.fill_between(x,a*(x-1)+b -2*sigmaCeph, a*(x-1)+b -3*sigmaCeph,linewidth = 0.5,alpha=0.15, color='r')
#3sigma oben
plt.fill_between(x,a*(x-1)+b +2*sigmaCeph, a*(x-1)+b +3*sigmaCeph,linewidth = 0.5,alpha=0.15, color='r')

print('PL-Beziehung: \n' + 'a = ', round(a,2), '+-', round(ea,2),'\n' + 'b = ', round(b,2), '+-', round(eb,2), '\n' + 'sigma = ', round(sigmaCeph,2))


# %%SNIa Kallibration
##############################################################################
#-----------------------------------------------------------------------------
#------------Kalibration der absoluten Helligkeit von SN Ia-------------------
#-----------------------------------------------------------------------------
##############################################################################

#----Bestimmung des Entfernungsmoduls zu Galaxien mit Cepheiden und SN Ia-----

### Daten Extragalaktischer Cepheiden aus Riess et al. (2016) ###
# URL: http://dx.doi.org/10.3847/0004-637X/826/1/56
data = pd.read_csv('Extragalactic_Cepheids_HBand.csv', sep= "\t",encoding='latin1')
data.columns

### Helligkeiten in verschiedenen Lichtbändern ###
VImag = data['V-I']
Hmag = data['F160W']

### Logarithmus der Periode ###
P = data['Per']
logP = np.log10(P)

### Absolute Helligkeit der Cepheiden auf Basis der PL-Beziehung
M = a*(logP - 1) + b 

### Extinktionskorrektur und Entfernungsmodul zu einzelnen Cepheiden ###
if ExtKorr == 'ja': ### Wesenheit-Magnitude für Cepheiden ###
    mu = Hmag - 0.386*VImag - M 
if ExtKorr == 'nein':
    mu = Hmag - M 

### Mittelwert der Entfernungsmodule für die einzelnen Galaxien
data['mu'] = mu
muCeph = (data.groupby(['Gal'], as_index=False).mean().groupby('Gal')['mu'].mean())

#---------------------------Helligkeit der SN Ia------------------------------

### Daten der Supernovae aus Burns et al. (2018) ###
# URL: http://dx.doi.org/10.3847/1538-4357/aae51c
SNdata = pd.read_csv('SNIa_Kalibration_Carnegie.csv', sep= "\t",encoding='latin1')

### Helligkeit im V-Band ###
SNVmag = SNdata['Vmax']

### Extinktionsparameter ###
EBV = SNdata['EBV']
Rv = SNdata['Rv']

### Rotverschiebung ###
zcmb = SNdata['zcmb']

### Form Faktor ###
decline = SNdata['dm15(B)']

### Extinktionskorrektur ###
if ExtKorr == 'ja':
    SNm = SNVmag - Rv*EBV
if ExtKorr == 'nein':
    SNm = SNVmag

### Korrektur der Lichtkurven-Form ###
if FormKorr == 'ja':
    SNm = SNm - 0.672*(decline - 1.0027) + 0.633*(decline - 1.0027)**2 

### K-Korrektur ###
if KKorr == 'ja': #Daten sind bereits K-korrigiert
    SNm = SNm
elif KKorr == 'nein': #Rückgängigmachen der K-Korrektur (Annahme)
    SNm = SNm - 2.5 *np.log10(1+zcmb)

### Fit-Funktion SN Ia-Helligkeit ###
def fitFunc(x,a):
    return x + a

popt, pcov = curve_fit(fitFunc, muCeph, SNm)

perr = np.sqrt(np.diag(pcov))

### Absolute SN Ia-Helligkeit ###
MEr = popt[0]
eMEr = perr[0]

### theoretische scheinbare Helligkeit der einzelnen SN Ia ###
mEr = muCeph + MEr
mEr.reset_index(drop=True, inplace=True)

### Standardabweichung aus Residuen ###
resid = (SNm - mEr)
sigmaSNM = np.std(resid)

### Plot darstellen ###
plt.figure(2)
plt.title('Absolute SN Ia-Helligkeit')
plt.scatter(muCeph, SNm, c='k', zorder = 10)
plt.plot(muCeph, mEr, c = 'r', zorder = 2)

plt.xlim(min(muCeph)-0.2, max(muCeph)+0.2)
plt.ylim(min(SNm)-1, max(SNm)+1)
plt.xlabel('$\mu_{Ceph}$')
plt.ylabel('$m_V^{SNIa}$ [mag]')

#-------------------Standardabweichung darstellen-----------------------------
x = np.linspace(min(muCeph), max(muCeph), 20)
y1o = x+MEr +sigmaSNM
y1u = x+MEr -sigmaSNM
y2o = x+MEr +2*sigmaSNM
y2u = x+MEr -2*sigmaSNM
y3o = x+MEr +3*sigmaSNM
y3u = x+MEr -3*sigmaSNM    

plt.fill_between(x, y3u,y2u, linewidth = 0.5,alpha=0.15, color='r')
plt.fill_between(x, y2u,y1u, linewidth = 0.5,alpha=0.3, color='r')
plt.fill_between(x, y1u,y1o, linewidth = 0.8,alpha=0.5, color='r')
plt.fill_between(x, y1o,y2o, linewidth = 0.5,alpha=0.3, color='r')
plt.fill_between(x, y2o,y3o, linewidth = 0.5,alpha=0.15, color='r')

print()
print('SN Ia-Kalibration: \n' + 'M = ', round(MEr,2), '+-', round(eMEr,2), '\n' + 'sigma = ', round(sigmaSNM,2))

# %% Hubble-Diagramm
##############################################################################
#-----------------------------------------------------------------------------
#---------HUBBLE-DIAGRAMM: SUPERNOVAE Ia gegen ROTVERSCHIEBUNG (z)------------
#-----------------------------------------------------------------------------
##############################################################################

### Daten der Supernovae aus Sako et al. (2018) ###
# Sloan Digital Sky Survey (SDSS)
# URL: http://dx.doi.org/10.1088/1538-3873/aab4e0
SN = pd.read_csv('SDSS_SNIa.csv', sep = '\t', delimiter=None)
SN.columns

### scheinbare Helligkeit im R-Band ###
m = SN['Peakrmag']

### Rotverschiebung ###
z = pd.to_numeric(SN['zCMB'])

### Form Faktor ###
decline = SN['dm15z']

### Extinktion ###
AV = SN['AVM']

### Korrektur der Lichtkurven-Form ###
if FormKorr == 'ja':
    m = m - 0.672*(decline - 1.0027) + 0.633*(decline - 1.0027)**2

### Bestimmung des Entfernungsmoduls auf Basis der zuvor ermittelten absoluten Helligkeit ###
### inklusive Extinktionskorrektur ###
if ExtKorr == 'ja': 
    mu = m - (MEr + 0.01) - AV 
if ExtKorr == 'nein':
    mu = m - (MEr + 0.01)

### K-Korrektur ###
if KKorr == 'ja':
    mu = mu + 2.5 * np.log10(1+z)


### Fit-Funktion des Hubble-Plots ###
c = 299792.458 #Lichtgeschwindigkeit
q0 = (-0.55) #Abbremsfaktor des Universums
j = 1 #Ruckfaktor des Universums

def fitFunc(x,a):
    return 5 *np.log10((c *x /a) *(1 + 1/2 *(1 - q0 ) *x - 1/6 *(1 - q0 - 3*q0**2 + j) *x **2)) + 25

popt, pcov = curve_fit(fitFunc, z, mu)
perr = np.sqrt(np.diag(pcov))

### Hubble-Konstante ###
H0 = popt[0]
eH0 = perr[0]

### Standardabweichung aus Residuen ###
resid = H0 - 1/(10**((mu - 25)/5) /(c*z) /(1 + 1/2 *(1 - q0 ) *z - 1/6 *(1 - q0 - 3*q0**2 + j) *z **2))
sigmaH0 = np.std(resid)


### Plot darstellen ###
plt.figure(3)
plt.title('Hubble-Plot (SDSS)')
maximal = max(z)
minimal = min(z)
x = np.linspace(minimal, maximal, num = 100)
y = 5 *np.log10((c *x /a) *(1 + 1/2 *(1 - q0 ) *x - 1/6 *(1 - q0 - 3*q0**2 + j) *x **2)) + 25

plt.plot(x, y, c='r')
plt.scatter(z, mu, s=0.5, c='k')
plt.xlim(minimal - 0.01*maximal, maximal + 0.01*maximal)

y1o=fitFunc(x,H0+sigmaH0)
y1u=fitFunc(x,H0-sigmaH0)
y2o=fitFunc(x,H0+2*sigmaH0)
y2u=fitFunc(x,H0-2*sigmaH0)
y3o=fitFunc(x,H0+3*sigmaH0)
y3u=fitFunc(x,H0-3*sigmaH0)

plt.fill_between(x, y3u,y2u, linewidth = 0.5,alpha=0.15, color='r')
plt.fill_between(x, y2u,y1u, linewidth = 0.5,alpha=0.3, color='r')
plt.fill_between(x, y1u,y1o, linewidth = 0.8,alpha=0.5, color='r')
plt.fill_between(x, y1o,y2o, linewidth = 0.5,alpha=0.3, color='r')
plt.fill_between(x, y2o,y3o, linewidth = 0.5,alpha=0.15, color='r')

plt.xlabel('$z$')
plt.ylabel('$\mu$')

print()
print('Hubble-Konstante (SDSS): \n' + 'H0 = ', round(H0,2), '+-', round(eH0,2), '\n' + 'sigma = ', round(sigmaH0,2))

#--------Hubble-Plot mit bereits vollständig korrigierten Daten---------------

### Daten der Supernovae aus Scolnic et al. (2018) ###
# Panoramic Survey Telescope And Rapid Response System (Pan-STARRS)
# URL: https://doi.org/10.3847/1538-4357/aab9bb
SN = pd.read_csv('PS_SNI_Scolnic.csv', sep='\s')
SN.columns
SN = SN.loc[SN['mb']>0]
m = SN['mb']
z = SN['zcmb']
mu = m - MEr

### Fit-Funktion des Hubble-Plots ###
c = 299792.458 #Lichtgeschwindigkeit
q0 = (-0.55) #Abbremsfaktor des Universums
j = 1 #Ruckfaktor des Universums

### Fit-Funktion ###
def fitFunc(x,a):
    return 5 *np.log10((c *x /a) *(1 + 1/2 *(1 - q0 ) *x - 1/6 *(1 - q0 - 3*q0**2 + j) *x **2)) + 25

popt, pcov = curve_fit(fitFunc, z, mu)
perr = np.sqrt(np.diag(pcov))

### Hubble-Konstante ###
H0 = popt[0]
eH0 = perr[0]

### Standardabweichung aus Residuen ###
resid = H0 - 1/(10**((mu - 25)/5) /(c*z) /(1 + 1/2 *(1 - q0 ) *z - 1/6 *(1 - q0 - 3*q0**2 + j) *z **2))
sigmaH0 = np.std(resid)


### Plot darstellen ###
plt.figure(4)
plt.title('Hubble-Plot (Pan-STARRS)')
maximal = max(z)
minimal = min(z)
x = np.linspace(minimal, maximal, num = 100)
y = 5 *np.log10((c *x /a) *(1 + 1/2 *(1 - q0 ) *x - 1/6 *(1 - q0 - 3*q0**2 + j) *x **2)) + 25

plt.plot(x, y, c='r')
plt.scatter(z, mu, s=0.5, c='k')
plt.xlim(minimal - 0.01*maximal, maximal + 0.01*maximal)

y1o=fitFunc(x,H0+sigmaH0)
y1u=fitFunc(x,H0-sigmaH0)
y2o=fitFunc(x,H0+2*sigmaH0)
y2u=fitFunc(x,H0-2*sigmaH0)
y3o=fitFunc(x,H0+3*sigmaH0)
y3u=fitFunc(x,H0-3*sigmaH0)

plt.fill_between(x, y3u,y2u, linewidth = 0.5,alpha=0.15, color='r')
plt.fill_between(x, y2u,y1u, linewidth = 0.5,alpha=0.3, color='r')
plt.fill_between(x, y1u,y1o, linewidth = 0.8,alpha=0.5, color='r')
plt.fill_between(x, y1o,y2o, linewidth = 0.5,alpha=0.3, color='r')
plt.fill_between(x, y2o,y3o, linewidth = 0.5,alpha=0.15, color='r')

plt.xlabel('$z$')
plt.ylabel('$\mu$')

print()
print('Hubble-Konstante (Pan-STARRS): \n' + 'H0 = ', round(H0,2), '+-', round(eH0,2), '\n' + 'sigma = ', round(sigmaH0,2))






