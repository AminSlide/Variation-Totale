# Amin 
#18/03/24

import numpy as np
from matplotlib import pyplot as plt
from operators import *

import cv2

I =cv2.imread('statue.png',0)
plt.figure()
plt.subplot(131)
plt.imshow(I)
plt.title("Image initiale")
plt.axis('off')

N = 9 #dimension

h = generatePSF('2D','gaussian',N)
                
def H(x):
    return A(x,h)

def Ht(x):
    return At(x,h)


bruit = np.random.standard_normal((I.shape))
z = H(I) + bruit
plt.subplot(132)
plt.imshow(z)
plt.title("Image floutée et bruitée")
plt.axis('off')

def Norme_1(x):
    return sum(abs(x))
def Norme_2(x):
    return np.sqrt(np.dot(np.transpose(x),x))



def Gradient_1(x,z,yk,mu):
    return 2*Ht(H(x) - z) - mu*Dt(yk-D(x))


def Gradient_2(y,xkplus1,mu) :
    return mu*(y-D(xkplus1)) 


def prox_g(y,coeff,gamma):
    xGamma = np.ones(y.shape)*gamma*coeff
    return np.maximum(np.minimum(np.zeros(y.shape),y+xGamma),y-xGamma)

# paramètres de l'algo
Niter = 200  # nombre max d'itérations
coeffLambda = 1000
mu = 1000

# initialisation des variables

#def des pas
a = 1/2

LDf2 = mu
gamma2 = a/(LDf2)	

LDf1 = 2*opNorm(H,Ht,'2D')**2 + mu*opNorm(D,Dt,'2D')**2
gamma1 = a/LDf1		

# Valeurs initiales
y0 = np.ones((2,I.shape[0],I.shape[1]))
x0 = np.ones(I.shape)*20 

X = [x0]
Y = [y0]

for k in range(Niter):

    # Calcul de xk+1
        X.append(X[k] - gamma1*Gradient_1(X[k],z,Y[k],mu))

    # Calcul de yk+1
        Y.append(Y[k] - gamma2*Gradient_2(Y[k],X[k+1],mu))
        Y[k+1] = prox_g(Y[k+1],coeffLambda,gamma2)
 

Iapprox = X[-1]

plt.subplot(133)
plt.imshow(Iapprox)
plt.title("Image défloutée")
plt.axis('off')


import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(image1, image2):
    """
    Calcule l'erreur Mean Squared Error (MSE) entre deux images.
    
    Args:
        image1 (numpy.ndarray): Première image.
        image2 (numpy.ndarray): Deuxième image.
        
    Returns:
        float: Valeur de MSE.
    """
    return np.mean((image1 - image2) ** 2)

def snr(image, reference):
    """
    Calcule le Signal to Noise Ratio (SNR) entre une image et une référence.
    
    Args:
        image (numpy.ndarray): Image dont le SNR est calculé.
        reference (numpy.ndarray): Image de référence.
        
    Returns:
        float: Valeur de SNR en dB.
    """
    mse_value = mse(image, reference)
    power_signal = np.sum(reference ** 2)
    snr_value = 10 * np.log10(power_signal / mse_value)
    return snr_value

def ssim_score(image, reference, data_range=256):
    """
    Calcule le Structural SIMilarity (SSIM) entre une image et une référence.
    
    Args:
        image (numpy.ndarray): Image dont le SSIM est calculé.
        reference (numpy.ndarray): Image de référence.
        data_range (float or None, optional): Plage de valeurs possibles pour les données. Si None, la plage de valeurs est déterminée par le type de données de l'image. Default is None.
        
    Returns:
        float: Valeur de SSIM.
    """
    return ssim(image, reference, data_range=data_range, multichannel=True)


# Exemple d'utilisation :
# Supposons que vous avez deux images img1 et img2
# où img1 est l'image de référence et img2 est l'image à comparer
# Vous pouvez les comparer comme suit :

# mse_value = mse(img1, img2)
# snr_value = snr(img2, img1)
# ssim_value = ssim_score(img2, img1)
# Amin et Livio
#18/03/24

import numpy as np
from matplotlib import pyplot as plt
from operators import *

import cv2

I =cv2.imread('statue.png',0)

N = 9 #dimension

h = generatePSF('2D','gaussian',N)
                
def H(x):
    return A(x,h)

def Ht(x):
    return At(x,h)


bruit = np.random.standard_normal((I.shape))
z = H(I) + bruit

def Norme_1(x):
    return sum(abs(x))
def Norme_2(x):
    return np.sqrt(np.dot(np.transpose(x),x))



def Gradient_1(x,z,yk,mu):
    return 2*Ht(H(x) - z) - mu*Dt(yk-D(x))


def Gradient_2(y,xkplus1,mu) :
    return mu*(y-D(xkplus1)) 


def prox_g(y,coeff,gamma):
    xGamma = np.ones(y.shape)*gamma*coeff
    return np.maximum(np.minimum(np.zeros(y.shape),y+xGamma),y-xGamma)

#définition des fonctions de coût
def g(y) :
    return 
def f(y) :
    return 

def TV(coeffLambda,mu,image) : 
    I =cv2.imread(image,0)

    N = 9 #dimension

    h = generatePSF('2D','gaussian',N)
                    
    def H(x):
        return A(x,h)

    def Ht(x):
        return At(x,h)


    bruit = np.random.standard_normal((I.shape))
    z = H(I) + bruit

    Niter = 20  # nombre max d'itérations
    # initialisation des variables

    #def des pas
    a = 2/4

    LDf2 = mu
    gamma2 = a/(LDf2)	

    LDf1 = 2*opNorm(H,Ht,'2D')**2 + mu*opNorm(D,Dt,'2D')**2
    gamma1 = a/LDf1		

    # Valeurs initiales
    y0 = np.ones((2,I.shape[0],I.shape[1]))
    x0 = np.ones(I.shape)*20 

    X = [x0]
    Y = [y0]

    for k in range(Niter):

        # Calcul de xk+1
            X.append(X[k] - gamma1*Gradient_1(X[k],z,Y[k],mu))

        # Calcul de yk+1
            Y.append(Y[k] - gamma2*Gradient_2(Y[k],X[k+1],mu))
            Y[k+1] = prox_g(Y[k+1],coeffLambda,gamma2)
    

    Iapprox = X[-1]
    return [I,Iapprox]

K = [0.5*k for k in range(1,201)]
mu = 1000
E1 = []
E2 = []
E3 = []
for k in range(len(K)) : 
    try : 
        I,Iapprox = TV(K[k],mu,"statue.png")
        E1.append(mse(I,Iapprox))
        E2.append(snr(Iapprox,I))
        E3.append(ssim_score(Iapprox,I))
    except : 
        K.remove(0.5*(k+1))
plt.figure()
plt.subplot(131)
plt.plot(K,E1)
plt.title("MSE")
plt.tick_params(axis='x', labelsize=5)
plt.tick_params(axis='y', labelsize=5)
plt.xlabel('λ')
plt.subplot(132)
plt.plot(K,E2)
plt.title("SNR")
plt.tick_params(axis='x', labelsize=5)
plt.tick_params(axis='y', labelsize=5)
plt.xlabel('λ')
plt.subplot(133)
plt.title("SSIM")
plt.tick_params(axis='x', labelsize=5)
plt.tick_params(axis='y', labelsize=5)
plt.xlabel('λ')

plt.plot(K,E3)
