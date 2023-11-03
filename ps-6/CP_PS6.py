"""
Name: CP_PS6.py
Author: Sandhya Sharma
Date: November 3, 2023
Description: Principal Component Analysis (PCA) of a set of galaxy spectra (flux). 

"""
import astropy.io.fits    
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit

filename = 'specgrid.fits'
curr_path = os.getcwd() + '/' + filename 

# A. READING IN THE DATA AND PLOTTING THE SPECTRUM
hdu_list = astropy.io.fits.open(curr_path)
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

plt.plot(logwave, flux[0,:], label = 'Galaxy 1')
plt.plot(logwave, flux[1,:], label = 'Galaxy 2')
plt.plot(logwave, flux[2,:], label = 'Galaxy 3')
plt.plot(logwave, flux[3,:], label = 'Galaxy 4')
plt.plot(logwave, flux[4,:], label = 'Galaxy 5')
plt.legend()
plt.title('Flux Spectrum of the First Five Galaxies')
plt.xlabel('Log Wavelength (log(Angstrom))')
plt.ylabel('Flux (10^-17 erg/(s*cm^2*Angstrom)')
plt.show()

# B. NORMALIZING  
flux_sum = flux.sum(axis=1)
flux_norm = flux/flux_sum[:,None]

# C. SUBTRACTING THE MEAN
flux_mean = np.mean(flux_norm, axis=1)
flux_without_offset = flux_norm - flux_mean[:,None]

# D. CALCULATING THE EIGENVECTORS OF THE COVARIANCE MATRIX
start1 = timeit.default_timer()
r = flux_without_offset
c = np.dot(r.T, r)
eig_val_cov, eig_vec_cov = np.linalg.eig(c)
stop1 = timeit.default_timer()

for i in range(0,5):
    plt.plot(logwave, eig_vec_cov[:,i], label = 'Eigenvector ' + str(i+1))
    plt.title("Eigenvectors using linalg.eig")
    plt.xlabel('Log Wavelength log((Angstrom))')
    plt.ylabel('Eigenvector (no units)')
    plt.legend()
plt.show()

# # E. CALCULATING EIGENVECTORS FROM SVD
start2 = timeit.default_timer()
u,w,vt = np.linalg.svd(flux_without_offset, full_matrices = False)
eig_vec_svd = vt.transpose()
stop2 = timeit.default_timer()

for i in range(0,5):
    plt.plot(logwave, eig_vec_svd[:,i])
    plt.xlabel('Log Wavelength log((Angstrom))')
    plt.ylabel('Eigenvector (no units)')
    plt.title("Eigenvectors using SVD")
    print(eig_vec_svd[:,i])
plt.show()

# F. EIG Vs. SVD

#comparing runtimes of eig and svd
print()
print("Comparing runtimes of each method:")
print("Runtime for calculating the eigenvectors using linalg.eig: ", stop1 - start1)
print("Runtime for calculating the eigenvectors using SVD: ", stop2 - start2)
print()

#comparing condition numbers of eig and svd
condition_number_eig = np.max(eig_val_cov)/np.min(eig_val_cov)
condition_number_svd = np.max(w)/np.min(w)

print("Comparing condition numbers of each method:")
print("Condition number of R = ", np.linalg.cond(r))
print("Condition number of C = ", np.linalg.cond(c))
print()

# G. DOING THE PCA 

#finding the first 5 principal components
c0 = np.dot(flux_without_offset, eig_vec_svd[:,0])
c1 = np.dot(flux_without_offset, eig_vec_svd[:,1])
c2 = np.dot(flux_without_offset, eig_vec_svd[:,2])
c3 = np.dot(flux_without_offset, eig_vec_svd[:,3])
c4 = np.dot(flux_without_offset, eig_vec_svd[:,4])

weights = np.vstack((c0, c1, c2, c3, c4))  #shape: (5, 9713)

eigenvectors = np.vstack((eig_vec_svd[:,0], eig_vec_svd[:,1], eig_vec_svd[:,2], eig_vec_svd[:,3], eig_vec_svd[:,4])) #shape = (5, 4001)

approx_spectra = np.dot(weights.T, eigenvectors) 

#adding the mean back to the result and multiplying by the sum of the flux
approx_spectra = approx_spectra + flux_mean[:,None]
approx_spectra = approx_spectra*flux_sum[:,None]

# H. PLOTTING c0 Vs. c1 AND c0 Vs. c2
fig, axes = plt.subplots(2, 1, figsize=(6, 8)) 

axes[0].scatter(c1, c0, color='midnightblue')
axes[0].set_xlabel('c1 (no units))')
axes[0].set_ylabel('c0 (no units)')
axes[0].set_title('Plot of c0 vs c1')

axes[1].scatter(c2, c0, color='maroon')
axes[1].set_xlabel('c2 (no units)')
axes[1].set_ylabel('c0 (no units)')
axes[1].set_xlim(-0.003, 0.003)
axes[1].set_title('Plot of c0 vs c2')

plt.tight_layout()
plt.show()

# I. COMPARING THE ORIGINAL AND RECONSTRUCTED SPECTRA

#function definition for performing PCA given an array of flux, eigenvector array and number of components n 
def PCA(flux_without_offset, eig_vec_svd, n):
    # u,w,vt = np.linalg.svd(flux_without_offset, full_matrices = False)
    # eig_vec_svd = vt.transpose()

    weights = np.zeros((flux.shape[0], n))
    eigenvectors = np.zeros((flux.shape[1], n))

    for i in range(0,n):
        c = np.dot(flux_without_offset, eig_vec_svd[:,i])
        weights[:,i] = c
        eigenvectors[:,i] = eig_vec_svd[:,i]
    
    eigen_spectra = np.dot(weights, eigenvectors.T) 

    return eigen_spectra 

#running a loop to perform PCA for different number of principal components (n = 1 to 20)
for i in range(1,21):
    eigen_spectra = PCA(flux_without_offset, eig_vec_svd, i)
    fractional_residuals = ((np.abs(flux_without_offset) - np.abs(eigen_spectra))/np.abs(flux_without_offset))**2 
    residual_mean = np.mean(fractional_residuals, axis=0)
    plt.scatter(logwave, residual_mean, label = 'Nc = ' + str(i))
plt.legend()
plt.xlabel('Log Wavelength (log(Angstrom))')
plt.ylabel('Fractional Mean Residuals (no units)')
plt.title('Plot of Fractional Mean Residuals vs Log Wavelength')
plt.show()





    

    














