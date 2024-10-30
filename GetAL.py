import numpy as np

def GetWeightedSTD(values, weights):
    '''
    Calculate the weighted standard deviation.
    ref: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Parameters
    ----------
    values:    Array of values for the standard deviation calculation
    weights:   Weights associated with the values array
    '''
    
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    
    return np.sqrt(variance)

def GetEign(CorePosX, CorePosY, CoreWeight=None):
    '''
    Calculates the eigenvalues and eigenvectors (principal axes) 
    for a set of 2D core positions by solving the (weighted) 
    principal component analysis (PCA).

    Parameters
    ----------
    CorePosX:   1D array representing the core positions along 
                the x-axis in pixels.
    CorePosY:   1D array representing the core positions along 
                the y-axis in pixels.
    CoreWeight: 1D array (optional) representing the weights 
                associated with each core. If specified, the 
                PCA calculation incorporates these weights.

    Returns
    -------
    Eign:    The eigenvalues along the two eigenvectors.
    EignVec: The eigenvectors or principal axes.
    '''
    
    Covar = np.array([[0., 0.], [0., 0.]])

    if CoreWeight is None:
        CoreWeight = np.ones_like(CorePosX)

    # Shift the core positions to center-of-mass frame
    CorePosX = CorePosX - np.average(CorePosX, weights=CoreWeight)
    CorePosY = CorePosY - np.average(CorePosY, weights=CoreWeight)

    # Construct covariance matrix
    for PosX, PosY, wi in zip(CorePosX, CorePosY, CoreWeight):
        Covar[0, 0] += wi*PosX**2/np.sum(CoreWeight)
        Covar[0, 1] += wi*PosX*PosY/np.sum(CoreWeight)
        Covar[1, 0] += wi*PosY*PosX/np.sum(CoreWeight)
        Covar[1, 1] += wi*PosY**2/np.sum(CoreWeight)

    # Solve the eigenfunction
    Eign, EignVec = np.linalg.eig(Covar)

    EignVec = EignVec[np.argsort(Eign)[::-1]]
    Eign    = Eign[np.argsort(Eign)[::-1]]

    return Eign, EignVec

def GetAL(CorePosX, CorePosY, CoreWeight, Simga_mPCA, Simga_mWPCA):
    '''
    Obtain the alignment parameters.

    Parameters
    ----------
    CorePosX:    1D array representing the core positions along 
                 the x-axis in pixels.
    CorePosY:    1D array representing the core positions along 
                 the y-axis in pixels.
    CoreWeight:  1D array representing the weights associated 
                 with each core.
    Simga_mPCA:  The characteristic minor axis length obtained 
                 from GetEign when CoreWeight is omitted.
    Simga_mWPCA: The characteristic minor axis length obtained 
                 from GetEign when CoreWeight is specified.

    Returns
    -------
    ALuw: The unweighted alignment parameter (Equation 2 in 
          the accompanying paper).
    ALw:  The weighted alignment parameter(Equation 3 in the 
          accompanying paper).
    '''
    
    CoreSijWiWj = []
    CorePos     = np.stack((CorePosX, CorePosY), axis=1)

    if len(CorePos) == 1:
        ALuw = -1
        AL   = -1
        Sigma_ALuw = 0
        Sigma_ALw  = 0

    elif len(CorePos) == 2:
        ALuw = -2
        AL   = -2
        Sigma_ALuw = 0
        Sigma_ALw  = 0
        
    else:
        for i, Posi in enumerate(CorePos):
            for j, Posj in enumerate(CorePos):
                if j!=i:
                    Sij      = np.linalg.norm(Posi-Posj)
                    SpijPCA  = Sij/Simga_mPCA  # Equation 1, sigma_m from PCA
                    SpijWPCA = Sij/Simga_mWPCA # Equation 1, sigma_m from weighted PCA
                    wiwj     = CoreWeight[i]*CoreWeight[j]
                    CoreSijWiWj.append([SpijPCA, SpijWPCA, wiwj])
        
        CoreSijWiWj = np.array(CoreSijWiWj)
    
        ALuw = np.average(CoreSijWiWj[:, 0]) # Equation 2
        ALw  = np.average(CoreSijWiWj[:, 1], weights=CoreSijWiWj[:, 2]) # Equation 3

        Sigma_ALuw = np.std(CoreSijWiWj[:, 0])/np.sqrt(len(CoreSijWiWj)) # the error of the mean
        Sigma_ALw  = GetWeightedSTD(CoreSijWiWj[:, 1], weights=CoreSijWiWj[:, 2])/np.sqrt(len(CoreSijWiWj)) # the error of the mean (calculate weighted inside the function)
    
    return ALuw, ALw, Sigma_ALuw, Sigma_ALw