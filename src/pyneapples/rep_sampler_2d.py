#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "joblib",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
# -*- coding: utf-8 -*-
"""
Program for the selection of the most representative molecular geometries for spectra modelling.

@author: Stepan Srsen
"""

import sys #allows access to system-specific parameters and functions
import numpy as np #numerical opperations and arrays
import random #random number generations / random selections
import math #mathematical functions
import time #tracking execution times / delays
import os #interacts with the oparating system, file and directory operations
from joblib import Parallel, delayed, cpu_count #imported tools for parallel processing
from argparse import ArgumentParser #creating a command-line interface (interacting with a computer program by inputting lines of text - command lines)
import datetime
from scipy.stats import gaussian_kde #Kernel Density Estimation (KDE), a non-parametric way to estimate a PDF
from scipy.spatial.distance import pdist, squareform #computing pairwise distances and converting them into square matrices
import matplotlib as mpl 
# mpl.use('agg') # noninteractive backend when there are problems with the $DISPLAY
import matplotlib.pyplot as plt

def read_cmd():
    """Function for command line parsing."""   #command line parsing (string analysis)
#    parser = calc_spectrum.read_cmd(parse=False)
    parser = ArgumentParser(description='Spectrum reduction.')   #creates an ArgumentParser object with a description of the programme
    parser.add_argument('infile', help='Input file.')   #defines a positional argument for the input file
    parser.add_argument('-n', '--nsamples', type=int, default=1,
                        help='Number of samples.')   #defines an optional argument for the number of samples
    parser.add_argument('-N', '--nstates', type=int, default=1,
                        help='Number of excited states (ground state not included).')   #efines an optional argument for the number of excited states (not counting the ground state)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Activate verbose mode.')   #defines a flag to enable 'verbose' output
    parser.add_argument('-j', '--ncores', type=int, default=1,
                        help='Number of cores for parallel execution of computatinally intensive subtasks:'
                        + ' cross-validation bandwidth setting, error bars, geometry reduction.')   #defines an optional argument for the number of cores used in parallel processing
 
    parser.add_argument('-S', '--subset', type=int, default=0,
                        help='Number of representative molecules.')   #defines an optional argument for the number of representative molecules (subset size)
    parser.add_argument('-c', '--cycles', type=int, default=1000,
                        help='Number of cycles for geometries reduction.')   #defines an optional argument for the number of cycles in the geometry reduction (optimisation) process
    parser.add_argument('-J', '--njobs', dest='njobs', type=int, default=1,
                        help='Number of reduction jobs.')   #defines an optional argument for the number of independent reduction jobs
    parser.add_argument('-w', '--weighted', action='store_true',
                        help='Weigh the distributions during optimization by spectroscopic importance ~E*tdm^2.')   #defines a flag to weigh distributions during optimization by spectroscopic importance (~E*tdm^2)
    parser.add_argument('--pdfcomp', choices=['KLdiv','JSdiv','KStest', 'kuiper', 'SAE', 'RSS', 'cSAE', 'cRSS'], default='KLdiv',
                        help='Method for comparison of probability density functions.')   #defines an optional argument to choose the method for comparing probability density functions
    parser.add_argument('--intweights', action='store_true',
                        help='Activate optimization of integer weights for individual geometries (instead of 0/1).')   #defines a flag (T/F value, this case 0/1) to optimize integer weights (instead of binary 0/1 selection) for each geometry

    return parser.parse_args()   #parses the command line (breaks into components) and returns the options

class PDFDiv:   ### PDF divergence methods
    """Class with different methods to calculate the divergence of two probability density functions."""

    @staticmethod
    def KLdiv(pdf1, pdf2, normalized=False, normalize=False):
        """Generalized Kullback-Leibler divergence. pdf1 is used for probabilities."""

        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Interpretations
        # maybe normalize both by pdf1 for exact but comparable results? currently done in get_PDF
        if normalize or not normalized:   #if normalisation is required or the data is not yet normalized, calculate the total weight
            norm1 = np.sum(pdf1)
            norm2 = np.sum(pdf2)
        if normalize:   #normalisation of the PDFs if requested
            pdf1 /= norm1
            pdf2 /= norm2
            normalized = True
        thr = 1e-15   #tiny threshold to avoid division by zero
        if not normalized:
            thr *= norm1
        indices = pdf1>thr   #considering indices only where pdf1 is above the threshold
        # print(pdf1.shape)
        pdf1 = pdf1[indices]
        # print(pdf1.shape)
        pdf2 = pdf2[indices]
        pdf1 = pdf1 + thr   #offset by the threshold to avoid taking logs of zero
        pdf2 = pdf2 + thr   #offset by the threshold to avoid taking logs of zero
        
        d = np.divide(pdf1,pdf2)   #computing the ratio
        np.log(d, out=d)   #computing the log of the ratio
        np.multiply(d, pdf1, out=d)   #multiplying elements by pdf1
        d = np.sum(d)
        if not normalized:
            d += -norm1 + norm2   #adjusting for the unnormalised case
    #    print(d)
        return d

    @staticmethod
    def JSdiv(pdf1, pdf2):
        """Jensen–Shannon divergence."""   #symmetrised and smoothed version of the KL divergence

        pdf3 = (pdf1 + pdf2) / 2   #average of the 2pdfs
        d = 0.5*PDFDiv.KLdiv(pdf1, pdf3) + 0.5*PDFDiv.KLdiv(pdf2, pdf3)   #computing the divergence as the average of the KL divergence from each pdf to the average
    #    print(d)
        return d
    
    @staticmethod
    def KStest(pdf1, pdf2):
        """Kolmogorov–Smirnov test."""   #computes the maximum difference between the cumulative distribution functions (CDFs)

        cdf1 = 0.0
        cdf2 = 0.0
        d = 0.0
        for i in range(len(pdf1)):   #incrementally build the CDFs and track the maximum difference
            cdf1 += pdf1[i]
            cdf2 += pdf2[i]
            dact = abs(cdf1-cdf2)
            if dact > d:
                d = dact
        return d
    
    @staticmethod
    def kuiper(pdf1, pdf2):
        """Kuiper test."""   #similar to the KS test but takes into account both the maximum positive and negative deviations

        cdf1 = 0.0
        cdf2 = 0.0
        dminus = 0.0
        dplus = 0.0
        for i in range(len(pdf1)):   #computes cumulative sums and track the deviations
            cdf1 += pdf1[i]
            cdf2 += pdf2[i]
            dminusact = cdf1-cdf2
            dplusact = -dminusact
            if dminusact > dminus:
                dminus = dminusact
            if dplusact > dplus:
                dplus = dplusact
        d = dplus+dminus   #the Kuiper statistic is the sum of the maximum deviations
        return d
    
    @staticmethod
    def SAE(pdf1, pdf2):
        """Sum of absolute errors/differences."""   #computes the total absolute difference between the two PDFs

        # proc ne suma ctvercu odchylek? ---> percent not sum of ??? deviations
        d = np.sum(np.abs(pdf1-pdf2))
        return d
    
    @staticmethod
    def RSS(pdf1, pdf2):
        """Residual sum of squares."""   #computes the sum of the squared differences between the two PDFs

        d = np.sum(np.power(pdf1-pdf2, 2))
        return d
    
    @staticmethod
    def cSAE(pdf1, pdf2):
        """Sum of absolute errors/differences of CDFs corresponding to given PDFs."""
        #computes the absolute differences between the cumulative distribution functions (CDFs) derived from the PDFs
        cdf1 = np.cumsum(pdf1)
        cdf2 = np.cumsum(pdf2)
        d = np.sum(np.abs(cdf1-cdf2))
        return d
    
    @staticmethod
    def cRSS(pdf1, pdf2):
        """Residual sum of squares of CDFs corresponding to given PDFs."""   #computes the squared differences between the CDFs corresponding to the PDF

        cdf1 = np.cumsum(pdf1)
        cdf2 = np.cumsum(pdf2)
        d = np.sum(np.power(cdf1-cdf2, 2))
        return d

class GeomReduction:
    """Main class for the optimization of representative sample."""

    def __init__(self, nsamples, nstates, subset, cycles, ncores, njobs, weighted, pdfcomp, intweights, verbose, dim1=False):
        self.nsamples = nsamples
        # if nstates > 1:
        #     print("ERROR: implemented only for 1 state!")
        #     return False
        self.nstates = nstates   #number of excited states
        self.exc = np.empty((nsamples, nstates))   #array holding the excitation energies
        self.trans = np.empty((nsamples, nstates, 3))   #array for 3D transition dipole moments
        self.grid = None   #grid later created for evaluating PDFs
        self.subset = subset   #number of representative molecules to select
        self.cycles = cycles   #number of cycles for the simulated annealing process   ???
        self.ncores = ncores   #number of CPU cores for parallel tasks
        self.njobs = njobs   #number of independent reduction jobs
        self.verbose = verbose   #verbosity flag
        self.subsamples = []   #list to store indices of the selected representative subsample
        self.sweights = None   #weights associated with each selected geometry
        self.origintensity = None   #PDF of the full (original) data
        self.weighted = weighted   #flag to use weighted PDFs
        self.calc_diff = getattr(PDFDiv, pdfcomp)   #dynamically assign the divergence function based on the chosen method
        self.intweights = intweights   #whether to optimize integer weights
        self.dim1 = dim1   #Flag: if True, operate in one dimension only (excitation energy only)
        self.pid = os.getpid()   #process ID, used for naming files and logging

    def read_data_direct(self, excitation_energies, transition_dipole_moments_x, transition_dipole_moments_y, transition_dipole_moments_z):

        self.infile = "Test_Filename"   #stores the filename
        self.time = datetime.datetime.now() #records the current date and time

        self.exc = excitation_energies #assign the provided excitation energies directly to the instance variable
        self.trans = np.stack((transition_dipole_moments_x, transition_dipole_moments_y, transition_dipole_moments_z), axis=-1)  #combine the separate transition dipole moment components (x, y, and z) into a single NumPy array. The np.stack() function creates a new dimension at the end, so that for each sample (and possibly state) you get a 3-element vector of [x, y, z]

        self.trans = np.power(self.trans,2)   #post-processing: square the transition dipole moments
        self.trans = np.sum(self.trans, axis=2)   #sum the squared components to obtain a single scalar value per transition
        self.weights = self.exc*self.trans   #calculate the weight for each transition as the product of excitation energy and the (summed) dipole moment
        self.wnorms = np.sum(self.weights, axis=0)/np.sum(self.weights) #provides a normalization factor for each state, ensuring that the weights are comparable across states

    def read_data_direct_osc(self, excitation_energies, oscillator_stregths):

        evs_in_au = 27.211396 # Conversion factor from electronvolts to atomic units

        self.infile = "Test_Filename"   #stores the filename
        self.exc = excitation_energies # Assign excitation energies in eV units
        self.time = datetime.datetime.now() #records the current date and time

        self.trans = abs((3*oscillator_stregths)/(2*(self.exc/evs_in_au)))  # Calculate transition dipole moment scaler value (as above) using energy in a.u.

        self.weights = self.exc*self.trans   #calculate the weight for each transition as the product of excitation energy and the (summed) dipole moment
        self.wnorms = np.sum(self.weights, axis=0)/np.sum(self.weights) #provides a normalization factor for each state, ensuring that the weights are comparable across states


    def read_data(self, infile):
        """Reads and parses input data from given input file."""

        self.infile = infile   #stores the filename
        self.time = datetime.datetime.now()   #timestamp for filenaming
        with open(self.infile, "r") as f:
            i = 0 #line counter
            j = 0 #sample counter
            k = -1 #state counter (will be incremented to start at 0)
            for line in f:
                if (i % 2 == 1):   #odd-numbered lines (contain the transition dipole moment components)
                    temp = line.split()   #splits the line into individual number strings
                    try:
                  # assigning transition dipole moments as a tuple
                        self.trans[j][k] = (float(temp[0]), float(temp[1]), float(temp[2]))
                    except:   #error handling if the expected three numbers are not present
                        print("Error: Corrupted line "+str(i+1)+" in file "+self.infile)
                        print("I expected 3 columns of transition dipole moments, got:")
                        print(line)
                        sys.exit(1)
                else:   #even-numbered lines (contain excitation energy)
                    k += 1   #move to the next state
                    if k == self.nstates:
                        k = 0   #reset state counter once all states have been read
                        j += 1   #move to the next sample
                    if j >= self.nsamples:
                        if line.strip() != "":   #check if there are extra transitions
                            print("Error: Number of transitions in the input file is bigger than the number of samples multiplied by the number of states.")
                            sys.exit(1)
                        break
                    try:
                        self.exc[j][k] = float(line)   #convert the excitation energy to a float and store it
                    except:
                        print("Error when reading file "+self.infile+" on line: "+str(i+1))
                        print("I expected excitation energy, but got:" + line)
                        sys.exit(1)
                i += 1
            if (i != 2*self.nsamples*self.nstates):   #verify that the file contained exactly the expected number of lines
                print("Error: Number of transitions in the input file is smaller than the number of samples multiplied by the number of states.")
                sys.exit(1)

        self.trans = np.power(self.trans,2)   #post-processing: square the transition dipole moments
        self.trans = np.sum(self.trans, axis=2)   #sum the squared components to obtain a single scalar value per transition
        self.weights = self.exc*self.trans   #calculate the weight for each transition as the product of excitation energy and the (summed) dipole moment
        self.wnorms = np.sum(self.weights, axis=0)/np.sum(self.weights)   #normalise the weights per state
        
    def get_name(self):
        """Defines the basename for the generated files."""

        bname = os.path.basename(self.infile)   #extracts the base name (without directory) of the input file
        name = bname.split(".")[0]   #removes the file extension
        return 'absspec.' + name + '.n' + str(self.nsamples) + '.' + self.time.strftime('%Y-%m-%d_%H-%M-%S') # + '.' + str(self.pid)   #creates a unique file name using the base name, number of samples, and a timestamp

        
    def get_PDF(self, samples=None, sweights=None, h='silverman', gen_grid=False):
        """Calculates probability density function for given data on a grid."""    #uses Gaussian kernel density estimation (KDE) to approximate the PDF from a set of samples

        # TODO: compare each state separately or create common grid and intensity?
        # TODO: weigh states by corresponding integral intensity, i.e. sum(ene*trans**2)?
        if samples is None:
            samples = slice(None)   #use all samples if none are specified
        
        if gen_grid:   #gnerates the grid on which the PDF will be evaluated
     
            # TODO: accept the params as argument, e.g. gen_grid=(100,1)
            self.n_points = 100   #number of grid points per dimesnion
            n_sigma = 1   #extends the grid by 1 standard deviation
            
            norm = 1
            if self.weighted:
                if sweights is not None:   #normalisation using the weighted sum of intensities
                    norm = np.sum(self.weights[samples]*sweights)/np.sum(sweights)
                else:
                    norm = np.sum(self.weights[samples])/len(self.weights[samples])
            h1 = np.amax(np.std(self.exc[samples], axis=0))   #estimates bandwidth based on the standard deviation of the excitation energies
            self.exc_min = self.exc[samples].min() - n_sigma*h1   #defines grid limits for excitation energies (minimum)
            self.exc_max = self.exc[samples].max() + n_sigma*h1   #defines grid limits for excitation energies (maximum)
            dX = (self.exc_max - self.exc_min)/(self.n_points-1)  #grid spacing
            if self.dim1:   #for 1D KDE, creates a linear grid
                self.grid = np.linspace(self.exc_min, self.exc_max, self.n_points)
                self.norm = dX/norm
            else:   #for 2D KDE, defines grid limits for the transition dipole moments
                h2 = np.amax(np.std(self.trans[samples], axis=0))
                self.trans_min = self.trans[samples].min() - n_sigma*h2
                self.trans_max = self.trans[samples].max() + n_sigma*h2
                X, Y = np.mgrid[self.exc_min : self.exc_max : self.n_points*1j, self.trans_min : self.trans_max : self.n_points*1j]   #creates a mesh grid using mgrid
                dY = (self.trans_max - self.trans_min)/(self.n_points-1)
                self.norm = dX*dY/norm   #sets the norm using full-sample PDF to obtain comparable values of divergences (combined normalisation factor)
                self.grid = np.vstack([X.ravel(), Y.ravel()])   #flattens the grid into a 2-row array (each row represents one coordinate axis)
            if self.subset == 1:   #for the special case of a single representative geometry, prepares a list to store kernels
                self.kernel = []
        
        # pdf = np.zeros((self.nstates, self.n_points**2))
        if self.dim1:   #initialise an array for the PDF values (different shape for 1D vs. 2D)
            pdf = np.zeros((self.n_points))
        else:
            pdf = np.zeros((self.n_points**2))
            
        for state in range(self.nstates):       #loops over each excited state
            exc = self.exc[samples,state]       #extracting excitation energies
            trans = self.trans[samples,state]   #extracting corresponding dipole data
            if self.dim1:
                values = exc[None,:]   #for 1D
            else:
                values = np.vstack([exc, trans]) # TODO: index values directly   ####for 2D, stacks excitation energies and dipole values
            # h = bandwidth
            norm = self.wnorms[state]   #sets normalisation value for the state
            weights = None
            if self.weighted:
                if sweights is not None:
                    norm = np.sum(self.weights[samples,state]*sweights)/np.sum(sweights)
                    weights = self.weights[samples,state]*sweights
                else:
                    norm = np.sum(self.weights[samples,state])/len(self.weights[samples,state])
                    weights = self.weights[samples,state]
            elif sweights is not None:
                weights = sweights
            if gen_grid and self.subset == 1:   #creates a Gaussian KDE object using the chosen bandwidth method and optional weights
                kernel = gaussian_kde(values, bw_method=h, weights=weights)
                # save the kernels so they can be reused later for self.subset=1 as they cannot be initialized in a regular way 
                self.kernel.append(kernel)   #saves the kernel so that it can be reused later (necessary when subset==1)
            elif self.subset == 1:
                # reuse the saved kernel when self.subset=1 as they cannot be initialized in a regular way 
                kernel = self.kernel[state]   #reuses the saved kernel
                kernel.dataset = values   #updates the data in the kernel
                # kernel.weights = weights[:, None]
                # print(kernel.dataset)
                # norm *= self.nsamples
            else:
                kernel = gaussian_kde(values, bw_method=h, weights=weights)
                
            # pdf[state] = kernel(self.grid[state])*self.norm[state]*norm #*self.gweights
            pdf += kernel(self.grid)*self.norm*norm#*self.wnorms[state] #*self.gweights           ###evaluates the kernel on the grid and accumulates the contribution
            # print('pdf sum', np.sum(pdf), norm)
        
        return pdf
        
    def select_subset(self, randomly=True):
        """Random selection of a subsample of a given size."""   #returns the indices (and optionally initial integer weights) of the representative geometries

        if randomly:   
            samples = np.array(random.sample(range(self.nsamples), self.subset))   #randomly selects 'subset' number of sample indices from the full set
        else:
            if self.nstates > 1:   #deterministic selection based on maximizing distances (only applicable for one state)
                print('ERROR: intial subset generation with maximal distances is not supported for multiple states.')
                return
            exc = self.exc[:,0] # only for one state   ###considers only the excitation energies of the single state
            trans = self.trans[:,0]
            weights = self.weights[:,0]
            exc = exc/np.average(exc, weights=weights)   #normalises the values by their weighted averages
            trans = trans/np.average(trans, weights=weights)
            values = np.vstack([exc, trans]).T   #stacks into 2D coordinates
            dists = squareform(pdist(values))   #computes the pairwise distances between points
            samples = [np.argmax(np.sum(dists, axis=1))]   #chooses the sample with the largest overall distance as the first representative
            while len(samples) < self.subset:   #iteratively adds the sample which maximizes the minimal distance to the current subset
                sample = np.argmax(np.min(dists[:,samples], axis=1))
                samples.append(sample)
            samples = np.array(samples)
        
        if self.intweights and self.subset>1:
            weights = int(self.nsamples/self.subset + 0.5)*np.ones(samples.shape, dtype=int)   #for integer weights, initialises each with nsamples/subset (rounded)
        else:
            weights = None
        return samples, weights
    
    def swap_samples(self, samples, weights=None):
        """Swap one datapoint between the representative subsample and the rest."""   #The function makes a small change in the current solution by either swapping one sample or adjusting weights

        index1 = random.randrange(len(samples))   #randomly chooses an index within the current subsample
        change_weights = np.random.randint(5) # prob to change weights instead of swapping given by 1-1/change_weights   ###decides randomly (with about a 1 in 5 chance) to change weights rather than swap
        # change_weights = 1
        if change_weights==0 or weights is None:
            rest = list(set(range(self.nsamples)) - set(samples))   #computes the set of indices not in the current subsample
            index2 = random.randrange(len(rest))
            samples[index1] = rest[index2]   #swaps the selected sample with a new one from the remaining set
            return samples, weights
        index2 = random.randrange(len(samples))   #otherwise, adjusts the integer weights
        while weights[index2]==1 or index1==index2:   #ensures a weight is not reduced below 1
            index1 = random.randrange(len(samples))
            index2 = random.randrange(len(samples))
        weights[index1] += 1
        weights[index2] -= 1
        # add = np.random.randint(2)
        # if add or weights[index1]==1:
        #     weights[index1] += 1
        # else:
        #     weights[index1] -= 1
        return samples, weights

    def SA(self, test=False, pi=0.9, pf=0.1, li=None, lf=None):
        """Simulated annealing optimization for the selection of a subsample minimizing given divergence."""   #Simulated Annealing is a probabilistic method that allows uphill moves (worse solutions) with a probability that decreases over time (temperature), helping to avoid local minima

        if test:   #if in test mode, performs a short run to calibrate parameters
            subsamples = self.subsamples
            weights = self.sweights
            it = 1
            diffmax = 0
            diffmin = np.inf
        else:
            subsamples, weights = self.select_subset()   #starts by selecting an initial subset
            subsamples_best = subsamples
            weights_best = weights
            d_best = np.inf   #initialises best divergence to infinity
            
            nn = self.subset*(self.nsamples-self.subset)   #computes the total number of possible swaps
            if not li:   
                itmin = 1
            else:
                itmin = nn*li
            if not lf:
                itmax = int(math.ceil(nn/self.nsamples))
            else:
                itmax = nn*lf
            if itmin==itmax:
                itc = 1
                loops = itmin*self.cycles
            else:
                itc = math.exp((math.log(itmax)-math.log(itmin))/self.cycles)   #adjusts the length of the Markov chain over cycles
                loops = int(itmin*(itc**(self.cycles)-1)/(itc-1))   #neglects rounding
            it = itmin
            
            self.subsamples = np.copy(subsamples)
            if weights is not None:
                self.sweights = np.copy(weights)
            sa_test_start = time.time()   #runs a short test cycle to estimate the appropriate temperature range
            ti, tf = self.SA(test=True, pi=pi, pf=pf)
            sa_test_time = time.time() - sa_test_start
            tc = math.exp((math.log(tf)-math.log(ti))/self.cycles)   #calculates the temperature decay coefficient
            temp = ti   #sets the initial temperature

        intensity = self.get_PDF(samples=subsamples, sweights=weights)   #computes the PDF of the current subsample
        d = self.calc_diff(self.origintensity, intensity)   #calculates the divergence (difference) between the original full-sample PDF and the subsample PDF
        # d = 0
        # for state in range(self.nstates):
        #     d += self.calc_diff(self.origintensity, intensity)*self.wnorms[state]
        
        if not test:   #estimates and prints the expected run time based on the test cycle
            m, s = divmod(int(round(sa_test_time*loops/self.cycles)), 60)
            h, m = divmod(m, 60)
            print('Ti', ti, 'Tf', tf)
            print('Li', itmin, 'Lf', itmax)
            toprint = str(self.pid)+":\tInitial temperature = "+str(ti)
            toprint += ", Final temperature = "+str(tf)+", Temperature coefficient = "+str(tc)
            toprint += "\n\tMarkov Chain Length coefficient = "+str(itc)+", Initial D-min = "+str(d)
            toprint += "\n\tEstimated run time: "+str(h)+" hours "+str(m)+" minutes "+str(s)+" seconds"
            print(toprint)
#         sys.stdout.flush()
            
        for _ in range(self.cycles):   #begins the SA cycles 
            for _ in range(int(round(it))):
                subsamples_i = np.copy(subsamples)   #creates a candidate solution by copying the current subsample
                weights_i = None
                if weights is not None:
                    weights_i = np.copy(weights)
                subsamples_i, weights_i = self.swap_samples(subsamples_i, weights_i)   #makes a small change (swap one sample or adjust weights)
                intensity = self.get_PDF(samples=subsamples_i, sweights=weights_i)   #evaluates the PDF and computes its divergence
                d_i = self.calc_diff(self.origintensity, intensity) 
                # d_i = 0
                # for state in range(self.nstates):
                #     d_i += self.calc_diff(self.origintensity, intensity)*self.wnorms[state]
                # print('d', d)
                if test:
                    prob = 1
                    diff = abs(d_i - d)
                    if diff > diffmax:
                        diffmax = diff
                    elif diff < diffmin and diff > 0:
                        diffmin = diff
                else:
                    if d_i < d:
                        prob = 1.0
                        if d_i < d_best:
                            subsamples_best = subsamples_i
                            weights_best = weights_i
                            d_best = d_i
                    else:
                        prob = math.exp((d - d_i)/ temp)   #accepts worse solutions with a probability that decreases with temperature
                if prob >= random.random():   #decides whether to accept the candidate solution
                    subsamples = subsamples_i
                    weights = weights_i
                    d = d_i
            if not test:   #decreases the temperature and adjusts the chain length
                temp *= tc
                it *= itc
        if test:
            print('diffmax', diffmax, 'diffmin', diffmin, 'd', d)
            return -diffmax/math.log(pi), -diffmin/math.log(pf)   #returns estimated initial and final temperatures
        
        pdf = self.get_PDF(subsamples_best, sweights=weights_best)   #evaluates the best candidate and prints the result
        print('PDF sum', np.sum(pdf))
        print('best d', d_best)
        self.subsamples = subsamples_best
        self.sweights = weights_best
        print(subsamples_best, weights_best)
        return d_best

    # def random_search(self):
    #     """Optimization of the representative sample using random search to minimize given divergence."""
    #
    #     self.sweights = None
    #     div = np.inf
    #     for i in range(self.cycles):
    #         subsamples, _ = self.select_subset()
    #         if self.recalc_sigma:
    #             intensity = self.spectrum.recalc_kernel(samples=subsamples)
    #         else:
    #             intensity = self.spectrum.recalc_spectrum(samples=subsamples)
    #         div_act = self.calc_diff(self.origintensity, intensity)
    #         if div_act <= div:
    #             self.subsamples = subsamples
    #             div = div_act
    #             print("Sample"+str(i)+": D-min ="+str(div))
    #     if self.recalc_sigma:
    #         self.spectrum.recalc_kernel(samples=self.subsamples)
    #     else:
    #         self.spectrum.recalc_spectrum(samples=self.subsamples)
    #     return div
    
    def extensive_search(self, i):
        """Optimization of the representative geometry using extensive search to minimize given divergence."""   #for a given sample index i, computes the divergence without modifying the subsample

        self.subsamples = [i]
        self.sweights = None
        # if self.recalc_sigma:
        #     self.spectrum.recalc_kernel(samples=self.subsamples)
        intensity = self.get_PDF(self.subsamples, self.sweights)
        div = self.calc_diff(self.origintensity, intensity)
        return div

    def reduce_geoms_worker(self, i, li=None, lf=None):
        """Wrapper for SA opt. for the selection of a subsample minimizing given divergence."""   #run as a worker in parallel processing

        name = self.get_name() + '.r' + str(self.subset)   #creates a unique directory name for the results
        os.chdir(name)   #changes directory into the result folder
        orig_stdout = sys.stdout   #saves the original standard output
        with open('output_r'+str(self.subset)+'.txt', 'a') as f:
           sys.stdout = f   #redirects the output to a log file
           div = self.SA(li=li, lf=lf)   #runs the simulated annealing optimisation
           #self.spectrum.writeout(i)
           self.writegeoms('r'+str(self.subset)+'.'+str(i))   #saves the selected geometries
        sys.stdout = orig_stdout   #restores the standard output
        os.chdir('..')   #goes back to the parent directory
        return div, self.subsamples, self.sweights

    #TODO: make random search work                                                                            
    #def random_geoms_worker(self, i):
    #    """Wrapper for representative sample opt. using random search to minimize given divergence."""
    #
    #    name = self.get_name() + '.r' + str(self.subset)
    #    os.chdir(name)
    #    orig_stdout = sys.stdout
    #    with open('output_r'+str(self.subset)+'_rnd.txt', 'a') as f:
    #       sys.stdout = f
    #       div = self.random_search()
    #       #self.spectrum.writeout("rnd."+str(i))
    #       self.writegeoms('r'+str(self.subset)+'.'+'rnd.'+str(i))
    #    sys.stdout = orig_stdout   
    #    os.chdir('..')
    #    return div, self.subsamples, self.sweights
    
    def extensive_search_worker(self, i):   #extensive search worker is an alternative (exhaustive search) method
        """Wrapper for representative geometry opt. using extensive search to minimize given divergence."""   

        name = self.get_name() + '.r' + str(self.subset)
        os.chdir(name)
        orig_stdout = sys.stdout
        with open('output_r'+str(self.subset)+'_ext.txt', 'a') as f:
           sys.stdout = f
           div = self.extensive_search(i)
           #self.spectrum.writeout("ext."+str(i))
           #self.writegeoms('r'+str(self.subset)+'.'+'.ext.'+str(i))
        sys.stdout = orig_stdout   
        os.chdir('..')
        return div, self.subsamples, self.sweights

    def process_results(self, divs, subsamples, sweights, suffix=''):
        """Process and print results from representative sample optimization."""    #selects the best solution (lowest divergence) and writes output files

        print('average divergence', np.average(divs))
        print('divergence std', np.std(divs))
        min_index = np.argmin(divs)
        min_div = divs[min_index]
        self.subsamples = subsamples[min_index]
        self.sweights = sweights[min_index]
        print('minimum divergence:', min_div, ', minimum index:', min_index)
        self.writegeoms('r'+str(self.subset)+'.'+suffix+str(min_index))
        intensity = self.get_PDF(self.subsamples, self.sweights)
        print('optimal PDF sum', np.sum(intensity))
        name = self.get_name()+'.r'+str(self.subset)+'.'+suffix+str(min_index)
        np.savetxt(name+'.exc.txt', self.exc[self.subsamples])   #saves the excitation energies, transition dipole moments, and PDF data to text files
        np.savetxt(name+'.tdm.txt', self.trans[self.subsamples])
        np.savetxt(name+'.pdf.txt', np.vstack((self.grid, intensity)).T)
        self.save_pdf(pdf=intensity, fname=name+'.pdf', markers=True)   #saves a plot image of the PDF

    def reduce_geoms(self):
        """Central function calling representative sample optimization based on user inputs."""    #calculates the full-sample PDF, saves initial data, and then runs optimization jobs in parallel

        self.origintensity = self.get_PDF(gen_grid=True)
        print('original PDF sum', np.sum(self.origintensity))
        #np.savetxt(self.get_name()+'.exc.txt', self.exc)
        #np.savetxt(self.get_name()+'.tdm.txt', self.trans)
        np.savetxt(self.get_name()+'.pdf.txt', np.vstack((self.grid, self.origintensity)).T)
        self.save_pdf(pdf=self.origintensity, fname=self.get_name()+'.pdf')
        if self.subset == 1:
            # edit the saved kernels for self.subset=1 as they cannot be initialized in a regular way
            # maybe move to get_PDF?
            for kernel in self.kernel:   #for the case of a single representative geometry, adjusts the saved kernels
                kernel.set_bandwidth(bw_method=1)
                kernel.n = 1
                kernel._neff = 1
                kernel._weights = np.ones((1))

        name = self.get_name() + '.r' + str(self.subset)   #creates a directory for saving the reduction results
        os.mkdir(name)
        
        with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:   #run the SA optimisation in parallel using the specified number of cores
            divs, subsamples, sweights = zip(*parallel(delayed(self.reduce_geoms_worker)(i) for i in range(self.njobs)))
        print('SA divergences:')
        self.process_results(divs, subsamples, sweights)

        # # calculate # of loops to provide comparable resources to random search        
        # nn = self.subset*(self.nsamples-self.subset)
        # itmin = 1
        # itmax = int(math.ceil(nn/self.nsamples))
        # itc = math.exp((math.log(itmax)-math.log(itmin))/self.cycles)
        # loops=0
        # it=itmin
        # for _ in range(self.cycles):
        #     for _ in range(int(round(it))):
        #         loops+=1
        #     it*=itc
        # print('# of loops', loops)
        # # print('loops approx.', int(itmin*(itc**(self.cycles)-1)/(itc-1)), 'Li', itmin, 'Lm', itmax)
        # self.cycles = loops
        # with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
        #     divs, subsamples, sweights = zip(*parallel(delayed(self.random_geoms_worker)(i) for i in range(self.njobs)))
        # print('Random divergences:')
        # self.process_results(divs, subsamples, sweights, suffix='rnd.')
        
        if self.subset==1:   #if only one representative geometry is requested, performs an exhaustive search over all samples
            with Parallel(n_jobs=self.ncores, verbose=1*int(self.verbose)) as parallel:
                divs, subsamples, sweights = zip(*parallel(delayed(self.extensive_search_worker)(i) for i in range(self.nsamples)))
            min_index = np.argmin(divs)
            print('Extensive search = global minimum:')
            self.process_results(divs, subsamples, sweights, suffix='ext.')

    def save_pdf(self, pdf, fname, markers=False, plot=False, ext='png', dpi=72):
        """Saves PDF as an image."""   #creates a plot of the PDF (and optionally overlays markers for selected geometries) and saves it to a file

        samples = self.subsamples
        if not plot:
            plt.ioff()   #turns off interactive plotting
        plt.figure()
        plt.xlim([self.exc_min, self.exc_max])
        plt.xlabel('$\mathit{E}$/eV')
        if self.dim1:
            if markers:   #for 1D data, plots the PDF versus excitation energy
                plt.plot(self.exc[samples].ravel(), np.zeros((len(self.exc[samples].ravel()))), 'k.', markersize=2)
            #    plt.plot(self.grid, self.origintensity)
            plt.plot(self.grid, pdf)
        else:
            Z = np.reshape(pdf.T, (self.n_points,self.n_points))   #for 2D data, reshapes the PDF to a grid and displays it as an image
            plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[self.exc_min, self.exc_max, self.trans_min, self.trans_max], aspect='auto')
            if markers:
                plt.plot(self.exc[samples].ravel(), self.trans[samples].ravel(), 'k.', markersize=2)
            plt.ylim([self.trans_min, self.trans_max])
            plt.ylabel('$\mathit{\mu}^2$/a.u.')
        plt.savefig(fname+'.'+ext, bbox_inches='tight', dpi=dpi)
        if plot:
            plt.show()
        else:
            plt.ion()   #re-enables interactive mode

    def writegeoms(self, index=None):
        """Writes a file with indices of the selected representative geometries."""   #the output file contains either one-based indices or indices with associated weights

        indexstr = ''
        if index is not None:
            indexstr = '.' + str(index)
        outfile = self.get_name() + indexstr + '.geoms.txt'
        with open(outfile, "w") as f:
            for i in range(len(self.subsamples)):
                if self.sweights is None:
                    f.write('%s\n' % (self.subsamples[i]+1))   #writes the sample index (adding 1 for one-based indexing)
                else:
                    f.write('%s %s\n' % (self.subsamples[i]+1, self.sweights[i]))   #writes the sample index and its corresponding integer weight

    def select_geoms(self, input_file, output_file, atom_count):
        """
        Reads the input file and writes to output_file, inserting an extra blank line
        immediately before any line beginning with "Properties".

        Parameters:
          input_file (str): The original file (e.g. "full_geometries.xyz")
          output_file (str): The output file with blank lines inserted (e.g. "formatted_geometries.xyz")
          atom_count (str or int): The number of atoms in the target molecule.
        """
        with open(input_file, "r") as fin:
            lines = fin.readlines()

        with open(output_file, "w") as fout:
            geom_count = 1
            write_block = False

            for line in lines:
                # If the line begins with the atom count this is a new geometry.
                if line.startswith(str(atom_count)):
                    if geom_count in self.subsamples:
                        write_block = True
                    else:
                        write_block = False
                    geom_count += 1

                if write_block:
                    fout.write(line)

if __name__ == "__main__":   #main programme entry point
    random.seed(0)   #seed the random number generator for reproducibility
    start_time = time.time()   #records the start time for the overall execution
    options = read_cmd()   #parse command-line arguments
    if options.verbose:
        print("OPTIONS:")
        for option in vars(options):
            print(option, getattr(options, option))
        print()
        print("Number of CPUs on this machine:", cpu_count())

    geomReduction = GeomReduction(options.nsamples, options.nstates, options.subset, options.cycles, options.ncores,
                                  options.njobs, options.weighted, options.pdfcomp, options.intweights, options.verbose)   #creates an instance of GeomReduction with parameters from the command-line
    geomReduction.read_data(options.infile)   #reads the input molecular data
    geomReduction.reduce_geoms()   #performs the geometry reduction (optimisation) to select representative geometries
    
    #if options.verbose:
    print('INFO: wall time', round(time.time()-start_time), 's')   #prints the total wall time