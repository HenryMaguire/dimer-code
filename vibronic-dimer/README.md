
# Background

This repository contains the notes and code for investigating the charge dynamics of a driven molecular dimer in an open-quantum systems treatment. The dimer is modeled as two interacting dipoles each strongly coupled to an independent bath of phonons. At this stage, the ambient, incoherent electromagnetic field couples weakly to the delocalised vibronic states.
There is:
- a central reaction coordinate Liouvillian builder called `dimer_phonons.py`. It creates the master equation for the strongly coupled vibrations/phonons.
- a module which deals with all of the different types of incoherent optical driving called `dimer_optical.py`.
- a module with several different types of checks such as steady state/dynamics convergence. In here there are also functions to generate arrays of steady-state density matrices for looking at dependence on certain parameters. Data structures are saved into the `DATA` directory.
- a plotting module with all the visualisation functionality in one place - this includes plotting steady state relationships.
- a directory with all of the accompanying notes and figures for the investigation, read `Vibronic_incoherent_notes.pdf` to get some more physical insight.
- `dimer_dynamics.py` which defines a physical system and the various baths, calls upon the above modules to calculate dynamics/steady states and plot them.
- There is also `electronic_lindblad.py` which is a work in progress. This uses a different version of `dimer_optical.py` (ignores system+RC eigenstructure when defining the optical Liouvillian).

# Requirements

All the python files are written in Python 2.7. The modules will need to be at least:
- Qutip >=3.1.0
- Numpy
- Scipy
- Matplotlib

Building the Liouvillian superoperators and calculating dynamics is parallelised. Change the variable `num_cpus` in the code to as many physical cores your machine can afford.

# Getting started
- Clone the repo and install the python dependencies
- Open ME_plotting.py, choose some parameters and run it.
- Check in the Notes/Images folder for default plots of dynamics and spectra, or alternatively use the data files in DATA to plot your own.

# Bugs
- Electronic Lindblad does not work at the moment. Agrees with neither the other two theories or the non-canonical thermal state (not accounting for RC bath).


# To do:
- Test across many parameter regimes.

# Notes:
- I have tested the code and it agrees with the TLS where V=0, mu=0 and everything else is the same.
