
# Background

This repository contains the notes and code for investigating the charge dynamics of a driven molecular dimer in an open-quantum systems treatment. The dimer is modeled as two interacting dipoles each strongly coupled to an independent bath of phonons. At this stage, the ambient, incoherent electromagnetic field couples weakly to the delocalised vibronic states.
There is:
- a central reaction coordinate Liouvillian builder called dimer_UD_liouv.py. It creates the master equation for the strongly coupled vibrations.
- a module which deals with all of the different types of incoherent optical driving called dimer_driving_liouv.py.
- a module with several different types of checks. Convergence, non-secularity etc.
- a plotting module which takes in the other two and plots graphs of the dynamics, coherences and spectra. This could be extended into an ipython notebook as well.
- a directory with all of the accompanying notes and figures for the investigation, read Vibronic_incoherent_notes.pdf to get some more physical insight.

# Requirements

All the python files are written in Python 2.7. The modules will need to be at least:
- Qutip >=3.1.0
- Numpy
- Scipy
- Matplotlib

# Getting started
- Clone the repo and install the python dependencies
- Open ME_plotting.py, choose some parameters and run it.
- Check in the Notes/Images folder for default plots of dynamics and spectra, or alternatively use the data files in DATA to plot your own.

# Bugs
- Electronic Lindblad does not work at the moment. Agrees with neither the other two theories or the non-canonical thermal state (not accounting for RC bath).


# To do:
- Incorporate excitation number restriction into the non-secular and secular equations.
- Test across many parameter regimes.
- Create a module which loads up datafiles and turns them into good pandas dataframes

# Notes:
- I have tested the code and it agrees with the TLS where V=0, mu=0 and everything else is the same.
