"""Module for calculating normal mode frequencies and plotting IR spectrum of molecules.

This module provides classes and methods required for calculating normal mode frequencies and plotting IR spectrum of molecules whose coordinates are given as .xyz files.

example:
  ::
    
    if __name__ == `__main__`:
      wat=harmonic.MolOptimize('H2O',0)
      wat.freq_show()
      wat.IR_plot()

"""

from pyscf import gto,scf,dft
import numpy as np
import scipy.linalg as linalg
import scipy.constants as const
import math
import pymatgen.core as mg
from pyscf import hessian
from pyscf.geomopt.berny_solver import optimize
import matplotlib.pyplot as plt

Eh_to_J = const.physical_constants['hartree-joule relationship'][0]
"""float: Hartree energy in joules quoted from ``scipy.constants``."""

Bohr = const.physical_constants['Bohr radius'][0]
"""float: Bohr radius in meters quoted from ``scipy.constants``."""

amu = const.physical_constants['atomic mass constant'][0]
"""float: Atomic mass constants in kilograms quoted from ``scipy.constats``."""

m_to_cm = 100
"""float: metets in centimeters."""

class MolOptimize:
  """A class for optimizing molecules and showing normal mode frequencies and IR spectrum calculated using optimized structures.
    
    Args:
      name(str): Name of the xyz file.
      hess_method(int):if 0, hessian is calculated analytically by using `pyscf.hessian`. if not, hessian is calculated numerically.

    Attributes:
      name(str): Name of the xyz file.
      hess_method(int):if 0, hessian is calculated analytically by using `pyscf.hessian`. if not, hessian is calculated numerically.
      mol(pyscf.gto.mole.Mole):Optimized molecule.
      x_eq(numpy.adarray): Coordinates of atoms in optimized molecule in Bohrs.
      elem(list): Element symbols of atoms in the given molecule.
      mf(Energy): RKS object which holds results of executing ``kernel`` method.
      freq_list(FreqAnalysis): Frequencies of normal modes.
      mwc(FreqAnalysis): normal mode coordinates.
      IR_list(IR_intensity): IR absorption intensities corresponding to each normal mode.
      
  """

  def __init__(self,name,hess_method=0):
    self.name = name
    self.hess_method = hess_method
    self.mol = self.opt()
    self.x_eq = self.mol.atom_coords(unit='Bohr').flatten()
    self.elem = Molecule(self.name).mol_read()[0]
    self.mf = Energy(self.mol).ene()[1]
    self.freq_list = FreqAnalysis(self.mf,self.elem,self.x_eq,self.hess_method).freq_mode()[0]
    self.mwc = FreqAnalysis(self.mf,self.elem,self.x_eq,self.hess_method).freq_mode()[1]
    self.IR_list = IR_intensity(self.freq_list,self.mwc,self.elem,self.x_eq).intensity()

  def opt(self):
    """Optimize coordinates from .xyz file.

    Geometry optimization with DFT calculation by ``pyscf.geomopt.berny_solver``.

    Returns:
      pyscf.gto.mole.Mole: Optimized molecule.

    """
    mol = gto.Mole()
    mol.atom=self.name+'.xyz'
    mol.verbose = 0
    mol.basis = "ccpvdz"
    mol.unit='Angstrom'
    mol.build()
    mf = dft.RKS(mol)
    mf.xc='b3lyp'
    return optimize(mf,verbose=0)

  def freq_show(self):
    """Show the frequency wave number of normal mode vibration.
    
    Frequency wave number is calculated by ``FreqAnalysis.freq_mode``.

    """
    self.mwc = np.array(self.mwc).reshape((len(self.freq_list),len(self.elem),3))
    print("----------Normal Mode Frequency-----------")
    for i in range(len(self.freq_list)):
      print("")
      print(f"Frequency:{self.freq_list[i]:.5f} cm^-1")
      print("Normal Mode Coordinate")
      for j in range(int(len(self.x_eq)/3)):
        print(self.elem[j],f": x {self.mwc[i,j,0]:2.5f} y {self.mwc[i,j,1]:2.5f} z {self.mwc[i,j,2]:2.5f}")
      print("-------------------------------------------")
    print("")
    print("frequency wave number[cm^-1]")
    print(self.freq_list)

  def IR_plot(self):
    """Show the IR spectrum.
    """
    IRPlot(self.freq_list,self.IR_list,self.name).plot()

class MolBuild:
  """A class for building molecule, whose coordinates of atoms are displaced.

    Args:
      dx(numpy.ndarray): Displacement of atoms from optimized coordinates.
      elem(list): Element symbols of atoms.
      x_eq(numpy.ndarray): Coordinates of atoms in optimized molecule in Bohrs.

  """
  def __init__(self,dx,elem,x_eq):
    self.dx = dx
    self.elem = elem
    self.x_eq = x_eq

  def mol_build(self):
    """Build Mole object, whose coordinates of atoms are displaced by dx.

    Returns:
      pyscf.gto.mole.Mole: Displaced molecule.
    
    """
    x = self.x_eq + self.dx
    mol = gto.Mole()
    mol.atom = [[self.elem[i],x[3*i:3*(i+1)]] for i in range(len(self.elem))]
    mol.basis = 'ccpvdz'
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build()
    return mol

class Energy:
  """ A class for calculating energy of the molecule.
    
    Args:
      mol(pyscf.gto.mole.Mole): Molecule object.

  """

  def __init__(self,mol):
      self.mol = mol

  def ene(self):
    """DFT calculation.

    Returns:
      tuple: (E,mf)
        - **ene** (`numpy.float`) - Energy calculated with DFT method.
        - **mf** (`pyscf.dft.rks.RKS`) -RKS object which holds results of executing ``kernel`` method.
    """
    mf = dft.RKS(self.mol)
    mf.xc = 'b3lyp'
    E = mf.kernel()
    return E,mf

class DipoleMoment:
  """Dipole moment of the molecule.
    
    Args:
      mol(pyscf.gto.mole.Mole): Mole object.

    Attributes:
      mf(Energy.mf): RKS object which holds results of executing ``kernel`` method.
  """

  def __init__(self,mol):
    self.mol = mol
    self.mf = Energy(self.mol).ene()[1]

  def dp_moment(self):
    """DFT calculation.

    Returns:
      numpy.ndarray: Dipole moment.
    """
    return self.mf.dip_moment(verbose=0)


class Hess:
  """Hessian matrix of energy.

  Args:
    mf(Energy.mf): RKS object which holds results of executing ``kernel`` method.
    elem(list): Element symbols of atoms in the given molecule.
    x_eq(numpy.adarray): Coordinates of atoms in optimized molecule in Bohrs.
    hess_method(int):if 0, hessian is calculated analytically by using `pyscf.hessian`. if not, hessian is calculated numerically.
    
  """

  def __init__(self,mf,elem,x_eq,hess_method):
    self.hess_method=hess_method
    self.mf = mf
    self.elem = elem
    self.x_eq = x_eq
    self.hess_method = hess_method

  def analytical_hess(self):
    """Analytically calculate hessian metrix of energy.
    
    Returns:
      numpy.ndarray: Analytical hessian matrix in cartesian coordinates.
    """
    self.mf.kernel()
    H = self.mf.Hessian().kernel()
    return H

  def numerical_hess(self):
    """Numerically calculate hessia matrix.

    Returns:
      numpy.ndarray: Numerical hessian matrix in cartesian coordinates.
    """
    h = np.eye(len(self.x_eq),dtype=float)*1e-10/Bohr
    H = np.empty((len(self.x_eq),len(self.x_eq)))
    for i in range(len(self.x_eq)):
      for j in range(i+1):
        H[i,j] = (
        Energy(MolBuild((h[i]+h[j]),self.elem,self.x_eq).mol_build()).ene()[0]
        - Energy(MolBuild((-h[i]+h[j]),self.elem,self.x_eq).mol_build()).ene()[0]
        - Energy(MolBuild((h[i]-h[j]),self.elem,self.x_eq).mol_build()).ene()[0]
        + Energy(MolBuild((-h[i]-h[j]),self.elem,self.x_eq).mol_build()).ene()[0]
        )/ (4.0*h[i,i]*h[j,j])
        if i != j:
          H[j,i] = H[i,j]
    return H

  def mw_hessian(self):
    """Caluculate hessian matrix in mass weighted coordinates.

    Convert unit from Hatree /(amu Bohr*Bohr) to J /(kg m*m).

    Returns:
      numpy.ndarray: Hessian matrix in mass weighted coordinates.
    """
    if self.hess_method == 0:
      H = self.analytical_hess()
    else:
      H = self.numerical_hess()
    H = H.reshape(len(self.x_eq),len(self.x_eq))
    mw_H = np.eye(len(self.x_eq),len(self.x_eq))
    for i in range(len(self.x_eq)):
      for j in range(i + 1):
        mw_H[i,j] = H[i,j]*Eh_to_J/(Bohr**2)\
                 /(amu*math.sqrt(mg.Element(self.elem[int(i/3)]).atomic_mass * mg.Element(self.elem[int(j/3)]).atomic_mass))
        mw_H[j,i] = mw_H[i,j]
    return mw_H

class FreqAnalysis:
  """A class for calculating normal mode frequencies and corresponding normal mode coordinates.

  Args:
    mf(Energy.mf): RKS object which holds results of executing ``kernel`` method.
    elem(list): Element symbols of atoms in the given molecule.
    x_eq(numpy.adarray): Coordinates of atoms in optimized molecule in Bohrs.
    hess_method(int):if 0, hessian is calculated analytically by using `pyscf.hessian`. if not, hessian is calculated numerically.
  
  Attributes:
    hess(Hess.mw_hessian): Hessian matrix with mass weighted coordinates.
  """

  def __init__(self,mf,elem,x_eq,hess_method):
    self.mf = mf
    self.elem = elem
    self.x_eq = x_eq
    self.hess_method = hess_method
    self.hess = Hess(self.mf,self.elem,self.x_eq,self.hess_method).mw_hessian()

  def diagonalization(self):
    """Diagonalize the hessian matrix.
  
    Returns:
      tuple: (scipy.linalg.eig[0],scipy.linalg.eig[1])
        - **scipy.linalg.eig[0]** (`numpy.ndarray`) - Eigenvalues of the hessian matrix.
        - **scipy.linalg.eig[1]** (`pyscf.dft.rks.RKS`) - Eigenvectors of the hessian matrix.
    """
    return linalg.eig(self.hess)

  def freq_mode(self):
    """Calculate the normal mode frequencies and corresponding normal mode coordinates.

    Returns:
      tuple: (freq_list,mwc_list)
        - **freq_list** (list) - List including all the normal mode frequencies.
        - **mwc_list** (list) - List of normal mode coordinates corresponding to each frequency.
      
    """
    omega_sq = self.diagonalization()[0]
    mwc = self.diagonalization()[1]
    freq_list = []
    mwc_list = []
    for i in range(len(omega_sq)):
      if omega_sq[i] > 0:
        nu = abs(omega_sq[i])**0.5/(2*math.pi*const.c*m_to_cm)
        freq_list.append(nu)
        mwc_list.append(mwc[:,i])
    return freq_list,mwc_list

class IR_intensity:
  """A class for calculating IR abosorption intensity of molecule.

  Args:
    freq_list(FrequencyAnalysis.freq_mode): List including all the normal mode frequencies
    mwc(FrequencyAnalysis.freq_mode): List of normal mode coordinates corresponding to each frequency.
    elem(list): Element symbols of atoms in the given molecule.
    x_eq(numpy.adarray): Coordinates of atoms in optimized molecule in Bohrs.
  
  Attributes:
    mwc(numpy.ndarray): normal mode coordinates.
    delta(float): infinitesimal displacement.
    dip_moment_grad(numpy.ndarray): gradient of dipole moment.
  
  """

  def __init__(self,freq_list,mwc,elem,x_eq):
    self.freq_list = freq_list
    self.mwc = np.array(mwc)
    self.elem = elem
    self.x_eq = x_eq
    self.delta = 1.e-5
    self.dip_moment_grad = self.dm_grad()

  def dm_grad(self):
    """Calculate gradient of dipole moment.

    Returns:
      numpy.darray: gradient of dipole moment.

    """
    dip_moment_grad = np.empty((len(self.freq_list),3))
    dq = np.empty((len(self.freq_list),len(self.x_eq)))
    for i in range(len(self.freq_list)):
      for j in range(len(self.elem)):
        for k in range(3):
          dq[i] = self.mwc[i]*self.delta
      dip_moment_grad[i] = (DipoleMoment(MolBuild(dq[i],self.elem,self.x_eq).mol_build()).dp_moment()-DipoleMoment(MolBuild(-dq[i],self.elem,self.x_eq).mol_build()).dp_moment())/(self.delta*2)
    return dip_moment_grad

  def intensity(self):
    """Calculate IR absorption intensity.
  
    Returns:
      list: list of IR absorption intensity of each normal mode frequency.
    """
    IR_list = []
    for i in range(len(self.freq_list)):
      intense = pow(linalg.norm(self.dip_moment_grad[i]),2.0)
      IR_list.append(intense)
    return IR_list

class Oscillator:
  """A class for oscillating IR intensity.

  Args:
    freq_list(FreqAnalysis): List including all the normal mode frequencies.
    intensity(IR_intensity): Intensity of IR absorption.
  
  """

  def __init__(self,freq_list,intensity):
    self.freq_list = freq_list
    self.intensity = intensity

  def oscillator(self):
    """Oscillate IR intensity.

    Returns:
      tuple: (x,y)
        - **x** (numpy.darray): frequency wave number between 200 and 3500(cm^-1). x-axis of IR spectrum.
        - **y** (numpy.darray): IR intensity of each x. y-axis of IR spectrum.
    """
    x = np.arange(200,3500,10)
    y = np.zeros(len(x))
    w = 5
    for i in range(len(self.freq_list)):
      y = y + ((1/math.pi) * self.intensity[i] * 5)/ ((x - self.freq_list[i])**2 + w**2)
    y = 1.2*np.amax(y) -y
    return x,y


class IRPlot:
  """A class for plotting IR spectrum.

  Args:
    freq_list(FrequencyAnalysis.freq_mode): List including all the normal mode frequencies.
    intensity(IR_intensity): Intensity of IR absorption.
    name(str): name of .xyz file.

  Attributes:
    x(list): oscillated frequencies, horizontal axis
    y(list): IR intensity, vertical axis
  """

  def __init__(self,freq_list,intensity,name):
    self.freq_list = freq_list
    self.intensity = intensity
    self.name = name
    self.x,self.y = Oscillator(freq_list,intensity).oscillator()

  def plot(self):
    """Generate IR spectrum in a .png file.    
    """
    plt.plot(self.x,self.y)
    plt.xlabel('Frequency(cm^-1)')
    plt.ylabel('IR intensity')
    plt.title('IR Spectrum of ' + self.name)
    plt.savefig(self.name + "-IR.png")


class Molecule:
  """a class for reading .xyz file.
  
  Args:
    name(str): name of .xyz file.

  Attributes:
    atom_list(list): List of the element symbol of atoms.
    coord_list(list): List of the coordinates of each atom. 
  """

  def __init__(self,name):
    self.name=name+'.xyz'
    self.atom_list=self.mol_read()[0]
    self.coord_list=self.mol_read()[1]

  def mol_read(self):
    """Read .xyz file, and extract element symbol and coordinates of atoms.

    Returns:
      tuple: (atom_list,coord_list)
        - **atom_list** (list) - List of the element symbol of atoms.
        - **coord_list** (list) - List of the coordinates of each atom.
    """
    atom_list=[]
    coord_list=[]
    with open(self.name,'r') as file:
      atom_num=int(file.readline())
      file.readline()

      for _ in range(atom_num):
        line = file.readline().split()
        atom = line[0]
        atom_list.append(atom)
        x,y,z = line[1:4]
        coord_list.append((x,y,z))
    return atom_list,coord_list

