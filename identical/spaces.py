import sympy as sp
from sympy import Matrix as M
from sympy.physics.quantum.tensorproduct import TensorProduct as tp
from sympy.physics.quantum.dagger import Dagger

import numpy as np

def bitwise_filter(data, selectors):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return ([d for d, s in zip(data, selectors) if s])

class FermionicSpace:
	def __init__ (self,dimension):
		self.dim = self.dimension = dimension
		self._One = M ([[1,0],[0,1]])
		self._C = M ([[0,1],[0,0]])
		self._A = M ([[0,0],[1,0]])
		self._Eta = M ([[1,0],[0,-1]])

		self._vac = M([0,1])
		self._occ = M([1,0]) # i.e., self._C*self._vac

		self.vac = tp (*tuple([self._vac]*(dimension)))
		self.C=[]
		for i in range (dimension):
			ones = [self._One]*(i)
			etas = [self._Eta]*(dimension-i-1)
			temp = tuple(ones+[self._C]+etas)
			temp_C = tp(*temp)
			self.C.append(temp_C)
		self.A = [Dagger(x) for x in self.C]
		self.N = [x[0]*x[1] for x in zip (self.C,self.A)]

		self.states = [self.vac]
		self.state_counter = 1

		self.phis = sp.symbols('phi1:%d'%(self.dim +1), commutative=False)
		self.fock_basis = [M([int(x) for x in format (y,'#0%db'%(self.dim+2))[-1:1:-1]]) for y in range (2**self.dim)]
		self.old_basis = [M([int(x) for x in format ((10**y),'#0%d'%(2**self.dim))]) for y in range (2**self.dim)]

		self.nA = np.array( [np.asarray(self.A[x].tolist(), dtype = float) for x in range (self.dim)] )
		self.nC = np.array( [np.asarray(self.C[x].tolist(), dtype = float) for x in range (self.dim)] )

		self.basis = self.new_basis(self.dim)
		self.qT_basis = self.basis[:]
		self.qT_basis.reverse()

		
	def new_basis (self, dim):
		basis = []
		powerdim = 2**dim
		for d in range (powerdim):
			p = np.identity(powerdim)
			for x in range (dim):
				if (self.fock_basis[d][x] != 0):
					p = np.dot(p,self.nC[x])
			basis.append(sp.Matrix(np.abs(p*self.vac, dtype = int)))

		return basis
	
	def add_fock_state(self,occupations):
		templist = reduce (lambda x,y: x*y, bitwise_filter(self.C, occupations))
		tempstate = templist*self.vac
		self.states.append(tempstate)
		self.state_counter += 1
		return tempstate

	def get_occupations (self, this_state):
		result = []
		x_list = []
		for i,x in enumerate(this_state[::-1]):
			if (x)!= 0:
				result.append (x*self.fock_basis[i])
				x_list.append (x)
		return result,x_list

	def print_slater(self, this_state):
		result = 0
		occupations, amplitudes = self.get_occupations (this_state)
		for i, occ in enumerate(occupations):
			this_phi = bitwise_filter(self.phis, occ)
			this_mat = M([this_phi]*len(this_phi))
			result += amplitudes[i]/sp.sqrt(len(this_phi))*sp.det (this_mat)
		return result
