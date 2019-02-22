"""
================================================================================
	HH_Test Model
================================================================================

	Author: Eric Johnson
	Date Created: Friday, February 22, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains the update function to implement the Hodgkin-Huxley
	neuron model.

================================================================================
================================================================================
"""
import numpy as np


def model(t, y, p):
	"""
	Hodgkin-Huxley Neuron Model

	DESRCIPTION:
		This function compute the derivatives of the the voltages and gating
		variables needed to model a neuron as described by Hodgkin and Huxley
		in 1952.  Specifically, the model has three ion currents: K, Na, and a
		leak current so that

		Cm dV/dt = I - gK(V - EK) - gNa(V - ENa) - gl(V - El)

		The conductances gK and gNa are voltage-dependent according to the 
		model

		gK = gKmax*n^4		and 	  gNa = gNamax*m^3*h

		and the gating variables n, m, and h are evolved according to the 
		equation

		dx/dt = (x0(V) - x)/tau(V),

		where

		x0(V) = alpha(V)/(alpha(V) + beta(V)), and
		tau(V) = 1/(alpha(V) + beta(V)),

		Using Dayan and Abbott we get that 

		alpha_n = 0.01(V + 55)/(1 - e^(-(V + 55)/10))
		beta_n = 0.125 e^(-(V+65)/80)

		alpha_m = 0.1(V + 40)/(1 - e^(-(V + 40)/10))
		beta_m = 4 e^(-(V + 65)/18)

		alpha_h = 0.07 e^(-(V + 65)/20)
		beta_h = 1/(1 + e^(-(V + 35)/10))

		In summary, the model has one voltage, two activation gating
		variables and one inactivation gating variable that are changing over
		time.

	INPUTS:
		t 		(float) Time at which derivative is computed.  In this case,
				this variable is not used, but most ODE solvers require the
				derivative function to take this input.

		y		(iterable of floats) Current values of voltages and gating
				variables.  As noted above, this model contains 4 variables
				which are assumed to be input in the following order:
					V:	Voltage
					n:	K activation variable
					m:	Na act. var.
					h:	Na inact. var.

		p		(dict) Dictionary of (25) model parameters:

					C:		(1 muF/mm^2)	Cell capacitance
					I:		(0 pA)			Input current

					Conductances...
					gK:		(0.36 mS/mm^2)	Max conductance of K channel
					gNa:	(1.2 mS/mm^2)	Max cond. of Na
					gl:		(.003 mS/mm^2)	Leak conductance

					Reversal Potentials...
					EK:		(-55 mV)		K Reversal Potential
					ENa:	(-77 mV)		Na Rev. Pot.
					El:		(50 mv)			Leak Rev. Pot.

					Kinetic parameters...
					a_n_a:		(0.01)			alpha_n factor
					a_n_Vhalf:	(55 mV)			alpha_n half-activation
					a_n_kx:		(10 mV)			alpha_n kinetic time

					b_n_a:		(0.125)			beta_n factor
					b_n_Vhalf:	(65 mV)			beta_n half-activation
					b_n_kx:		(80 mV)			beta_n kinetic time

					a_m_a:		(0.1)			alpha_m factor
					a_m_Vhalf:	(40 mV)			alpha_m half-activation
					a_m_kx:		(10 mV)			alpha_m kinetic time

					b_m_a:		(4.)			beta_m factor
					b_m_Vhalf:	(65 mV)			beta_m half-activation
					b_m_kx:		(18 mV)			beta_m kinetic time

					a_h_a:		(0.07)			alpha_h factor
					a_h_Vhalf:	(65 mV)			alpha_h half-activation
					a_h_kx:		(20 mV)			alpha_h kinetic time

					b_h_Vhalf:	(35 mV)			beta_h half-activation
					b_h_kx:		(10 mV)			beta_h kinetic time

		OUTPUTS:
			z		(float array) 4 numbers:
						dVdt
						dmdt
						dndt
						dhdt
	"""

###############################################################################
# 	Initialize Temporary Variables
###############################################################################

	[V, m, n, h] = y

###############################################################################
# 	Calculate Gating Variables
###############################################################################
	
	# Potassium Activation
	alpha_n = x_inf(V, p['a_n_a'], p['a_n_Vhalf'], p['a_n_kx'])
	beta_n = boltz(V,  p['b_n_a'], p['b_n_Vhalf'], p['b_n_kx'])

	# Sodium Activation
	alpha_m = x_inf(V, p['a_m_a'], p['a_m_Vhalf'], p['a_m_kx'])
	beta_m = boltz(V,  p['b_m_a'], p['b_m_Vhalf'], p['b_m_kx'])

	# Sodium Inactivation
	alpha_h = boltz(V,  p['a_h_a'], p['a_h_Vhalf'], p['a_h_kx'])
	beta_h = tau_h(V, p['b_h_Vhalf'], p['b_h_kx'])

	tau_n = 1./(alpha_n + beta_n)
	n_Inf = alpha_n*tau_n

	tau_m = 1./(alpha_m + beta_m)
	m_Inf = alpha_m*tau_m

	tau_h = 1./(alpha_h + beta_h)
	h_Inf = alpha_h*tau_h

###############################################################################
# 	Calculate Channel Currents
###############################################################################

	IK = p['gK']*n**4*(V - p['EK'])

	INa = p['gNa']*m**3*h*(V - p['ENa'])

	Il = p['gl']*(V - p['El'])

###############################################################################
# 	Calculate Derivatives to V, n, m, and h
###############################################################################

	z = np.zeros((4))

	z[0] = -1./p['C']*(IK + INa + Il - p['I'])

	z[1] = (n_inf - n)/tau_n
	z[2] = (m_inf - m)/tau_m
	z[3] = (h_inf - h)/tau_h

	return z


def boltz(V, a, Vhalf, kx):
	return a*np.exp(-(V - Vhalf)/kx)

def x_inf(V, a, Vhalf, kx):
	return a*(V - Vhalf)/(1-boltz(V, 1, Vhalf, kx))

def tau_h(V, Vhalf, kx):
	return 1./(1 + boltz(V, 1, Vhalf, kx))