import numpy as np
import QuantLib as ql
from dataclasses import dataclass
 
def basis_functions(X, k):
  """
  Generate basis functions for the regression.
 
  Parameters:
 	X : array
 	Asset prices
 	k : int
 	Number of basis functions
 	Returns:
 	numpy.ndarray
 	Basis functions evaluated at X
 	"""
 	if k == 1:
 		A = np.vstack([np.ones(X.shape), 1-X]).T
 	elif k == 2:
 		A = np.vstack([np.ones(X.shape), 1-X, 0.5*(2-4*X+X**2)]).T
 	elif k == 3:
 		A = np.vstack([np.ones(X.shape), 1-X, 0.5*(2-4*X+X**2),
 				1/6*(6-18*X+9*X**2-X**3)]).T
 	elif k == 4:
 		A = np.vstack([np.ones(X.shape), 1-X, 0.5*(2-4*X+X**2),
 				1/6*(6-18*X+9*X**2-X**3),
 				1/24*(24-96*X+72*X**2-16*X**3+X**4)]).T
 	elif k == 5:
 		A = np.vstack([np.ones(X.shape), 1-X, 0.5*(2-4*X+X**2),
 				1/6*(6-18*X+9*X**2-X**3),
 				1/24*(24-96*X+72*X**2-16*X**3+X**4),
 				1/120*(120-600*X+600*X**2-200*X**3+25*X**4-X**5)]).T
 	else:
 		raise ValueError('Too many basis functions requested')
 	return A	

@dataclass
class Option:
  """
 	Representation of an option derivative
 	"""
 
 	s0: float
 	T: int
 	K: int
 	v0: float = None
 	call: bool = True
 
 	def payoff(self, s: np.ndarray) -> np.ndarray:
 		payoff = np.maximum(0, s - self.K) if self.call 
 			else np.maximum(0, self.K - s)
 		return payoff	
 		
def LSM(option:Option, r:float, sigma:float, N:int, M:int, k=3) -> float:
 	"""
 	Price an American option using the Longstaff-Schwartz method.
 		
 	Parameters:
 	option : Option
 	An instance of the Option dataclass.
 	r : float
 	Risk-free interest rate.
 	sigma : float
 	Volatility of the underlying asset.
 	N : int
 	Number of time steps.
 	M : int
 	Number of simulated paths.

	Returns:
 	float
 	Estimated price of the American option.
 	"""
 	# Set up the parameters for the process
 	dt = option.T / N
 	discount_factor = np.exp(-r * dt)
 			
 	# Setup QuantLib objects
 	spot_handle = ql.QuoteHandle(ql.SimpleQuote(option.s0))
 	flat_ts = ql.YieldTermStructureHandle(
 				ql.FlatForward(0, ql.NullCalendar(), 
 				 ql.QuoteHandle(ql.SimpleQuote(r)), ql.Actual365Fixed()))
 	vol_ts = ql.BlackVolTermStructureHandle(
 			   ql.BlackConstantVol(0, ql.NullCalendar(), 
 			    ql.QuoteHandle(ql.SimpleQuote(sigma)),ql.Actual365Fixed()))
 	process = ql.GeneralizedBlackScholesProcess(
 				spot_handle, flat_ts, flat_ts, vol_ts)
 			
 	# Simulate paths
 	rng = ql.GaussianRandomSequenceGenerator(
 			ql.UniformRandomSequenceGenerator(
 			 N, ql.UniformRandomGenerator()))
 	seq = ql.GaussianPathGenerator(process, option.T, N, rng, False)
 			
 	# Generate paths
 	paths = np.zeros((M, N + 1))
 	for i in range(M):
 		sample_path = seq.next()
 		values = sample_path.value()
 		path = np.array([values[j] for j in range(len(values))])
 		paths[i, :] = path
 			
 	# Calculate the payoff at each node
 	payoffs = option.payoff(paths[:, -1])

	# Perform regression to estimate continuation value
 	for t in range(N - 1, 0, -1):
 		X = paths[:, t]
 		Y = discount_factor * payoffs
 		A = basis_functions(X, k)
 		beta = np.linalg.lstsq(A, Y, rcond=None)[0]
 		continuation_value = np.dot(A, beta)
 		exercise_value = option.payoff(X)
 		payoffs = np.where(exercise_value > continuation_value, 
 					exercise_value, discount_factor * payoffs)
 		
 	# Discount the payoff back to present value
 	option_price = np.mean(discount_factor * payoffs)
 			
