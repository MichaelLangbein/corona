- get good literature value for percentage of NO2 from traffic
- write down interpretation to the calibrated parameter values
- connectivity: n2 as second parameter
- final model: do mcmc with pymc3, using the model:
	n(t) = n_sim(t, theta) + e(t)
		where theta ~ U(0, 1)
		and e ~ Poisson(n(t-1) - n_sim(t), inf...)
