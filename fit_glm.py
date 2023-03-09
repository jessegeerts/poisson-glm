from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.families.links import identity, log
import matplotlib.pyplot as plt
from simulate.simulate_spatial_cells import sim_place_cell_spikes


# load some example data
data = loadmat('notebooks/hipp_data')
xPos = data['xN'].squeeze() * 100  # convert to cm
yPos = data['yN'].squeeze() * 100
samp_freq = np.round(1 / np.diff(data['T'].squeeze())[0])
spikes = data['spikes2'].squeeze()

# get the position at the spikes
spike_xPos = xPos[spikes > 0]
spike_yPos = yPos[spikes > 0]

# plot spikes at position
plt.plot(xPos, yPos)
plt.scatter(spike_xPos, spike_yPos, color='r', zorder=20)
plt.show()

# Create a dataframe of predictors
predictors = pd.DataFrame(data={
    'Intercept': np.ones_like(xPos),
    'X': xPos,
    'Y': yPos,
    'X2': xPos ** 2,
    'Y2': yPos ** 2,
    'XY': xPos * yPos
})

# GLM model with Poisson family and identity link function
model = sm.GLM(spikes, predictors, family=Poisson())  # Create the model
results = model.fit()
betas = results.params

# Now, use the model to predict FR everywhere in the maze
xgrid, ygrid = np.meshgrid(np.arange(-100, 100, .05), np.arange(-100, 100, .05))
new_predictors = pd.DataFrame(data={
    'Intercept': np.ones_like(xgrid.flatten()),
    'X': xgrid.flatten(),
    'Y': ygrid.flatten(),
    'X2': xgrid.flatten() ** 2,
    'Y2': ygrid.flatten() ** 2,
    'XY': ygrid.flatten() * xgrid.flatten()
})
predicted_lambda = results.predict(new_predictors) * samp_freq  # * samp_freq to convert to Hz
predicted_lambda = np.reshape(predicted_lambda.to_numpy(), xgrid.shape)

# and plot the predicted firing rate as a contour
plt.contourf(xgrid, ygrid, predicted_lambda)
plt.colorbar()
plt.show()


# now given the same path let's simulate a neuron with a given field shape and location
field_centre = np.array([0, -40])
field_cov_mat = np.array([[50, 20], [20, 50]])

sim_spikes = sim_place_cell_spikes(field_centre, np.array([xPos, yPos]).T,
                                   max_rate=30, variance=field_cov_mat, sampling_frequency=samp_freq)

spike_xPos = xPos[sim_spikes > 0]
spike_yPos = yPos[sim_spikes > 0]

plt.plot(xPos, yPos)
plt.scatter(spike_xPos, spike_yPos, color='r', zorder=20)
plt.show()

predictors = pd.DataFrame(data={  # Create a dataframe of predictors
    'Intercept': np.ones_like(xPos),
    'X': xPos,
    'Y': yPos,
    'X2': xPos ** 2,
    'Y2': yPos ** 2,
    'XY': xPos * yPos
})

# GLM model with Poisson family and identity link function
model2 = sm.GLM(sim_spikes, predictors, family=Poisson())  # Create the model
results2 = model2.fit()

xgrid, ygrid = np.meshgrid(np.arange(-100, 100, 5), np.arange(-100, 100, 5))

new_predictors = pd.DataFrame(data={
    'Intercept': np.ones_like(xgrid.flatten()),
    'X': xgrid.flatten(),
    'Y': ygrid.flatten(),
    'X2': xgrid.flatten() ** 2,
    'Y2': ygrid.flatten() ** 2,
    'XY': ygrid.flatten() * xgrid.flatten()
})

predicted_lambda2 = results2.predict(new_predictors) * samp_freq

outside_maze = np.linalg.norm(np.array([xgrid.flatten(), ygrid.flatten()]), axis=0) > 100
predicted_lambda2[outside_maze] = np.nan
predicted_lambda2 = np.reshape(predicted_lambda2.to_numpy(), xgrid.shape)


fig, (ax1, ax2) = plt.subplots(1, 2)
plt.sca(ax1)
plt.plot(xPos, yPos)
plt.scatter(spike_xPos, spike_yPos, color='r', zorder=20)
plt.show()
plt.sca(ax2)
plt.contourf(xgrid, ygrid, predicted_lambda2)
plt.colorbar()
plt.show()


pred_frate = results2.predict(predictors) * samp_freq
pred_frate.plot()
plt.plot(sim_spikes)