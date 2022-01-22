
# Overview
This project intends to run and build system to get a better reward from the CarRacingv0
environment game by using the Reinforcement Learning with Convolutional
Reservoir Computing (RCRC) model. The RCRC model which combine the
Convolutional Neural Networks (CNN) and Echo State Networks (ESN) model with a
fixed random weight. 


Students are required to demonstrate an ability to run the code
and setup the game environment in addition to change the parameters by experience
and optimize them to have a better performance and getting more rewards from the
Car Racing game.
This is the modified code from the original (https://github.com/Narsil/rl-baselines) and we added the Bayesian optimization to optimize the following parameters:
<ul>
  <li>Spectral radius in Echo state network </li>
  <li>The size of the Conv and ESN output weights</li>
  <li>The number of epochs</li>
</ul>

# Running the code
The Car_Racing.ipynb file is a colab file to run and install all the requirements needed to run this code


# Note
We just modified and added the Bayesian Optimization for the rcrc.py code 
