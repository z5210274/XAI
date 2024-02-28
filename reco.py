# Apply my TCAV coefficients for a decomposition starter
# Then continue with learning and convergence to find best reward decomposition function
# 1. Train a separate Q-Learning function that approximates our DQN but is decomposed into n reward types for explanation
#    Hopefully that Q-Function approximate converges to act like our DQN but is now HRA
# 1a. Using HRA create n reward functions (n = variable of TCAV) DrDQN???
#     which will hopefully represent each coefficent then train each one to sum of Hybrid Q-Value
#     then the weights of each agent will depict their importance in the hybrid Q-Value
#     The Hybrid Q-Value function will minimize loss of all n functions
# 1b. Each n function will represent a specific reward type by omitting/emphasising a specific environment variable
#     Each of our sub reward functions will omit a selection of Actions in the Q-Table and based on what is included
#     Or each column can be the y output each Move left + etc... = Theta
#     That will represent the reward type e.g: Move left + Move Right = Reposition vs Aim left + Aim Right = Reaim
# Reward Decomposition explanation works by comparing the tradeoffs between sub rewards
# 2. Compare the subrewards between each Action at a state to explain the tradeoff
# 3. See if the tradeoffs between subrewards align with the TCAV coefficients to see which category of decision making