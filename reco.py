# Apply my TCAV coefficients for a decomposition starter
# Then continue with learning and convergence to find best reward decomposition function
# 1. Figure out a reward function that decomposes the rewards into sub rewards and classes them
# 1a. Using HRA create n reward functions (n = variable of TCAV)
#     which will hopefully represent each coefficent then train each one to sum of Hybrid Q-Value
#     then the weights of each agent will depict their importance in the hybrid Q-Value
#     The Hybrid Q-Value function will minimize loss of all n functions
# 1b. Each n function will represent a specific reward type by omitting/emphasising a specific environment variable
# Reward Decomposition explanation works by comparing the tradeoffs between sub rewards
# 2. Compare the subrewards between each Action at a state to explain the tradeoff
# 3. See if the tradeoffs between subrewards align with the TCAV coefficients to see which category of decision making