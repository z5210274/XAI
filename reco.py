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

import os
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_reco():
    filename = './q_values.csv'
    filename2 = './tcav.csv'

    check_file = os.path.isfile(filename)
    print("q_values.csv exists: " + str(check_file))
    check_file2 = os.path.isfile(filename2)
    print("tcav.csv exists: " + str(check_file2))

    df_tcav = pd.read_csv(filename2)
    df_q = pd.read_csv(filename)

############################################### Move Right #############################################

    x_mr = df_q.drop('Move_right', axis = 1)
    y_mr = df_q['Move_right']
    x_mrdata = x_mr.to_numpy()
    y_mrdata = y_mr.to_numpy()

    x_mrtrain,x_mrtest,y_mrtrain,y_mrtest = train_test_split(x_mrdata,y_mrdata,test_size=0.80)

    mr_model = LinearRegression()
    mr_model.fit(x_mrtrain,y_mrtrain)

    y_mrpredict = mr_model.predict(x_mrtest)

    # The coefficients
    print("Coefficients: \n", mr_model.coef_)
    # The mean squared error
    print("Mean squared error: " + str(mean_absolute_error(y_mrtest, y_mrpredict)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_mrtest, y_mrpredict))

############################################### Move Left #############################################

    x_ml = df_q.drop('Move_left', axis = 1)
    y_ml = df_q['Move_left']
    x_mldata = x_ml.to_numpy()
    y_mldata = y_ml.to_numpy()

    x_mltrain,x_mltest,y_mltrain,y_mltest = train_test_split(x_mldata,y_mldata,test_size=0.80)

    ml_model = LinearRegression()
    ml_model.fit(x_mltrain,y_mltrain)

    y_mlpredict = ml_model.predict(x_mltest)

    # The coefficients
    print("Coefficients: \n", ml_model.coef_)
    # The mean squared error
    print("Mean squared error: " + str(mean_absolute_error(y_mltest, y_mlpredict)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_mltest, y_mlpredict))

############################################### Aim Right #############################################

    x_ar = df_q.drop('Aim_right', axis = 1)
    y_ar = df_q['Aim_right']
    x_ardata = x_ar.to_numpy()
    y_ardata = y_ar.to_numpy()

    x_artrain,x_artest,y_artrain,y_artest = train_test_split(x_ardata,y_ardata,test_size=0.80)

    ar_model = LinearRegression()
    ar_model.fit(x_artrain,y_artrain)

    y_arpredict = ar_model.predict(x_artest)

    # The coefficients
    print("Coefficients: \n", ar_model.coef_)
    # The mean squared error
    print("Mean squared error: " + str(mean_absolute_error(y_artest, y_arpredict)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_artest, y_arpredict))

############################################### Aim Left #############################################

    x_al = df_q.drop('Aim_left', axis = 1)
    y_al = df_q['Aim_left']
    x_aldata = x_al.to_numpy()
    y_aldata = y_al.to_numpy()

    x_altrain,x_altest,y_altrain,y_altest = train_test_split(x_aldata,y_aldata,test_size=0.80)

    al_model = LinearRegression()
    al_model.fit(x_altrain,y_altrain)

    y_alpredict = al_model.predict(x_altest)

    # The coefficients
    print("Coefficients: \n", al_model.coef_)
    # The mean squared error
    print("Mean squared error: " + str(mean_absolute_error(y_altest, y_alpredict)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_altest, y_alpredict))

############################################### Shoot #############################################

    x_s = df_q.drop('Shoot', axis = 1)
    y_s = df_q['Shoot']
    x_sdata = x_s.to_numpy()
    y_sdata = y_s.to_numpy()

    x_strain,x_stest,y_strain,y_stest = train_test_split(x_sdata,y_sdata,test_size=0.80)

    s_model = LinearRegression()
    s_model.fit(x_strain,y_strain)

    y_spredict = s_model.predict(x_stest)

    # The coefficients
    print("Coefficients: \n", s_model.coef_)
    # The mean squared error
    print("Mean squared error: " + str(mean_absolute_error(y_stest, y_spredict)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_stest, y_spredict))

############################################### Nothing #############################################

    x_n = df_q.drop('Nothing', axis = 1)
    y_n = df_q['Nothing']
    x_ndata = x_n.to_numpy()
    y_ndata = y_n.to_numpy()

    x_ntrain,x_ntest,y_ntrain,y_ntest = train_test_split(x_ndata,y_ndata,test_size=0.80)

    n_model = LinearRegression()
    n_model.fit(x_ntrain,y_ntrain)

    y_npredict = n_model.predict(x_ntest)

    # The coefficients
    print("Coefficients: \n", n_model.coef_)
    # The mean squared error
    print("Mean squared error: " + str(mean_absolute_error(y_ntest, y_npredict)))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_ntest, y_npredict))

############################################### HRA #############################################



run_reco()