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
    mr = False

    x_maxmrdata = []
    y_maxmrdata = []

    for i in range(0,len(x_mrdata)-1):
        if (y_mrdata[i] >= max(x_mrdata[i])):
            x_maxmrdata.append(x_mrdata[i])
            y_maxmrdata.append(y_mrdata[i])

    if (len(x_maxmrdata) > 0):
        mr = True
        x_mrtrain,x_mrtest,y_mrtrain,y_mrtest = train_test_split(x_maxmrdata,y_maxmrdata,test_size=0.80)

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
    ml = False

    x_maxmldata = []
    y_maxmldata = []

    for i in range(0,len(x_mldata)-1):
        if (y_mldata[i] >= max(x_mldata[i])):
            x_maxmldata.append(x_mldata[i])
            y_maxmldata.append(y_mldata[i])

    if (len(x_maxmldata) > 0):
        ml = True
        x_mltrain,x_mltest,y_mltrain,y_mltest = train_test_split(x_maxmldata,y_maxmldata,test_size=0.80)

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
    ar = False

    x_maxardata = []
    y_maxardata = []

    for i in range(0,len(x_ardata)-1):
        if (y_ardata[i] >= max(x_ardata[i])):
            x_maxardata.append(x_ardata[i])
            y_maxardata.append(y_ardata[i])

    if (len(x_maxardata) > 0):
        ar = True
        x_artrain,x_artest,y_artrain,y_artest = train_test_split(x_maxardata,y_maxardata,test_size=0.80)

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
    al = False

    x_maxaldata = []
    y_maxaldata = []

    for i in range(0,len(x_aldata)-1):
        if (y_aldata[i] >= max(x_aldata[i])):
            x_maxaldata.append(x_aldata[i])
            y_maxaldata.append(y_aldata[i])

    if (len(x_maxaldata) > 0):
        al = True
        x_altrain,x_altest,y_altrain,y_altest = train_test_split(x_maxaldata,y_maxaldata,test_size=0.80)

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
    s = False

    x_maxsdata = []
    y_maxsdata = []

    for i in range(0,len(x_sdata)-1):
        if (y_sdata[i] >= max(x_sdata[i])):
            x_maxsdata.append(x_sdata[i])
            y_maxsdata.append(y_sdata[i])

    if (len(x_maxsdata) > 0):
        s = True
        x_strain,x_stest,y_strain,y_stest = train_test_split(x_maxsdata,y_maxsdata,test_size=0.80)

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
    n = False

    x_maxndata = []
    y_maxndata = []

    for i in range(0,len(x_ndata)-1):
        if (y_ndata[i] >= max(x_ndata[i])):
            x_maxndata.append(x_ndata[i])
            y_maxndata.append(y_ndata[i])

    if (len(x_maxndata) > 0):
        n = True
        x_ntrain,x_ntest,y_ntrain,y_ntest = train_test_split(x_maxndata,y_maxndata,test_size=0.80)

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

    # Combine all 6 models into one model and use weights of each model to see which is strongest
    # Using Q-Value table of decision, predict for each model then that is my q-value table of significance

    q_table = [90.19,91.34,89.79,90.85,91.39,90.29]
    move_right = 0
    move_left = 0
    aim_right = 0
    aim_left = 0
    shoot = 0
    nothing = 0

    if (mr == True):
        mr_table = [q_table[:0] + q_table[1 :]]
        move_right = mr_model.predict(mr_table)[0]

    if (ml == True):
        ml_table = [q_table[:1] + q_table[2 :]]
        move_left = ml_model.predict(ml_table)[0]

    if (ar == True):
        ar_table = [q_table[:2] + q_table[3 :]]
        aim_right = ar_model.predict(ar_table)[0]

    if (al == True):
        al_table = [q_table[:3] + q_table[4 :]]
        aim_left = al_model.predict(al_table)[0]

    if (s == True):
        s_table = [q_table[:4] + q_table[5 :]]
        shoot = s_model.predict(s_table)[0]

    if (n == True):
        n_table = [q_table[:5] + q_table[6 :]]
        nothing = n_model.predict(n_table)[0]

    hra = [move_right,move_left,aim_right,aim_left,shoot,nothing]
    print(hra)
    reco_diff = []
    for i in range(0,len(q_table)):
        reco_diff.append(abs(hra[i]-q_table[i]))

    print(reco_diff)

run_reco()