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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

def run_reco_linear():
    filename = './q_values.csv'
    filename2 = './tcav.csv'
    filename3 = './state.csv'

    check_file = os.path.isfile(filename)
    print("q_values.csv exists: " + str(check_file))
    check_file2 = os.path.isfile(filename2)
    print("tcav.csv exists: " + str(check_file2))
    check_file3 = os.path.isfile(filename3)
    print("state.csv exists: " + str(check_file3))

    df_tcav = pd.read_csv(filename2)
    df_q = pd.read_csv(filename)
    df_state = pd.read_csv(filename3)

############################################### Move Right #############################################

    x_mr = df_state
    y2_mr = df_q.drop('Move_right', axis = 1)
    y_mr = df_q['Move_right']
    x_mrdata = x_mr.to_numpy()
    y2_mrdata = y2_mr.to_numpy()
    y_mrdata = y_mr.to_numpy()

    x_mrtrain,x_mrtest,y_mrtrain,y_mrtest = train_test_split(x_mrdata,y_mrdata,test_size=0.80)

    mr_model = LinearRegression()
    mr_model.fit(x_mrtrain,y_mrtrain)

    y_mrpredict = mr_model.predict(x_mrtest)

    mr_q = mr_model.predict(x_mrdata)
    df_mr = pd.DataFrame(mr_q, columns=['Move_right'])
    x_mr.insert(0,'Move_right',df_mr)
    x_mrdata = x_mr.to_numpy()

    multi_mr = MultiOutputRegressor(Ridge(random_state=123)).fit(x_mrdata, y2_mrdata)
    mr_anti = multi_mr.predict(x_mrdata)
    df_mr_anti = pd.DataFrame(mr_anti, columns=['Move_left','Aim_right','Aim_left','Shoot','Nothing'])
    df_mr_anti.insert(0,'Move_right',df_mr)
    mr_rewardq = df_mr_anti.to_numpy()

    # The coefficients
    #print("Coefficients: \n", mr_model.coef_)
    # The mean squared error
    #print("Mean squared error: " + str(mean_absolute_error(y_mrtest, y_mrpredict)))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_mrtest, y_mrpredict))

############################################### Move Left #############################################

    x_ml = df_state
    y2_ml = df_q.drop('Move_left', axis = 1)
    y_ml = df_q['Move_left']
    x_mldata = x_ml.to_numpy()
    y2_mldata = y2_ml.to_numpy()
    y_mldata = y_ml.to_numpy()

    x_mltrain,x_mltest,y_mltrain,y_mltest = train_test_split(x_mldata,y_mldata,test_size=0.80)

    ml_model = LinearRegression()
    ml_model.fit(x_mltrain,y_mltrain)

    y_mlpredict = ml_model.predict(x_mltest)

    ml_q = ml_model.predict(x_mldata)
    df_ml = pd.DataFrame(ml_q, columns=['Move_left'])
    x_ml.insert(0,'Move_left',df_ml)
    x_mldata = x_ml.to_numpy()

    multi_ml = MultiOutputRegressor(Ridge(random_state=123)).fit(x_mldata, y2_mldata)
    ml_anti = multi_ml.predict(x_mldata)
    df_ml_anti = pd.DataFrame(ml_anti, columns=['Move_right','Aim_right','Aim_left','Shoot','Nothing'])
    df_ml_anti.insert(1,'Move_left',df_ml)
    ml_rewardq = df_ml_anti.to_numpy()

    # The coefficients
    #print("Coefficients: \n", ml_model.coef_)
    # The mean squared error
    #print("Mean squared error: " + str(mean_absolute_error(y_mltest, y_mlpredict)))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_mltest, y_mlpredict))

############################################### Aim Right #############################################

    x_ar = df_state
    y2_ar = df_q.drop('Aim_right', axis = 1)
    y_ar = df_q['Aim_right']
    x_ardata = x_ar.to_numpy()
    y2_ardata = y2_ar.to_numpy()
    y_ardata = y_ar.to_numpy()

    x_artrain,x_artest,y_artrain,y_artest = train_test_split(x_ardata,y_ardata,test_size=0.80)

    ar_model = LinearRegression()
    ar_model.fit(x_artrain,y_artrain)

    y_arpredict = ar_model.predict(x_artest)

    ar_q = ar_model.predict(x_ardata)
    df_ar = pd.DataFrame(ar_q, columns=['Aim_right'])
    x_ar.insert(0,'Aim_right',df_ar)
    x_ardata = x_ar.to_numpy()

    multi_ar = MultiOutputRegressor(Ridge(random_state=123)).fit(x_ardata, y2_ardata)
    ar_anti = multi_ar.predict(x_ardata)
    df_ar_anti = pd.DataFrame(ar_anti, columns=['Move_right','Move_left','Aim_left','Shoot','Nothing'])
    df_ar_anti.insert(2,'Aim_right',df_ar)
    ar_rewardq = df_ar_anti.to_numpy()

    # The coefficients
    #print("Coefficients: \n", ar_model.coef_)
    # The mean squared error
    #print("Mean squared error: " + str(mean_absolute_error(y_artest, y_arpredict)))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_artest, y_arpredict))

############################################### Aim Left #############################################

    x_al = df_state
    y2_al = df_q.drop('Aim_left', axis = 1)
    y_al = df_q['Aim_left']
    x_aldata = x_al.to_numpy()
    y2_aldata = y2_al.to_numpy()
    y_aldata = y_al.to_numpy()

    x_altrain,x_altest,y_altrain,y_altest = train_test_split(x_aldata,y_aldata,test_size=0.80)

    al_model = LinearRegression()
    al_model.fit(x_altrain,y_altrain)

    y_alpredict = al_model.predict(x_altest)

    al_q = al_model.predict(x_aldata)
    df_al = pd.DataFrame(al_q, columns=['Aim_left'])
    x_al.insert(0,'Aim_left',df_al)
    x_aldata = x_al.to_numpy()

    multi_al = MultiOutputRegressor(Ridge(random_state=123)).fit(x_aldata, y2_aldata)
    al_anti = multi_al.predict(x_aldata)
    df_al_anti = pd.DataFrame(al_anti, columns=['Move_right','Move_left','Aim_right','Shoot','Nothing'])
    df_al_anti.insert(3,'Aim_left',df_al)
    al_rewardq = df_al_anti.to_numpy()

    # The coefficients
    #print("Coefficients: \n", al_model.coef_)
    # The mean squared error
    #print("Mean squared error: " + str(mean_absolute_error(y_altest, y_alpredict)))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_altest, y_alpredict))

############################################### Shoot #############################################

    x_s = df_state
    y2_s = df_q.drop('Shoot', axis = 1)
    y_s = df_q['Shoot']
    x_sdata = x_s.to_numpy()
    y2_sdata = y2_s.to_numpy()
    y_sdata = y_s.to_numpy()

    x_strain,x_stest,y_strain,y_stest = train_test_split(x_sdata,y_sdata,test_size=0.80)

    s_model = LinearRegression()
    s_model.fit(x_strain,y_strain)

    y_spredict = s_model.predict(x_stest)

    s_q = s_model.predict(x_sdata)
    df_s = pd.DataFrame(s_q, columns=['Shoot'])
    x_s.insert(0,'Shoot',df_s)
    x_sdata = x_s.to_numpy()

    multi_s = MultiOutputRegressor(Ridge(random_state=123)).fit(x_sdata, y2_sdata)
    s_anti = multi_s.predict(x_sdata)
    df_s_anti = pd.DataFrame(s_anti, columns=['Move_right','Move_left','Aim_right','Aim_left','Nothing'])
    df_s_anti.insert(4,'Shoot',df_s)
    s_rewardq = df_s_anti.to_numpy()

    # The coefficients
    #print("Coefficients: \n", s_model.coef_)
    # The mean squared error
    #print("Mean squared error: " + str(mean_absolute_error(y_stest, y_spredict)))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_stest, y_spredict))

############################################### Nothing #############################################

    x_n = df_state
    y2_n = df_q.drop('Nothing', axis = 1)
    y_n = df_q['Nothing']
    x_ndata = x_n.to_numpy()
    y2_ndata = y2_n.to_numpy()
    y_ndata = y_n.to_numpy()

    x_ntrain,x_ntest,y_ntrain,y_ntest = train_test_split(x_ndata,y_ndata,test_size=0.80)

    n_model = LinearRegression()
    n_model.fit(x_ntrain,y_ntrain)

    y_npredict = n_model.predict(x_ntest)

    n_q = n_model.predict(x_ndata)
    df_n = pd.DataFrame(n_q, columns=['Nothing'])
    x_n.insert(0,'Nothing',df_n)
    x_ndata = x_n.to_numpy()

    multi_n = MultiOutputRegressor(Ridge(random_state=123)).fit(x_ndata, y2_ndata)
    n_anti = multi_n.predict(x_ndata)
    df_n_anti = pd.DataFrame(n_anti, columns=['Move_right','Move_left','Aim_right','Aim_left','Shoot'])
    df_n_anti.insert(5,'Nothing',df_n)
    n_rewardq = df_n_anti.to_numpy()

    # The coefficients
    #print("Coefficients: \n", n_model.coef_)
    # The mean squared error
    #print("Mean squared error: " + str(mean_absolute_error(y_ntest, y_npredict)))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_ntest, y_npredict))

############################################### HRA #############################################

    # Combine all 6 models into one model and use weights of each model to see which is strongest
    # Using Q-Value table of decision, predict for each model then that is my q-value table of significance

    q_table = [1.427055,0.5298118,0.97687083,1.0144249,0.09637004,0.99098486] # Row 11

    hra_list = [mr_rewardq[9], ml_rewardq[9], ar_rewardq[9], al_rewardq[9], s_rewardq[9], n_rewardq[9]]

    print("Differences between HRA Q-Values and actual:")
    for i in range(0,len(hra_list)):
        diff = 0
        for j in range(0,len(q_table)):
            diff += abs(hra_list[i][j]-q_table[j])
        print(diff)


    '''move_right = 0
    move_left = 0
    aim_right = 0
    aim_left = 0
    shoot = 0
    nothing = 0

    hra = [move_right,move_left,aim_right,aim_left,shoot,nothing]
    print("Predicted")
    print(hra)
    print("Actual")
    print(q_table)
    reco_diff = []
    for i in range(0,len(q_table)):
        reco_diff.append(abs(hra[i]-q_table[i]))
    print("Difference")
    print(reco_diff)'''

# More chosen moves means more accurate learning meaning more significance
# a) Accuracy of prediction
# b) Significance of prediction by tradeoff of reward types
# True because if it is constantly optimal then it is significant
# When explaining decision that move isn't chosen but most significant
# Case: Shoot was taken but huge prediction difference
# Explanation, Confidence in Theta and accuracy in aiming over movement, which is good
# As taking the shot should be based off aiming over movement


# NEW CURRENT APPROACH
# Architecture for HRA, using gamestate, predict q-value of each action meaning 6 different networks
# Train NN with all gamestates, with predictor of q-value for each action
# Lot of data

# Then using each 6 trained NN, again use game state and substited predicted action q-value to predict other 5 q-values
# Then merge the 6 trained NN with all their Q-values to HRA a q-value predictor of action to explain
# Then you can reco compare the significance of each NN compared to each to use tradeoff as explanation

# Then that can be aligned with TCAV

def run_reco_nn():
    filename = './q_values.csv'
    filename2 = './tcav.csv'
    filename3 = './state.csv'

    check_file = os.path.isfile(filename)
    print("q_values.csv exists: " + str(check_file))
    check_file2 = os.path.isfile(filename2)
    print("tcav.csv exists: " + str(check_file2))
    check_file3 = os.path.isfile(filename3)
    print("state.csv exists: " + str(check_file3))

    df_tcav = pd.read_csv(filename2)
    df_q = pd.read_csv(filename)
    df_state = pd.read_csv(filename3)

    mr_model, ml_model, ar_model, al_model, s_model, n_model = create_hra()

############################################### Move Right #############################################


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

################### Neural Network Helpers ################################

def create_hra():
    mr_model = create_q_model()
    ml_model = create_q_model()
    ar_model = create_q_model()
    al_model = create_q_model()
    s_model = create_q_model()
    n_model = create_q_model()

    return mr_model, ml_model, ar_model, al_model, s_model, n_model

def train_hra(q_table, state, mr_model, ml_model, ar_model, al_model, s_model, n_model):
    mr_model.fit


def create_q_model():
    inputs = layers.Input(shape=(84, 84, 3,))
    layer1 = Conv2D(filters = 16, kernel_size = 3,  padding='same', activation = 'relu')(inputs)
    layer2 = BatchNormalization(synchronized=True)(layer1)
    layer3 = MaxPooling2D(pool_size = 2)(layer2)
    layer4 = Conv2D(filters = 32, kernel_size = 3,  padding='same', activation = 'relu')(layer3)
    layer5 = BatchNormalization(synchronized=True)(layer4)
    layer6 = MaxPooling2D(pool_size = 2)(layer5)
    layer7 = Conv2D(filters = 64, kernel_size = 3,  padding='same', activation = 'relu')(layer6)
    layer8 = MaxPooling2D(pool_size = 2)(layer7)
    layer9 = Flatten()(layer8)
    layer10 = Dense(units = 64, activation = 'relu')(layer9)
    outputs = Dense(units = 1, activation = 'softmax')(layer10)

    return keras.Model(inputs=inputs,outputs=outputs)

run_reco_linear()
#run_reco_nn()