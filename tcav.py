import csv
import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_tcav():
    filename = './human_hist.csv'
    filename_social = './human_social.csv'
    filename_cultural = './human_cultural.csv'
    filename_neuro = './human_neuro.csv'

    check_file = os.path.isfile(filename)
    print("human.csv exists: " + str(check_file))
    check_file_social = os.path.isfile(filename_social)
    print("human_social.csv exists: " + str(check_file_social))
    check_file_cultural = os.path.isfile(filename_cultural)
    print("human_cultural.csv exists: " + str(check_file_cultural))
    check_file_neuro = os.path.isfile(filename_neuro)
    print("human_neuro.csv exists: " + str(check_file_neuro))

    field_names = ['Shooter_x_pos', 'Shooter_y_pos', 'Projectile_x_pos', 'Projectile_y_pos', 
                    'Player_x_pos_current', 'Player_y_pos_current', 'Player_x_pos_initial', 'Player_y_pos_initial', 
                    'Distance_x', 'Distance_y', 'Displacement_x', 'Displacement_y',
                    'Theta', 'Hit']

    df_base = pd.read_csv(filename)
    df_social= pd.read_csv(filename_social)
    df_cultural= pd.read_csv(filename_cultural)
    df_neuro= pd.read_csv(filename_neuro)
    #print(data_base[0])
    #print(df_base.loc[0][1])

    # Train with random set

    x_base = df_base.drop('Hit', axis=1)
    y_base = df_base['Hit']
    x_basedata = x_base.to_numpy()
    y_basedata = y_base.to_numpy()

    x_basetrain,x_basetest,y_basetrain,y_basetest = train_test_split(x_basedata,y_basedata,test_size=0.80)

    base_logmodel = LogisticRegression()
    base_logmodel.fit(x_basetrain,y_basetrain)

    prediction = base_logmodel.predict(x_basetest)
    print("Base model classification")
    print(classification_report(y_basetest,prediction))
    print(accuracy_score(y_basetest,prediction))
    #print(confusion_matrix(y_basetest,prediction))

    # Train with Concept set

    x_social = df_social.drop('Hit', axis=1)
    y_social = df_social['Hit']
    x_socialdata = x_social.to_numpy()
    y_socialdata = y_social.to_numpy()

    x_socialtrain,x_socialtest,y_socialtrain,y_socialtest = train_test_split(x_socialdata,y_socialdata,test_size=0.80)

    social_logmodel = LogisticRegression()
    social_logmodel.fit(x_socialtrain,y_socialtrain)

    prediction = social_logmodel.predict(x_socialtest)
    print("Social model classification")
    print(classification_report(y_socialtest,prediction))
    print(accuracy_score(y_socialtest,prediction))
    #print(confusion_matrix(y_socialtest,prediction))

    # Train with off-concept set

    x_cultural = df_cultural.drop('Hit', axis=1)
    y_cultural = df_cultural['Hit']
    x_culturaldata = x_cultural.to_numpy()
    y_culturaldata = y_cultural.to_numpy()


    # Log model to find activation difference between random and concept
    # (PUT Label to classify between random and concept and train to classify which is concept)

    #print(base_logmodel.coef_)
    #print(social_logmodel.coef_)
    df_social['Concept'] = np.ones(len(df_social), dtype=int)
    df_base['Concept'] = np.zeros(len(df_base), dtype=int)
    df_cultural['Concept'] = np.zeros(len(df_cultural), dtype=int)
    x_csocial = df_social.drop('Concept', axis=1)
    x_cbase = df_base.drop('Concept', axis=1)
    x_ccultural = df_cultural.drop('Concept', axis=1)
    y_csocial = df_social['Concept']
    y_cbase = df_base['Concept']
    y_ccultural = df_cultural['Concept']

    x_csocialdata = x_csocial.to_numpy()
    y_csocialdata = y_csocial.to_numpy()
    x_cbasedata = x_cbase.to_numpy()
    y_cbasedata = y_cbase.to_numpy()
    x_cculturaldata = x_ccultural.to_numpy()
    y_cculturaldata = y_ccultural.to_numpy()

    # Choose which concept we want to train here
    x_total = np.append(x_cbasedata,x_csocialdata, axis=0)
    y_total = np.append(y_cbasedata,y_csocialdata, axis=0)

    x_concepttrain,x_concepttest,y_concepttrain,y_concepttest = train_test_split(x_total,y_total,test_size=0.80)

    concept_logmodel = LogisticRegression()
    concept_logmodel.fit(x_concepttrain,y_concepttrain)

    concept_prediction = concept_logmodel.predict(x_concepttest)

    print("Classification of Social model against base data")
    print(classification_report(y_concepttest,concept_prediction))
    print(accuracy_score(y_concepttest,concept_prediction))
    print(confusion_matrix(y_concepttest,concept_prediction))

    concept_coef = concept_logmodel.coef_[0]
    concept_coef = np.resize(concept_coef, concept_coef.size - 1)
    concept_hitweight = concept_logmodel.coef_[0][13]
    concept_bias = concept_logmodel.intercept_

    df_socialpos = df_social.loc[df_social['Hit'] == 1]
    df_socialpos = df_socialpos.drop('Hit', axis=1)
    df_socialpos = df_socialpos.drop('Concept', axis=1)
    #concept_avg = np.mean(y_csocialdata, axis=0)
    concept_avg = np.mean(df_socialpos, axis=0)
    print(concept_avg)

    print(concept_coef)
    print(concept_hitweight, concept_bias)
    # See if hitweight is passed a threshold 50%, if so then check using hitavg, move theta away/to player_final_pos based on hitweight * hitavg

    # Generate visualization of concept

    concept_ranges = []
    for i in range(0,len(concept_coef)):
        scaled = (10*concept_coef[i]*concept_avg[i])+concept_avg[i]
        concept_ranges.append(scaled)

    tcav = []
    for i in range(0,len(concept_ranges)):
        low = min(int(concept_avg[i]), int(concept_ranges[i]))
        high = max(int(concept_avg[i]), int(concept_ranges[i]))
        if (low != high):
            generated = random.randrange(low, high)
        else: 
            generated = low
        tcav.append(generated)

    print(tcav)

    # Test conceptual sentivity with user defined off-concept

    x_total2 = np.append(x_cculturaldata,x_csocialdata, axis=0)
    y_total2 = np.append(y_cculturaldata,y_csocialdata, axis=0)

    concept_prediction2 = concept_logmodel.predict(x_total2)

    print(classification_report(y_total2,concept_prediction2))
    print(accuracy_score(y_total2,concept_prediction2))
    print(confusion_matrix(y_total2,concept_prediction2))

    return(tcav, concept_logmodel.coef_[0])

    # Conceptual sensitivity by % prediction through concept_model and actual (state + action/q-values) user wants explained
