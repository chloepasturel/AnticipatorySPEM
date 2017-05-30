#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from psychopy import visual, core, gui, event, data
import numpy as np
import pickle

### IMPORTANT ###
# Avant de lancer ce script, lancer d'abord le script 'liste_binomiale.py', pour génèrer les listes binomiales

#####################################################
###                   PARAMETRE                   ###
#####################################################

# ---------------------------------------------------
# ECRAN
# ---------------------------------------------------
# Présente un dialogue pour changer les paramètres
expInfo = {"Sujet":'test', "Session":'a'}
Nom_exp = u'Probabilité variable par bloc'
dlg = gui.DlgFromDict(expInfo, title=Nom_exp)

# Largeur et hauteur de l'écran en pixel
Largeur_px = 800 #---------------------------------------------------------------- A MODIFIER (1024 ou 1920)
Hauteur_px = 600 #---------------------------------------------------------------- A MODIFIER (768 ou 1080)

# écran où se deroulera l'expèrience
win=visual.Window([Largeur_px, Hauteur_px], units='pix') # ajouter : fullscr=True pour écran total

# ---------------------------------------------------
# PARAMETRE EXPERIMENTAL
# ---------------------------------------------------
session = expInfo["Session"]
sujet = expInfo["Sujet"]
nb_essais = 100 # ATTENTION faire correspondre avec nb_essais liste_binomiale.py-- A MODIFIER

# définition de la vitesse de la cible
Largeur_cm = 20. # largeur de l'écran en cm -------------------------------------- A MODIFIER (57.?)
Distance_oeil_cm = 20. # distance de l'oeil en cm -------------------------------- A MODIFIER (57.?)
Largeur_deg = 2. * np.arctan((Largeur_cm/2) / Distance_oeil_cm) * 180./np.pi # largeur de l'écran en degrés
nb_px_deg = Largeur_px / Largeur_deg # nombre de pixel par degré

Vitesse_deg_s = 12. # vitesse en degrés par seconde ------------------------------ A MODIFIER
Vitesse_px_s = nb_px_deg * Vitesse_deg_s # vitesse en pixel par seconde

# définition du tps (en secondes) que la cible met à arriver à son point final (0.9*demi ecran)
tps_mvt = (0.9*(Largeur_px/2) / Vitesse_px_s)

# liste binomiale
with open('parametre_exp/%s'%session, 'rb') as fichier:
    f = pickle.Unpickler(fichier)
    liste_binomiale = f.load()


# ---------------------------------------------------
# PAUSE ET STOP
# ---------------------------------------------------
# Pause
intervalle_pause = 10 #----------------------------------------------------------- A MODIFIER
x = 0
pauses = []
for essais in range(nb_essais) :
    if x==intervalle_pause :
        pauses.append(essais)
        x = 0
    else :
        x = x +1

# fonction pause avec possibilité de quitter l'expérience
msg_pause = visual.TextStim(win, text="PAUSE",
                            font='calibri', height=70,
                            alignHoriz='center', alignVert='bottom')
msg_touche = visual.TextStim(win, text=u"\n\n\nTaper sur une touche pour continuer\n\nESCAPE pour arrêter l'expérience",
                            font='calibri', height=25,
                            alignHoriz='center', alignVert='top')

def pause() :
    msg_pause.draw()
    msg_touche.draw()
    win.flip()
    allKeys=event.waitKeys()
    for thisKey in allKeys:
        if thisKey in ['escape']:
            core.quit()

# définition d'une touche pour stopper l'expérience
def escape_possible() :
    if event.getKeys(keyList=["escape"]):
        core.quit()

# ---------------------------------------------------
# EVALUATION DE LA PROBABILITE
# ---------------------------------------------------
# échelle pour évaluation la probabilité
ratingScale = visual.RatingScale(win, scale=None,
    low=0, high=1, precision=100,
    labels=('gauche', 'neutre', 'droite'), tickMarks=[0, 0.5, 1], tickHeight=-1.0,
    marker='triangle', markerColor='black',
    lineColor='White',
    acceptPreText='', showValue=False, acceptText='Ok',
    minTime=0.4, maxTime=0.0)




#####################################################
###                  EXPÉRIENCE                   ###
#####################################################

# point de fixation :
fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=6)
frame = win.getActualFrameRate()    # renvoi le nombre de frame par seconde

essais = 0

resultat_evaluation = []

for essais in range(nb_essais):
    
    # Pause
    if essais in pauses :
        pause()
    
    # échelle d'évaluation
    ratingScale.reset()
    while ratingScale.noResponse :
        ratingScale.draw()
        escape_possible()
        win.flip()
    resultat_evaluation.append(ratingScale.getRating())
    
    # Tps de fixation à durée variable
    tps_fixation = np.random.uniform(0.4, 0.8) # durée du point de fixation (entre 400 et 800 ms)
    fixation.draw()     # dessine le point de fixation
    escape_possible()
    win.flip()
    core.wait(tps_fixation)
    
    # GAP - écran gris (300ms)
    win.flip()
    core.wait(0.3)      # durée du GAP
    
    # Mouvement cible
    x = 0
    if liste_binomiale[essais]==1 :
        for frameN in range(int(tps_mvt*frame)):
            cible = visual.Circle(win, lineColor='white', size=6, lineWidth=2, pos=(x, 0))
            cible.draw()
            x = x + (Vitesse_px_s/frame)
            escape_possible()
            win.flip()
    else :
        for frameN in range(int(tps_mvt*frame)):
            cible = visual.Circle(win, lineColor='white', size=6, lineWidth=2, pos=(x, 0))
            cible.draw()
            x = x - (Vitesse_px_s/frame)
            escape_possible()
            win.flip()

win.close()
print resultat_evaluation

# enregistrement parametre session (voir ce qu'il peut être ajouté) :
f = open('parametre_exp/parametre_session.txt', 'a') # 'a' ajoute texte, 'w' l'efface
f.write('%s\n'%data.getDateStr())
f.write('Sujet = %s\n'%sujet)
f.write('session = %s\n'%session)
f.write('Vitesse de la cible = %2.2f\n\n'%Vitesse_deg_s)
f.close()
