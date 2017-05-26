#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from psychopy import visual, core, gui, event
import numpy as np

#####################################################
###                   PARAMETRE                   ###
#####################################################

# ---------------------------------------------------
# ECRAN
# ---------------------------------------------------
# Présente un dialogue pour changer les paramètres
expInfo = {"Nombre d'essais":'100'}
Nom_exp = 'Probabilite variable par bloc'
dlg = gui.DlgFromDict(expInfo, title=Nom_exp)

<<<<<<< HEAD:2017-05-23_exp_modif_proba_blocs_poisson.py
# Largeur et hauteur de l'écran en pixel
Largeur_px = 800 #---------------------------------------------------------------- A MODIFIER (1024 ou 1920)
Hauteur_px = 600 #---------------------------------------------------------------- A MODIFIER (768 ou 1080)

# écran où se deroulera l'expèrience
win=visual.Window([Largeur_px, Hauteur_px], units='pix') # ajouter : fullscr=True pour écran total


# ---------------------------------------------------
# PARAMETRE EXPERIMENTAL
# ---------------------------------------------------
# nombre d'essais entrer dans la boite de dialogue
nb_essais = int(expInfo["Nombre d'essais"])

# définition des probabilités d'un mouvement à droite
probabiliteD = [0.25, 0.5, 0.9] #------------------------------------------------- A MODIFIER

# définition de la vitesse de la cible
Largeur_cm = 20. # largeur de l'écran en cm -------------------------------------- A MODIFIER (57.?)
Distance_oeil_cm = 20. # distance de l'oeil en cm -------------------------------- A MODIFIER (57.?)
Largeur_deg = 2. * np.arctan((Largeur_cm/2) / Distance_oeil_cm) * 180./np.pi # largeur de l'écran en degrés
nb_px_deg = Largeur_px / Largeur_deg # nombre de pixel par degré

Vitesse_deg_s = 12. # vitesse en degrés par seconde
Vitesse_px_s = nb_px_deg * Vitesse_deg_s # vitesse en pixel par seconde

# définition du tps (en secondes) que la cible met à arriver à son point final (0.9*demi ecran)
tps_mvt = (0.9*(Largeur_px/2) / Vitesse_px_s)


# ---------------------------------------------------
# BLOC DE DIFFÉRENTE PROBABILITÉ
# ---------------------------------------------------
# liste des tailles des différent blocs
taille_moyenne_bloc = 15 #-------------------------------------------------------- A MODIFIER
liste_blocs = []
essais_add = 0      # essais additionés
while essais_add <= nb_essais :
    reste_essais = nb_essais - essais_add
    taille_bloc = np.random.poisson(taille_moyenne_bloc)
    essais_add = essais_add + taille_bloc
    if essais_add <= nb_essais :
        liste_blocs.append(taille_bloc)
    else :
        liste_blocs.append(reste_essais)

# liste binomiale en fonction des probabilités des différents blocs
liste_binomiale = []
for bloc in liste_blocs :
    proba_bloc = np.random.choice(probabiliteD)
    liste_binomiale_bloc = np.random.binomial(1, proba_bloc, bloc)
    for x in range(len(liste_binomiale_bloc)):
        liste_binomiale.append(liste_binomiale_bloc[x])


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



#####################################################
###                  EXPÉRIENCE                   ###
#####################################################

# point de fixation :
fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=6)
frame = win.getActualFrameRate()    # renvoi le nombre de frame par seconde

essais = 0
for essais in range(nb_essais):
    
    # Pause
    if essais in pauses :
        pause()
    
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
