from psychopy import visual, core, gui, event
import numpy as np

#####################################################
### PARAMETRE
#####################################################

# ---------------------------------------------------
# ECRAN
# ---------------------------------------------------
# Presente un dialogue pour changer les parametres
expInfo = {"Nombre d'essais":'100'}
Nom_exp = 'Probabilite variable par bloc'
dlg = gui.DlgFromDict(expInfo, title=Nom_exp)

# ecran ou se deroulera l experience
win=visual.Window([800, 600]) # ajouter : (1920, 1080), fullscr=True pour ecran total
# ---------------------------------------------------

# nombre dessais entrer
nb_essais = int(expInfo["Nombre d'essais"])
# definition des probabilite mouvement a droite
probabiliteD = [0.25, 0.5, 0.9]

# ---------------------------------------------------
# BLOC DE DIFERENTE PROBABILITE
# ---------------------------------------------------
# liste des tailles des different blocs
taille_moyenne_bloc = 15
liste_blocs = []
essais_additionnes = 0
while essais_additionnes <= nb_essais :
    reste_essais = nb_essais - essais_additionnes
    taille_bloc = np.random.poisson(taille_moyenne_bloc)
    essais_additionnes = essais_additionnes + taille_bloc
    if essais_additionnes <= nb_essais :
        liste_blocs.append(taille_bloc)
    else :
        liste_blocs.append(reste_essais)

# liste binomiale en fonction des probabilites des differents blocs
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
intervalle_pause = 5
x = 0
pauses = []
for essais in range(nb_essais) :
    if x==intervalle_pause :
        pauses.append(essais)
        x = 0
    else :
        x = x +1

# fonction pause avec possibilite de quitter expe
msg_pause = visual.TextStim(win, text="PAUSE : taper sur une touche pour continuer")
def pause() :
    msg_pause.draw()
    win.flip()
    allKeys=event.waitKeys()
    for thisKey in allKeys:
        if thisKey in ['escape']:
            core.quit()

# definition dune touche pour stopper lexperience
def escape_possible() :
    if event.getKeys(keyList=["escape"]):
        core.quit()
# ---------------------------------------------------

#####################################################
### EXPERIENCE
#####################################################

# point de fixation :
fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=0.02)
frame = win.getActualFrameRate() # renvoi le nombre de trame par seconde
s = 1 # tps (en secondes) que met la cible a arriver a son point final
#print '1 =', s/frame
#print '2 =', np.ceil(s/frame)

essais = 0
for essais in range(nb_essais):
    
    # Pause
    if essais in pauses :
        pause()
    
    # Tps de fixation a duree variable
    tps_fixation = np.random.uniform(0.4, 0.8) # duree du point de fixation (entre 400 et 800 ms)
    fixation.draw() # dessine le point de fixation
    escape_possible()
    win.flip()
    core.wait(tps_fixation)
    
    # GAP - ecran noir (300ms)
    win.color='black'
    win.flip()
    win.flip() # il en faut deux sinon marche pas !
    core.wait(0.3) # duree de GAP
    
    # Mouvement cible
    x = 0
    win.color='grey' # ecran gris
    if liste_binomiale[essais]==1 :
        for frameN in range(int(frame*s)):
            cible = visual.Circle(win, lineColor='black', size=0.02, pos=(x, 0))
            cible.draw()
            x = x+(0.9/(frame*s))
            escape_possible()
            win.flip()
    else :
        for frameN in range(int(frame*s)):
            cible = visual.Circle(win, lineColor='black', size=0.02, pos=(x, 0))
            cible.draw()
            x = x-(0.9/(frame*s))
            escape_possible()
            win.flip()
    core.wait(1) # temps d attente apres le mouvement de la cible

win.close()