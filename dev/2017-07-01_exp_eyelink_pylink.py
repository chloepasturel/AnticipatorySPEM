#!/usr/bin/env python
# -*- coding: utf-8 -*-

from psychopy import visual, core, gui, event #sound,
import numpy as np
# ---------------------------------------------------
import pylink
import time
import gc
import sys
import os
# ---------------------------------------------------
import time
timeStr = time.strftime("%y%m%d", time.localtime())

### IMPORTANT ###
# Avant de lancer ce script, lancer d'abord le script 'liste_binomiale.py', pour génèrer les listes binomiales

#####################################################
###                   PARAMETRE                   ###
#####################################################
# ---------------------------------------------------
# DIALOGUE
# ---------------------------------------------------
# Présente un dialogue pour changer les paramètres
expInfo = {"Sujet (2 lettres max)":'', "Session":'a'}
Nom_exp = u'test'
dlg = gui.DlgFromDict(expInfo, title=Nom_exp)
session = expInfo["Session"]
sujet = expInfo["Sujet (2 lettres max)"]
# ---------------------------------------------------

Oeil_droit = 1
#Oeil_gauche = 0
#Binoculaire = 2
nb_essais = 15
Largeur_ecran = 1280    # largeur écran
Hauteur_ecran = 1024    # hauteur écran

# écran où se deroulera l'expèrience
win = visual.Window([Largeur_ecran, Hauteur_ecran], units='pix', screen=0, fullscr=True)

# ---------------------------------------------------
# PARAMETRE EXPERIMENTAL
# ---------------------------------------------------
# point de fixation :
fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=8)
fixation_non = visual.GratingStim(win, mask='circle', sf=0, color='black', size=8)

# nombre de frame par seconde
frame = win.getActualFrameRate()    # renvoi le nombre de frame par seconde

# définition de la vitesse de la cible
Largeur_cm = 20. # largeur de l'écran en cm (57.?)
Distance_oeil_cm = 20. # distance de l'oeil en cm (57.?)
Largeur_deg = 2. * np.arctan((Largeur_cm/2) / Distance_oeil_cm) * 180./np.pi # largeur de l'écran en degrés
nb_px_deg = Largeur_ecran / Largeur_deg # nb de px par degré

Vitesse_deg_s = 12. # vitesse en degrés par seconde
Vitesse_px_s = nb_px_deg * Vitesse_deg_s # vitesse en pixel par seconde

# définition du tps (en secondes) que la cible met à arriver à son point final (0.9*demi ecran)
tps_mvt = (0.9*(Largeur_ecran/2) / Vitesse_px_s)

# liste binomiale
liste_binomiale = np.load(os.path.join('parametre_exp', session + '.npy'))

# ---------------------------------------------------
# Taille de la cible de fixation précoce
x = Largeur_ecran/2
y = Hauteur_ecran/2

W=8 # Largeur
H=8 #Hauteur

# Taille de la fenêtre de FIXATION
W_FW = W + 120 # Largeur en pixels
H_FW = H + 120 # Hauteur

# Taille de la fenêtre de MOUVEMENT
W_MW = 2*(0.9*(Largeur_ecran/2)) # Largeur en pixels (égale à 2 * longueur déplacement cible)
H_MW = 200 # Hauteur

boite_fixation = [x-W_FW/2, y-H_FW/2, x+W_FW/2, y+H_FW/2]
boite_mouvement = [x-W_MW/2, y-H_MW/2, x+W_MW/2, y+H_MW/2]

rectangle_fixation = visual.Rect(win=win, width=W_FW, height=H_FW)

colour = 7 # blanc

# ---------------------------------------------------
# PAUSE ET STOP
# ---------------------------------------------------
def fin_essai():
    '''Fin de l'enregistrement: ajoute 100 ms de données pour les événements finaux'''
    pylink.endRealTimeMode() # Fin mode temps réel
    pylink.pumpDelay(100)
    eyelink.stopRecording()
    while eyelink.getkey() :
        pass

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
    
    event.clearEvents()
    
    allKeys=event.waitKeys()
    for thisKey in allKeys:
        if thisKey in ['escape', 'a', 'q']:
            fin_essai()
            core.quit()
            win.close()
    
    #----------------------------------------------
    # CORRECTION DE LA DERIVE pendant la pause
    
    win.winHandle.set_fullscreen(False)
    #win.winHandle.set_visible(False)
    try:
        error = eyelink.doDriftCorrect((Largeur_ecran/2), (Hauteur_ecran/2), 1, 1)
        if error != 27:
            fin_essai()
            core.quit()
            win.close()
        else:
            eyelink.doTrackerSetup()
    except:
        eyelink.doTrackerSetup()
    eyelink.setOfflineMode()
    pylink.msecDelay(50)
    #win.winHandle.set_visible(True)
    win.winHandle.set_fullscreen(True)

# définition d'une touche pour stopper l'expérience
def escape_possible() :
    event.clearEvents()
    if event.getKeys(keyList=["escape", "Q", "a"]):
        print('escape requis')
        fin_essai()
        win.close()
        core.quit()

def verification():
    eyelink.flushKeybuttons(0) # Réinitialise touches et boutons sur tracker
    error = eyelink.isRecording()  # Vérifiez d'abord si l'enregistrement est interrompu
    if error != 0:
        fin_essai()
        return error
    if(eyelink.breakPressed()): # Vérifie fin du programme ou les touches ALT-F4 ou CTRL-C
        fin_essai()
        return pylink.ABORT_EXPT
    elif(eyelink.escapePressed()): # Vérifier si escape pressé
        fin_essai()
        return pylink.SKIP_TRIAL
# ---------------------------------------------------

def faire_essai(essai):
    verification()
    
    # Initialiser les données d'échantillons et les variables d'entrée de bouton
    nSData = None
    sData = None
    button = 0
    
    # Titre essai en cours au bas de l'écran d'eyetracker
    message = "record_status_message 'essai %d/%d'" %(essai + 1, nb_essais)
    eyelink.sendCommand(message)
    
    # EyeLink Data Viewer définit le début d'un essai par le message TRIALID.
    msg = "TRIALID %d" % essai
    eyelink.sendMessage(msg)
    msg = "!V TRIAL_VAR_DATA %d" % essai
    eyelink.sendMessage(msg)
    
    # Commutez le tracker sur ide et donnez-lui le temps d'effectuer un interrupteur de mode complet
    eyelink.setOfflineMode()
    pylink.msecDelay(50) 
    
    # Commencez à enregistrer des échantillons et des événements sur le fichier edf et sur le lien.
    error = eyelink.startRecording(1, 1, 1, 1)
    if error :
        fin_essai()
        return error
    
    gc.disable()  # Désactiver la collecte python pour éviter les retards
    pylink.beginRealTimeMode(100)  # Commencer le mode temps réel
    
    # ---------------------------------------------------
    # FIXATION
    # ---------------------------------------------------
    fixation.draw()
    tps_start_fix = time.time()
    win.flip()
    # ---------------------------------------------------
    eyelink.sendMessage('StimulusOn')
    eyelink.sendMessage("%d DISPLAY ON" %tps_start_fix)
    eyelink.sendMessage("SYNCTIME %d" %tps_start_fix)
    eyelink.sendCommand("clear_screen 0")
    eyelink.sendCommand('draw_box %d %d %d %d %d'%(boite_fixation[0], boite_fixation[1], boite_fixation[2], boite_fixation[3], colour))
    
    # ---------------------------------------------------
    ## SUPPRIMER ????
    # ---------------------------------------------------
    try: 
        eyelink.waitForBlockStart(100,1,0) 
    except RuntimeError: 
        if pylink.getLastError()[0] == 0: # Temps d'attente expiré sans données de lien
            fin_essai()
            print ("ERROR: No link samples received!") 
            return pylink.TRIAL_ERROR 
        else:
            raise
    #------------------------------------------
    eye_used = eyelink.eyeAvailable() # Déterminez quel oeil (s) est disponible
    if eye_used == Oeil_droit:
        eyelink.sendMessage("EYE_USED 1 RIGHT")
    else:
        print ("Error in getting the eye information!")
        fin_essai()
        return pylink.TRIAL_ERROR
    # ---------------------------------------------------
    # Tps de fixation à durée variable
    duree_fixation = np.random.uniform(0.4, 0.8) # durée du point de fixation (400-800 ms)
    tps_fixation = 0
    escape_possible()
    # ---------------------------------------------------
    while (tps_fixation < duree_fixation) :
        escape_possible()
        tps_actuel = time.time() #currentTime()
        tps_fixation = tps_actuel - tps_start_fix
        nSData = eyelink.getNewestSample() # Vérifiez la nouvelle mise à jour de l'échantillon
        if(nSData != None and (sData == None or nSData.getTime() != sData.getTime())):
            sData = nSData
            escape_possible()
            
            # Détectez si le nouvel échantillon a des données pour l'oeil en cours de suivi
            if eye_used == Oeil_droit and sData.isRightSample():
                # Obtenir l'échantillon sous la forme d'une structure événementielle
                gaze = sData.getRightEye().getGaze()
                valid_gaze_pos = isinstance(gaze, (tuple, list))
                
                if valid_gaze_pos : #Si les données sont valides, comparez la position du regard avec les limites de la fenêtre de tolérance
                    escape_possible()
                    x_eye = gaze[0]
                    y_eye = gaze[1]
                    fixation.draw()
                    win.flip()
                    diffx = abs(x_eye-Largeur_ecran/2) - W_FW/2
                    diffy = abs(y_eye-Hauteur_ecran/2) - H_FW/2
                    
                    if diffx>0 or diffy>0 :
                        escape_possible()
                        win.flip()
                        tps_start_fix = time.time()
                
                else : #Si les données sont invalides (par exemple, en cas de clignotement)
                    escape_possible()
                    win.flip()
                    fixation.draw()
                    win.flip()
                    core.wait(0.1)
                    tps_start_fix = time.time()
        
        else :
            escape_possible()
            error = eyelink.isRecording()
            if(error != 0) :
                core.wait(0.1)
            tps_start_fix = time.time()
    
    # ---------------------------------------------------
    # GAP - écran gris (300ms)
    # ---------------------------------------------------
    escape_possible()
    win.flip()
    eyelink.sendCommand("clear_screen 0") # Efface la boîte de l'écran Eyelink
    eyelink.sendMessage('StimulusOff')
    eyelink.sendCommand('draw_box %d %d %d %d %d'%(boite_mouvement[0], boite_mouvement[1], boite_mouvement[2], boite_mouvement[3], colour)) 
    core.wait(0.3) # durée du GAP
    
    # ---------------------------------------------------
    # Mouvement cible
    # ---------------------------------------------------
    escape_possible()
    x = 0
    signe_direction = liste_binomiale[essai]*2 - 1
    eyelink.sendMessage('TargetOn')
    
    for frameN in range(int(tps_mvt*frame)):
        escape_possible()
        cible = visual.Circle(win, lineColor='white', size=8, lineWidth=2, pos=(x, 0))
        cible.draw()
        x = x + signe_direction*(Vitesse_px_s/frame)
        win.flip()
    
    escape_possible()
    
    eyelink.sendMessage('TargetOff')
    eyelink.sendCommand("clear_screen 0") # Efface la boîte de l'écran Eyelink
    eyelink.sendMessage("TRIAL_RESULT %d" % button)
    
    ret_value = eyelink.getRecordingStatus() # état de l'enregistrement de sortie
    fin_essai()
    
    gc.enable() # Réactivez la collecte python pour nettoyer la mémoire à la fin de l'essai
    return ret_value

def run_essais():
    # Effectuez la configuration du suivi au début de l'expérience.
    win.winHandle.set_fullscreen(False)
    #win.winHandle.set_visible(False)
    eyelink.doTrackerSetup() # configuration du suivi
    #win.winHandle.set_visible(True)
    win.winHandle.set_fullscreen(True)

    for essai in range(nb_essais):
        if(not eyelink.isConnected() or eyelink.breakPressed()):
            break
        
        # Pause
        if essai in pauses :
            pause()
        
        while True:
            ret_value = faire_essai(essai)
            pylink.endRealTimeMode()
            if (ret_value == pylink.TRIAL_OK):
                eyelink.sendMessage("TRIAL OK")
                break
            elif (ret_value == pylink.SKIP_TRIAL):
                eyelink.sendMessage("TRIAL ABORTED")
                break
            elif (ret_value == pylink.ABORT_EXPT):
                eyelink.sendMessage("EXPERIMENT ABORTED")
                return pylink.ABORT_EXPT
            elif (ret_value == pylink.REPEAT_TRIAL):
                eyelink.sendMessage("TRIAL REPEATED")
            else: 
                eyelink.sendMessage("TRIAL ERROR")
                break
    return 0


spath = os.path.dirname(sys.argv[0])
if len(spath) !=0: os.chdir(spath)

eyelinktracker = pylink.EyeLink()
eyelink = pylink.getEYELINK()

# ---------------------------------------------------
# point de départ de l'expérience
# ---------------------------------------------------
pylink.openGraphics((Largeur_ecran, Hauteur_ecran),32) # Initialise les graphiques

edfFileName = "%s%s.EDF"%(sujet, timeStr) # ne doit pas contenir plus de 8 caractères
eyelink.openDataFile(edfFileName) # Ouvre le fichier EDF.

# réinitialise les touches et réglez le mode de suivi en mode hors connexion.
pylink.flushGetkeyQueue()
eyelink.setOfflineMode()

# Définit le système de coordonnées d'affichage et envoie un message à cet effet au fichier EDF;
eyelink.sendCommand("screen_pixel_coords =  0 0 %d %d" %(Largeur_ecran - 1, Hauteur_ecran - 1))
eyelink.sendMessage("DISPLAY_COORDS  0 0 %d %d" %(Largeur_ecran - 1, Hauteur_ecran - 1))

# ---------------------------------------------------
# NETOYER ??? version = 3
# ---------------------------------------------------
tracker_software_ver = 0
eyelink_ver = eyelink.getTrackerVersion()

if eyelink_ver == 3:
    tvstr = eyelink.getTrackerVersionString()
    vindex = tvstr.find("EYELINK CL")
    tracker_software_ver = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))

if eyelink_ver>=2:
    eyelink.sendCommand("select_parser_configuration 0")
    if eyelink_ver == 2: # Éteignez les caméras scenelink
        eyelink.sendCommand("scene_camera_gazemap = NO")
else:
    eyelink.sendCommand("saccade_velocity_threshold = 35")
    eyelink.sendCommand("saccade_acceleration_threshold = 9500")
    
# Définir le contenu du fichier EDF
eyelink.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
if tracker_software_ver>=4:
    eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
else:
    eyelink.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")

# Définir les données du lien (utilisé pour le curseur du regard)
eyelink.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
if tracker_software_ver>=4:
    eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET,INPUT")
else:
    eyelink.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT")
# ---------------------------------------------------

pylink.setCalibrationColors((255, 255, 255),(128, 128, 128)) # Définit couleur de la cible d'étalonnage (blanc) et de l'arrière-plan (gris)
pylink.setTargetSize(Largeur_ecran//70, Largeur_ecran//300) # Définit taille de la cible d'étalonnage
pylink.setCalibrationSounds("", "", "")
pylink.setDriftCorrectSounds("", "off", "off")

if(eyelink.isConnected() and not eyelink.breakPressed()): # Connexion établie et pas de fin de programme ou touches ALT-F4 ou CTRL-C pressés
    run_essais()

if eyelink != None:
    # Transfert et nettoyage de fichiers!
    eyelink.setOfflineMode()
    pylink.msecDelay(500) 

    # Fermez le fichier et transférez-le sur Display PC
    eyelink.closeDataFile()
    eyelink.receiveDataFile(edfFileName, edfFileName)
    eyelink.close()


# Fermer les graphiques de l'expérience
pylink.closeGraphics()
win.close()
core.quit()
