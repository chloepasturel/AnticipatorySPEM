#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Problème d'affichage de l'expérience sur le bon écran !!!
si fullscr sur écran 1 s'affiche sur écran 0 (problème avec Mac) -> aucune solution trouvée jusqu'à présent
doit être mettre fullscr=False pour être sur le bon écran
mais écran de configuration eyelink est en fullsrc donc ne s'affiche que sur écran 0 !

SOLUTION :
les 2 écrans sont de résolution différente : mettre en écran principal écran 1 puis les mettre en mirroir (pas très pratique tous ça)
sinon la calibration ne s'affiche pas bien et donc ne se fait pas correctement !
'''

from psychopy import visual, core, event
import pylink

Largeur_ecran = 1280    # largeur écran 1
Hauteur_ecran = 1024    # hauteur écran 1

win = visual.Window([Largeur_ecran, Hauteur_ecran], units='pix', screen=0, fullscr=True) # écran expèrience

nb_essais = 5
point = visual.GratingStim(win, mask='circle', sf=0, color='white', size=8)

def fin_essai():
    eyelink.stopRecording() # stop enregistrement
    while eyelink.getkey() :
        pass

def faire_essai(essai):
    eyelink.startRecording(1, 1, 1, 1) # commence enregistrement
    point.draw() # affiche point
    win.flip()
    core.wait(1) # atend 1seconde
    fin_essai()

def run_essais():
    win.winHandle.set_fullscreen(False) # enlève écran expèrience pour que écran de configuration puisse s'afficher
    #win.winHandle.set_visible(False) # ne change pas grand chose
    eyelink.doTrackerSetup() # affiche écran configuration (calibration, correction de la dérive...)
    #win.winHandle.set_visible(True)
    win.winHandle.set_fullscreen(True) # réaffiche écran expèrience (met un peut de tps à s'afficher correctement)

    for essai in range(nb_essais):
        faire_essai(essai)
    return 0

eyelinktracker = pylink.EyeLink()
eyelink = pylink.getEYELINK()

# Initialise les graphiques
pylink.openGraphics((Largeur_ecran, Hauteur_ecran),32)
eyelink.sendCommand("screen_pixel_coords =  0 0 %d %d" %(Largeur_ecran - 1, Hauteur_ecran - 1))

# Connexion eyelink et qu'aucune fin de programme ou ALT-F4 ou CTRL-C n'a été pressé
if(eyelink.isConnected() and not eyelink.breakPressed()):
    run_essais()

if eyelink != None:
    eyelink.close()

pylink.closeGraphics()
win.close()
core.quit()