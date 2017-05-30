#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import os

def liste(session) :

    nb_essais = 100 #----------------------------------------------------------------- A MODIFIER

    # définition des probabilités d'un mouvement à droite
    probabiliteD = [0.25, 0.5, 0.9] #------------------------------------------------- A MODIFIER

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
    proba_blocs = []
    for bloc in liste_blocs :
        proba_bloc = np.random.choice(probabiliteD)
        proba_blocs.append(proba_bloc)
        liste_binomiale_bloc = np.random.binomial(1, proba_bloc, bloc)
        for x in range(len(liste_binomiale_bloc)):
            liste_binomiale.append(liste_binomiale_bloc[x])
    
    # enregistrement des paramêtres :
    f = open('parametre_exp/%s_parametre.txt'%session, 'w')
    f.write('nb_essais = %d \ntaille_moyenne_bloc = %d \nliste_blocs = %s \nproba_blocs = %s \nliste_binomiale = %s'%(nb_essais, taille_moyenne_bloc, liste_blocs, proba_blocs, liste_binomiale))
    f.close
    return liste_binomiale

a = os.path.join('parametre_exp', 'a.npy')
np.save(a, liste('a'))

b = os.path.join('parametre_exp', 'b.npy')
np.save(b, liste('b'))

c = os.path.join('parametre_exp', 'c.npy')
np.save(c, liste('c'))
