#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import sys
import os
import numpy as np
import pickle

def binomial_motion(N_trials, N_blocks, tau, seed, Jeffreys=True, N_layer=3):
    """

    about Jeffrey's prior : see wikipedia
    st_dict = dict( fontsize =14, fontstyle = 'italic’) % déjà j’ai pensé à ca :)

    """

    from scipy.stats import beta
    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)
    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau # switch
        if Jeffreys: # /!\ REDEMANDER à laurent
            p_random = beta.rvs(a=.5, b=.5, size=N_blocks)
        else:
            p_random = np.random.rand(1, N_blocks)
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * p_random # probability
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks) # binomial

    return (trials, p)

def exponentiel (x, tau, maxi, start_anti, v_anti, latence, bino) :
    '''
    tau -- courbe
    maxi -- maximum
    latence -- tps où commence le mvt
    bino -- binomial
    
    start_anti = debut de l'anticipation
    v_anti =  vitesse de l'anticipation
    ''' 
    v_anti = v_anti/1000 # pour passer de sec à ms
    time = np.arange(len(x))
    vitesse = []
                
    for t in range(len(time)):
        
        if start_anti >= latence :
            if time[t] < latence :
                vitesse.append(0)
            else :
                vitesse.append((bino*2-1)*maxi*(1-np.exp(-1/tau*(time[t]-latence))))
        else :

            if time[t] < start_anti :
                vitesse.append(0)
            else :
                if time[t] < latence :
                    #vitesse.append((bino*2-1)*(time[t]-start_anti)*v_anti)
                    vitesse.append((time[t]-start_anti)*v_anti)
                    x = (time[t]-start_anti)*v_anti
                else :
                    vitesse.append((bino*2-1)*maxi*(1-np.exp(-1/tau*(time[t]-latence)))+x)
    return vitesse

def liste_p_hat(PARI):
    import bayesianchangepoint as bcp
    p_hat_bcp_e = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    p_hat_bcp_m = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

    for x in range(len(PARI)):

        N_trials = PARI[x]['N_trials']
        N_blocks = PARI[x]['N_blocks']
        p = PARI[x]['p']
        tau = N_trials/5.
        h = 1./tau

        p_hat_block_e = [[],[],[]]
        p_hat_block_m = [[],[],[]]


        for block in range(N_blocks):
            liste = [0,50,100,150,200]
            for a in range(len(liste)-1) :
                #----------------------------------------------------
                p_bar, r, beliefs = bcp.inference(p[liste[a]:liste[a+1], block, 0], h=h, p0=.5)
                p_hat_e, r_hat_e = bcp.readout(p_bar, r, beliefs, mode='expectation')
                p_hat_m, r_hat_m = bcp.readout(p_bar, r, beliefs, mode='max')

                p_hat_block_e[block].extend(p_hat_e)
                p_hat_block_m[block].extend(p_hat_m)

        p_hat_bcp_e[x] = p_hat_block_e
        p_hat_bcp_m[x] = p_hat_block_m
        
    return p_hat_bcp_e, p_hat_bcp_m

def liste_tout(PARI, ENREGISTREMENT, P_HAT=None):
    
    # liste de tout
    full_proba = []
    full_bino = []
    full_results = []
    full_va = []

    # listes de tout par sujet
    proba_sujet = []
    bino_sujet = []
    results_sujet = []
    va_sujet = []
    
    if P_HAT is not None :
        full_p_hat_e = []
        full_p_hat_m = []
        p_hat_sujet_e = []
        p_hat_sujet_m = []
        p_hat_bcp_e, p_hat_bcp_m = liste_p_hat(PARI)
        
    for x in range(len(PARI)):

        N_trials = PARI[x]['N_trials']
        N_blocks = PARI[x]['N_blocks']

        p = PARI[x]['p']
        results = (PARI[x]['results']+1)/2
        v_anti = ENREGISTREMENT[x]['v_anti']


        liste_proba = []
        liste_bino = []
        liste_results = []
        liste_va = []

        
        if P_HAT is not None :
            p_hat_e = p_hat_bcp_e[x]
            p_hat_m = p_hat_bcp_m[x]
            liste_p_hat_e = []
            liste_p_hat_m = []
            
        for block in range(N_blocks):

            switch = []
            for s in range(N_trials):
                if s in [0,50,100,150] :
                    switch.append(s)
                if p[s, block, 2]==1 :
                    switch.append(s)
            switch.append(N_trials)

            for s1 in range(len(switch)-1) :

                for trial in np.arange(switch[s1], switch[s1+1]) :
                    full_proba.append(p[trial, block, 1])
                    full_bino.append(p[trial, block, 0])
                    full_results.append(results[trial, block])
                    full_va.append(v_anti[block][trial])

                    liste_proba.append(p[trial, block, 1])
                    liste_bino.append(p[trial, block, 0])
                    liste_results.append(results[trial, block])
                    liste_va.append(v_anti[block][trial])
                    
                    if P_HAT is not None :
                        full_p_hat_e.append(p_hat_e[block][trial])
                        full_p_hat_m.append(p_hat_m[block][trial])

                        liste_p_hat_e.append(p_hat_e[block][trial])
                        liste_p_hat_m.append(p_hat_m[block][trial])


        proba_sujet.append(liste_proba)
        bino_sujet.append(liste_bino)
        results_sujet.append(liste_results)
        va_sujet.append(liste_va)
        
        if P_HAT is not None :
            p_hat_sujet_e.append(liste_p_hat_e)
            p_hat_sujet_m.append(liste_p_hat_m)
    if P_HAT is not None :
        return full_proba, full_bino, full_results, full_va, proba_sujet, bino_sujet, results_sujet, va_sujet, full_p_hat_e, full_p_hat_m, p_hat_sujet_e, p_hat_sujet_m
    else :
        return full_proba, full_bino, full_results, full_va, proba_sujet, bino_sujet, results_sujet, va_sujet



class aSPEM(object):
    """ docstring for the aSPEM class. """

    def __init__(self, mode, timeStr, observer='test') :
        self.mode = mode
        self.observer = observer
        self.timeStr = str(timeStr)
        self.init()

    def init(self) :

        self.dry_run = True
        self.dry_run = False
        self.experiment = 'aSPEM'
        self.instructions = """ TODO """

        # ---------------------------------------------------
        # setup values
        # ---------------------------------------------------
        cachedir = 'data_cache'
        datadir = 'data'
        import os
        for dir_ in [datadir, cachedir] :
            try:
                os.mkdir(dir_)
            except:
                pass

        file = self.mode + '_' + self.observer + '_' + self.timeStr + '.pkl'
        if file in os.listdir(datadir) :
            with open(os.path.join(datadir, file), 'rb') as fichier :
                self.exp = pickle.load(fichier, encoding='latin1')

        else :
            expInfo = {"Sujet":'', "Age":''}
            if not self.mode is 'model':
                # Présente un dialogue pour changer les paramètres
                Nom_exp = u'aSPEM'
                try:
                    from psychopy import gui
                    dlg = gui.DlgFromDict(expInfo, title=Nom_exp)
                    PSYCHOPY = True
                except:
                    PSYCHOPY = False

            self.observer = expInfo["Sujet"]
            self.age = expInfo["Age"]

            # width and height of your screen
            screen_width_px = 1280 #1920 #1280 for ordi enregistrement
            screen_height_px = 1024 #1080 #1024 for ordi enregistrement
            framerate = 60 #100.for ordi enregistrement
            screen = 0 # 1 pour afficher sur l'écran 2 (ne marche pas pour enregistrement (mac))

            screen_width_cm = 37 #57. # (cm)
            viewingDistance = 57. # (cm) TODO : what is the equivalent viewing distance?
            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi
            #px_per_deg = screen_height_px / screen_width_deg
            px_per_deg = screen_width_px / screen_width_deg

            # ---------------------------------------------------
            # stimulus parameters
            # ---------------------------------------------------
            dot_size = 10 # (0.02*screen_height_px)
            V_X_deg = 15 #20   # deg/s   # 15 for 'enregistrement'
            V_X = px_per_deg * V_X_deg     # pixel/s

            RashBass  = 100  # ms - pour reculer la cible à t=0 de sa vitesse * latence=RashBass

            saccade_px = .618*screen_height_px
            offset = 0 #.2*screen_height_px

            # ---------------------------------------------------
            # exploration parameters
            # ---------------------------------------------------
            N_blocks = 3 # 4 blocks avant
            seed = 51 #119 #2017
            N_trials = 200
            tau = N_trials/5.
            (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)
            stim_tau = .75 #1 #.35 # in seconds # 1.5 for 'enregistrement'

            gray_tau = .0 # in seconds
            T =  stim_tau + gray_tau
            N_frame_stim = int(stim_tau*framerate)
            # ---------------------------------------------------

            self.exp = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials, p=p, tau=tau,
                            stim_tau =stim_tau,
                            N_frame_stim=N_frame_stim, T=T,
                            datadir=datadir, cachedir=cachedir,
                            framerate=framerate,
                            screen=screen,
                            screen_width_px=screen_width_px, screen_height_px=screen_height_px,
                            px_per_deg=px_per_deg, offset=offset,
                            dot_size=dot_size, V_X_deg=V_X_deg, V_X =V_X, RashBass=RashBass, saccade_px=saccade_px,
                            mode=self.mode, observer=self.observer, age=self.age, timeStr=self.timeStr)

    def print_protocol(self):
        if True: #try:
            N_blocks = self.exp['N_blocks']
            N_trials = self.exp['N_trials']
            N_frame_stim = self.exp['N_frame_stim']
            T = self.exp['T']
            return "TODO"
    #         return """
    # ##########################
    # #  PROTOCOL  #
    # ##########################
    #
        # except:
        #     return 'blurg'


    def exp_name(self):
        return os.path.join(self.exp['datadir'], self.mode + '_' + self.observer + '_' + self.timeStr + '.pkl')

    def run_experiment(self, verb=True):

        #if verb: print('launching experiment')

        from psychopy import visual, core, event, logging, prefs
        prefs.general['audioLib'] = [u'pygame']
        from psychopy import sound

        if self.mode=='enregistrement' :
            import EyeTracking as ET
            ET = ET.EyeTracking(self.exp['screen_width_px'], self.exp['screen_height_px'], self.exp['dot_size'], self.exp['N_trials'], self.observer, self.exp['datadir'], self.timeStr)

#        logging.console.setLevel(logging.WARNING)
#        if verb: print('launching experiment')
#        logging.console.setLevel(logging.WARNING)
#        if verb: print('go!')

        # ---------------------------------------------------
        win = visual.Window([self.exp['screen_width_px'], self.exp['screen_height_px']], color=(0, 0, 0),
                            allowGUI=False, fullscr=True, screen=self.exp['screen'], units='pix') # enlever fullscr=True pour écran 2

        win.setRecordFrameIntervals(True)
        win._refreshThreshold = 1/self.exp['framerate'] + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance

        # ---------------------------------------------------
        if verb: print('FPS = ',  win.getActualFrameRate() , 'framerate=', self.exp['framerate'])

        # ---------------------------------------------------
        target = visual.Circle(win, lineColor='white', size=self.exp['dot_size'], lineWidth=2)
        fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.exp['dot_size'])
        ratingScale = visual.RatingScale(win, scale=None, low=-1, high=1, precision=100, size=.7, stretch=2.5,
                        labels=('Left', 'unsure', 'Right'), tickMarks=[-1, 0., 1], tickHeight=-1.0,
                        marker='triangle', markerColor='black', lineColor='White', showValue=False, singleClick=True,
                        showAccept=False, pos=(0, -self.exp['screen_height_px']/3)) #size=.4

        #Bip_pos = sound.Sound('2000', secs=0.05)
        #Bip_neg = sound.Sound('200', secs=0.5) # augmenter les fq

        # ---------------------------------------------------
        # fonction pause avec possibilité de quitter l'expérience
        msg_pause = visual.TextStim(win, text=u"\n\n\nTaper sur une touche pour continuer\n\nESCAPE pour arrêter l'expérience",
                                    font='calibri', height=25,
                                    alignHoriz='center')#, alignVert='top')

        text_score = visual.TextStim(win, font='calibri', height=30, pos=(0, self.exp['screen_height_px']/9))

        def pause(mode) :
            msg_pause.draw()
            win.flip()

            event.clearEvents()

            allKeys=event.waitKeys()
            for thisKey in allKeys:
                if thisKey in ['escape', 'a', 'q']:
                    core.quit()
                    win.close()
                    if mode=='enregistrement' :
                        ET.End_trial()
                        ET.End_exp()
            if mode=='enregistrement' :
                win.winHandle.set_fullscreen(False)
                win.winHandle.set_visible(False) # remis pour voir si ça enléve l'écran blanc juste après calibration
                ET.drift_correction()
                win.winHandle.set_visible(True) # remis pour voir si ça enléve l'écran blanc juste après calibration
                win.winHandle.set_fullscreen(True)

        def escape_possible(mode) :
            if event.getKeys(keyList=['escape', 'a', 'q']):
                win.close()
                core.quit()
                if mode=='enregistrement' :
                    ET.End_trial()
                    ET.End_exp()

        # ---------------------------------------------------
        def presentStimulus_fixed(dir_bool):
            dir_sign = dir_bool * 2 - 1
            target.setPos((dir_sign * (self.exp['saccade_px']), self.exp['offset']))
            target.draw()
            win.flip()
            core.wait(0.3)

        clock = core.Clock()
        myMouse = event.Mouse(win=win)

        def presentStimulus_move(dir_bool):
            clock.reset()
            #myMouse.setVisible(0)
            dir_sign = dir_bool * 2 - 1
            while clock.getTime() < self.exp['stim_tau']:
                escape_possible(self.mode)
                # la cible à t=0 recule de sa vitesse * latence=RashBass (ici mis en s)
                target.setPos(((dir_sign * self.exp['V_X']*clock.getTime())-(dir_sign * self.exp['V_X']*(self.exp['RashBass']/1000)), self.exp['offset']))
                target.draw()
                win.flip()
                win.flip()
                escape_possible(self.mode)
                #win.flip()

        # ---------------------------------------------------
        # EXPERIMENT
        # ---------------------------------------------------
        if self.mode == 'pari' :
            results = np.zeros((self.exp['N_trials'], self.exp['N_blocks'] ))

        if self.mode == 'enregistrement':
            ET.Start_exp()

            # Effectuez la configuration du suivi au début de l'expérience.
            win.winHandle.set_fullscreen(False)
            win.winHandle.set_visible(False) # remis pour voir si ça enléve l'écran blanc juste après calibration
            ET.calibration()
            win.winHandle.set_visible(True) # remis pour voir si ça enléve l'écran blanc juste après calibration
            win.winHandle.set_fullscreen(True)

        if self.mode == 'pari' :
            score = 0

        for block in range(self.exp['N_blocks']):

            x = 0

            if self.mode == 'pari' :
                text_score.text = '%1.0f/100' %(score / 50 * 100)
                text_score.draw()
                score = 0
            pause(self.mode)


            for trial in range(self.exp['N_trials']):

                # ---------------------------------------------------
                # PAUSE tous les 50 essais
                # ---------------------------------------------------
                if x == 50 :
                    if self.mode == 'pari' :
                        text_score.text = '%1.0f/100' %(score / 50 * 100)
                        text_score.draw()
                        score = 0

                    pause(self.mode)
                    x = 0

                x = x +1

                # ---------------------------------------------------
                # FIXATION
                # ---------------------------------------------------
                if self.mode == 'pari' :

                    event.clearEvents()
                    ratingScale.reset()

                    while ratingScale.noResponse :
                        fixation.draw()
                        ratingScale.draw()
                        escape_possible(self.mode)
                        win.flip()

                    ans = ratingScale.getRating()
                    results[trial, block] = ans

                if self.mode == 'enregistrement':

                    ET.check()
                    ET.Start_trial(trial)

                    fixation.draw()
                    tps_start_fix = time.time()
                    win.flip()
                    escape_possible(self.mode)

                    ET.StimulusON(tps_start_fix)
                    ET.Fixation(fixation, tps_start_fix, win, escape_possible)

                # ---------------------------------------------------
                # GAP
                # ---------------------------------------------------
                win.flip()
                escape_possible(self.mode)
                if self.mode == 'enregistrement':
                    ET.StimulusOFF()
                core.wait(0.3)

                # ---------------------------------------------------
                # Mouvement cible
                # ---------------------------------------------------
                escape_possible(self.mode)
                dir_bool = self.exp['p'][trial, block, 0]
                if self.mode == 'enregistrement':
                    ET.TargetON()
                presentStimulus_move(dir_bool)
                escape_possible(self.mode)
                win.flip()

                if self.mode == 'pari' :
                    score_trial = ans * (dir_bool * 2 - 1)
                #    if score_trial > 0 :
                #        Bip_pos.setVolume(score_trial)
                #        Bip_pos.play()
                #    else :
                #        Bip_neg.setVolume(-score_trial)
                #        Bip_neg.play()
                #    core.wait(0.1)

                    score += score_trial

                if self.mode == 'enregistrement':
                    ET.TargetOFF()
                    ret_value = ET.fin_enregistrement()
                    ET.check_trial(ret_value)

        if self.mode == 'pari' :
            self.exp['results'] = results

        if self.mode == 'enregistrement':
            ET.End_exp()

        with open(self.exp_name(), 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(self.exp)

        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()

        core.quit()



class Analysis(object):
    """ docstring for the aSPEM class. """

    def __init__(self, observer=None, mode=None) :
        self.subjects = ['AM','BMC','CS','DC','FM','IP','LB','OP','RS','SR','TN','YK']
        self.mode = mode
        self.observer = observer
        self.init()
    
    def init(self) :

        self.dry_run = True
        self.dry_run = False

        # ---------------------------------------------------
        # setup values
        # ---------------------------------------------------
        cachedir = 'data_cache'
        datadir = 'data'
        import os
        for dir_ in [datadir, cachedir] :
            try:
                os.mkdir(dir_)
            except:
                pass
        
        # ---------------------------------------------------
        # récuperation de toutes les données
        # ---------------------------------------------------
        import glob
        liste = []
        for fname in glob.glob('data/*pkl'):
            a = fname.split('/')[1].split('.')[0].split('_')
            liste.append(a)

        self.PARI = []
        for x in range(len(liste)) :
            if liste[x][0]=='pari' and liste[x][1] in self.subjects:
                a = 'data/%s_%s_%s_%s.pkl'%(liste[x][0], liste[x][1],liste[x][2],liste[x][3])
                with open(a, 'rb') as fichier :
                    b = pickle.load(fichier, encoding='latin1')
                    self.PARI.append(b)
        
        self.ENREGISTREMENT = []
        for x in range(len(liste)) :
            if liste[x][0]=='enregistrement' and liste[x][1] in self.subjects:
                a = 'parametre/param_Fit_%s.pkl'%(liste[x][1])
                with open(a, 'rb') as fichier :
                    b = pickle.load(fichier, encoding='latin1')
                    self.ENREGISTREMENT.append(b)
        # ---------------------------------------------------
        if self.observer is None :
            self.observer = liste[12][1]
        if self.mode is None :
            self.mode = 'enregistrement'

        for x in range(len(liste)):
            if liste[x][1] == self.observer and liste[x][0] == self.mode :
                self.timeStr = liste[x][2]+'_'+liste[x][3]
        
        if self.mode == 'pari' :
            for x in range(len(self.PARI)):
                if self.PARI[x]['observer'] == self.observer :
                    self.exp = self.PARI[x]
        else :
            for x in range(len(self.PARI)):
                if self.PARI[x]['observer'] == self.observer :
                    self.exp = self.PARI[x]
            for x in range(len(self.ENREGISTREMENT)):
                if self.ENREGISTREMENT[x]['observer'] == self.observer :
                    self.param = self.ENREGISTREMENT[x]


    def plot_enregistrement(self, mode=None, fig=None, axs=None, fig_width=5) :

        import matplotlib.pyplot as plt
        # from pygazeanalyser.edfreader import read_edf
        from edfreader import read_edf

        resultats = os.path.join(self.exp['datadir'], self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        screen_width_px = self.exp['screen_width_px']
        screen_height_px = self.exp['screen_height_px']
        V_X = self.exp['V_X']
        RashBass = self.exp['RashBass']
        stim_tau = self.exp['stim_tau']
        p = self.exp['p']

        for block in range(N_blocks) :

            if fig is None:
                fig_width= fig_width
                fig, axs = plt.subplots(N_trials, 1, figsize=(fig_width, (fig_width*(N_trials/2))/1.6180))

            for trial in range(N_trials) :

                #if trial <= 2 :
                trial_data = trial + N_trials*block

                data_x = data[trial_data]['x']
                data_y = data[trial_data]['y']
                trackertime = data[trial_data]['trackertime']

                TRIALID = data[trial_data]['events']['msg'][0][0]
                StimulusOn = data[trial_data]['events']['msg'][10][0]
                StimulusOf = data[trial_data]['events']['msg'][14][0]
                TargetOn = data[trial_data]['events']['msg'][15][0]
                TargetOff = data[trial_data]['events']['msg'][16][0]
                fixations = data[trial_data]['events']['Efix']
                saccades = data[trial_data]['events']['Esac']

                start = TargetOn

                TRIALID = TRIALID - start
                StimulusOn = StimulusOn - start
                StimulusOf = StimulusOf - start
                TargetOn = TargetOn - start
                TargetOff = TargetOff - start
                trackertime = trackertime - start

                #------------------------------------------------
                # TARGET
                #------------------------------------------------
                dir_bool = p[trial, block, 0]*2 - 1
                tps_mvt = TargetOff-TargetOn
                Target_trial = []
                x = screen_width_px/2

                d = 100
                for t in range(len(trackertime)):
                    if t < (TargetOn-trackertime[0]) :
                        x = screen_width_px/2
                    elif t == (TargetOn-trackertime[0]) :
                        # la cible à t=0 recule de sa vitesse * latence=RashBass (ici mis en ms)
                        x = x -(dir_bool * ((V_X/1000)*RashBass))
                    elif (t > (TargetOn-trackertime[0]) and t <= ((TargetOn-trackertime[0])+stim_tau*1000)) :
                        x = x + (dir_bool*(V_X/1000))
                    else :
                        x = x
                    Target_trial.append(x)
                #------------------------------------------------
                
                axs[trial].cla() # pour remettre ax figure a zero
                axs[trial].axis([StimulusOf-10, TargetOff+10, 0, 1280])

                axs[trial].plot(trackertime, np.ones(len(trackertime))*(screen_height_px/2), color='grey', linewidth=1.5)
                axs[trial].plot(trackertime, data_y, color='c', linewidth=1.5)

                axs[trial].plot(trackertime, Target_trial, color='k', linewidth=1.5)
                axs[trial].plot(trackertime, data_x, color='r', linewidth=1.5)

                #axs[trial].bar(TRIALID, 1280, color='g', width=5, linewidth=0)
                #axs[trial].bar(StimulusOn, 1280, color='r', width=5, linewidth=0)
                axs[trial].bar(StimulusOf, 1280, color='r', width=5, linewidth=0)
                axs[trial].bar(TargetOn, 1280, color='k', width=5, linewidth=0)
                axs[trial].bar(TargetOff, 1280, color='k', width=5, linewidth=0)

                axs[trial].set_xlabel('Time (ms)', fontsize=9)
                axs[trial].xaxis.set_ticks(range(StimulusOf+1, TargetOff, 100))
                axs[trial].xaxis.set_ticklabels(range(StimulusOf+1, TargetOff, 100), fontsize=8)
                axs[trial].set_ylabel(trial+1, fontsize=9)
                axs[trial].yaxis.set_ticks(range(0, 1280, 600))
                axs[trial].yaxis.set_ticklabels(range(0, 1280, 600), fontsize=8)
                axs[trial].xaxis.set_ticks_position('bottom')
                axs[trial].yaxis.set_ticks_position('left')
                
                for f in range(len(fixations)) :
                    axs[trial]. axvspan(fixations[f][0]-start, fixations[f][1]-start, color='r', alpha=0.1)
                for s in range(len(saccades)) :
                    axs[trial]. axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)
            
            plt.tight_layout() # pour supprimer les marge trop grande
            plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

            plt.savefig('figures/enregistrement_%s_%s.pdf'%(self.observer, block+1))
        plt.close()
        return fig, axs

    def Fit (self) :

        import matplotlib.pyplot as plt
        from lmfit import  Model, Parameters
        from edfreader import read_edf

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']


        liste_start_anti = []
        liste_liste_v_anti = []
        liste_latence = []
        liste_tau = []
        liste_maxi = []
        liste_mean = []

        for block in range(N_blocks) :
            fig_width= 12
            fig, axs = plt.subplots(N_trials, 1, figsize=(fig_width, (fig_width*(N_trials/2))/1.6180))

            block_start_anti = []
            block_liste_v_anti = []
            block_latence = []
            block_tau = []
            block_maxi = []
            block_mean = []

            for trial in range(N_trials) :

                print('block, trial = ', block, trial)

                trial_data = trial + N_trials*block
                data_x = data[trial_data]['x']
                data_y = data[trial_data]['y']
                trackertime = data[trial_data]['trackertime']

                StimulusOn = data[trial_data]['events']['msg'][10][0]
                StimulusOf = data[trial_data]['events']['msg'][14][0]
                TargetOn = data[trial_data]['events']['msg'][15][0]
                TargetOff = data[trial_data]['events']['msg'][16][0]
                saccades = data[trial_data]['events']['Esac']

                trackertime_0 = data[trial_data]['trackertime'][0]

                gradient_x = np.gradient(data_x)
                gradient_deg = gradient_x * 1/self.exp['px_per_deg'] * 1000 # gradient en deg/sec

                ##################################################
                # SUPPRESSION DES SACCADES
                ##################################################
                gradient_deg_NAN = gradient_deg
                for s in range(len(saccades)) :
                    if saccades[s][1]-trackertime_0+15 <= (len(trackertime)) :
                        for x_data in np.arange((saccades[s][0]-trackertime_0-5), (saccades[s][1]-trackertime_0+15)) :
                            gradient_deg_NAN[x_data] = np.nan
                    else :
                        for x_data in np.arange((saccades[s][0]-trackertime_0-5), (len(trackertime))) :
                            gradient_deg_NAN[x_data] = np.nan

                stop_latence = []    
                for s in range(len(saccades)) :
                    if (saccades[s][0]-trackertime_0) >= (TargetOn-trackertime_0+100) :
                        stop_latence.append((saccades[s][0]-trackertime_0))
                if stop_latence==[] :
                    stop_latence.append(len(trackertime))
                ##################################################

                start = TargetOn

                StimulusOn_s = StimulusOn - start
                StimulusOf_s = StimulusOf - start
                TargetOn_s = TargetOn - start
                TargetOff_s = TargetOff - start
                trackertime_s = trackertime - start

                ##################################################
                # FIT
                ##################################################
                model = Model(exponentiel)
                bino=p[trial, block, 0]
                params = Parameters()
                params.add('tau', value=15., min=13., max=70.)#, vary=False)
                params.add('maxi', value=15., min=1., max=40.)#, vary=False)
                params.add('latence', value=TargetOn-trackertime_0+100, min=TargetOn-trackertime_0+50, max=stop_latence[0])
                params.add('start_anti', value=TargetOn-trackertime_0-100, min=StimulusOf-trackertime_0, max=TargetOn-trackertime_0-50) #min=StimulusOf-trackertime_0+100, max=TargetOn-trackertime_0-50
                params.add('v_anti', value=0., min=-40., max=40.)
                params.add('bino', value=bino, min=0, max=1, vary=False)

                result_deg = model.fit(gradient_deg_NAN[:-250], params, x=trackertime[:-250], fit_kws={'nan_policy': 'omit'})
                ##################################################

                axs[trial].cla() # pour remettre ax figure a zero
                axs[trial].axis([StimulusOn_s-10, TargetOff_s+10, -40, 40])

                axs[trial].plot(trackertime_s, gradient_deg_NAN, color='k', alpha=0.6)
                axs[trial].plot(trackertime_s[:-250], result_deg.init_fit, 'r--', linewidth=2)
                axs[trial].plot(trackertime_s[:-250], result_deg.best_fit, color='r', linewidth=2)
                axs[trial].plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*(bino*2-1)*(15), color='k', linewidth=0.2, alpha=0.2)
                axs[trial].plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*(bino*2-1)*(10), color='k', linewidth=0.2, alpha=0.2)
                axs[trial].axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
                axs[trial].axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
                axs[trial].axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)
                for s in range(len(saccades)) :
                    axs[trial].axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)

                debut  = TargetOn - trackertime_0 # TargetOn - temps_0

                start_anti = result_deg.values['start_anti']-debut
                v_anti = result_deg.values['v_anti']
                latence = result_deg.values['latence']-debut
                tau = result_deg.values['tau']
                maxi = result_deg.values['maxi']

                if np.isnan(gradient_deg_NAN[int(result_deg.values['latence'])]) and np.isnan(gradient_deg_NAN[int(result_deg.values['latence'])-30]) and np.isnan(gradient_deg_NAN[int(result_deg.values['latence'])-70]) ==True :
                    start_anti = np.nan
                    v_anti = np.nan
                    latence = np.nan
                    tau = np.nan
                    maxi = np.nan
                else :
                    axs[trial].bar(latence, 80, bottom=-40, color='r', width=6, linewidth=0)
                    if trial==0 :
                        axs[trial].text(latence+25, -35, "Latence"%(latence), color='r', fontsize=14)

                block_start_anti.append(start_anti)
                block_liste_v_anti.append(v_anti)
                block_latence.append(latence)
                block_tau.append(tau)
                block_maxi.append(maxi)
                block_mean.append(np.nanmean(gradient_deg_NAN[debut-50:debut+50]))

                #axs[trial].bar(latence, 80, bottom=-40, color='r', width=6, linewidth=0)

                if trial==0 :
                    axs[trial].text(StimulusOn_s+(StimulusOf_s-StimulusOn_s)/2, 31, "FIXATION", color='k', fontsize=16, ha='center', va='bottom')
                    axs[trial].text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", color='r', fontsize=16, ha='center', va='bottom')
                    axs[trial].text(TargetOn_s+(TargetOff_s-TargetOn_s)/2, 31, "POURSUITE", color='k', fontsize=16, ha='center', va='bottom')
                    #axs[trial].text(latence+25, -35, "Latence"%(latence), color='r', fontsize=14)#,  weight='bold')
                #axs[trial].text(StimulusOn+15, -2, "%s"%(result.fit_report()), color='k', fontsize=15)
                axs[trial].text(StimulusOn_s+15, 18, "start_anti: %s \nv_anti: %s"%(start_anti, v_anti), color='k', fontsize=14, va='bottom')
                axs[trial].text(StimulusOn_s+15, -18, "latence: %s \ntau: %s \nmaxi: %s"%(latence, tau, maxi), color='k', fontsize=14, va='top')

                axs[trial].set_xlabel('Time (ms)', fontsize=9)
                axs[trial].set_ylabel(trial+1, fontsize=9)

            liste_start_anti.append(block_start_anti)
            liste_liste_v_anti.append(block_liste_v_anti)
            liste_latence.append(block_latence)
            liste_tau.append(block_tau)
            liste_maxi.append(block_maxi)
            liste_mean.append(block_mean)

            plt.tight_layout() # pour supprimer les marge trop grande
            plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

            plt.savefig('figures/Fit_%s_%s.pdf'%(observer, block+1))

        plt.close()

        param = {}
        param['observer'] = observer
        param['start_anti'] = liste_start_anti
        param['v_anti'] = liste_liste_v_anti
        param['latence'] = liste_latence
        param['tau'] = liste_tau
        param['maxi'] = liste_maxi
        param['moyenne'] = liste_mean

        file = os.path.join('parametre', 'param_Fit_' + observer + '.pkl')
        with open(file, 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(param)

        print('FIN !!!')

    def Fit_essai(self, block=0, trial=70, fig_width=15, t_titre=35, t_label=20) :

        import matplotlib.pyplot as plt
        from edfreader import read_edf
        from lmfit import  Model, Parameters

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']


        fig, axs = plt.subplots(1, 1, figsize=(fig_width, (fig_width/2)/1.6180))

        trial_data = trial + N_trials*block

        data_x = data[trial_data]['x']
        data_y = data[trial_data]['y']
        trackertime = data[trial_data]['trackertime']

        StimulusOn = data[trial_data]['events']['msg'][10][0]
        StimulusOf = data[trial_data]['events']['msg'][14][0]
        TargetOn = data[trial_data]['events']['msg'][15][0]
        TargetOff = data[trial_data]['events']['msg'][16][0]
        saccades = data[trial_data]['events']['Esac']
        trackertime_0 = data[trial_data]['trackertime'][0]

        gradient_x = np.gradient(data_x) # gradient en px/ms
        gradient_deg = gradient_x * 1/self.exp['px_per_deg'] * 1000 # gradient en deg/sec

        ##################################################
        # SUPPRESSION DES SACCADES
        ##################################################
        gradient_deg_NAN = gradient_deg

        for s in range(len(saccades)) :
            if saccades[s][1]-trackertime_0+15 <= (len(trackertime)) :
                for x_data in np.arange((saccades[s][0]-trackertime_0-5), (saccades[s][1]-trackertime_0+15)) :
                    gradient_deg_NAN[x_data] = np.nan
            else :
                for x_data in np.arange((saccades[s][0]-trackertime_0-5), (len(trackertime))) :
                    gradient_deg_NAN[x_data] = np.nan

        stop_latence = []    
        for s in range(len(saccades)) :
            if (saccades[s][0]-trackertime_0) >= (TargetOn-trackertime_0+100) :
                stop_latence.append((saccades[s][0]-trackertime_0))
        if stop_latence==[] :
            stop_latence.append(len(trackertime))
        ##################################################

        start = TargetOn

        StimulusOn_s = StimulusOn - start
        StimulusOf_s = StimulusOf - start
        TargetOn_s = TargetOn - start
        TargetOff_s = TargetOff - start
        trackertime_s = trackertime - start

        # FIT
        model = Model(exponentiel)
        bino=p[trial, block, 0]
        params = Parameters()

        params.add('tau', value=15., min=13., max=80.)#, vary=False)
        params.add('maxi', value=15., min=1., max=40.)#, vary=False)
        params.add('latence', value=TargetOn-trackertime_0+100, min=TargetOn-trackertime_0+50, max=stop_latence[0])
        params.add('start_anti', value=TargetOn-trackertime_0-100, min=StimulusOf-trackertime_0, max=TargetOn-trackertime_0-50)
        params.add('v_anti', value=(bino*2-1)*0, min=-40., max=40.)
        params.add('bino', value=bino, min=0, max=1, vary=False)

        #result_deg = model.fit(new_gradient_deg, params, x=new_time)
        result_deg = model.fit(gradient_deg_NAN[:-250], params, x=trackertime[:-250], fit_kws={'nan_policy': 'omit'})

        debut  = TargetOn - trackertime_0 # TargetOn - temps_0

        axs.axis([StimulusOn_s-10, TargetOff_s+10, -40, 40])

        axs.plot(trackertime_s, gradient_deg_NAN, color='k', alpha=0.6)
        axs.plot(trackertime_s[:-250], result_deg.init_fit, 'r--', linewidth=2)
        axs.plot(trackertime_s[:-250], result_deg.best_fit, color='r', linewidth=2)
        axs.plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*(bino*2-1)*15, color='k', linewidth=0.2, alpha=0.2)
        axs.plot(trackertime_s, np.ones(np.shape(trackertime_s)[0])*(bino*2-1)*10, color='k', linewidth=0.2, alpha=0.2)


        axs.axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
        axs.axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
        axs.axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)
        for s in range(len(saccades)) :
            axs.axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)

        start_anti = result_deg.values['start_anti']-debut
        v_anti = result_deg.values['v_anti']
        latence = result_deg.values['latence']-debut
        tau = result_deg.values['tau']
        maxi = result_deg.values['maxi']

        if np.isnan(gradient_deg_NAN[int(result_deg.values['latence'])]) and np.isnan(gradient_deg_NAN[int(result_deg.values['latence'])-30]) and np.isnan(gradient_deg_NAN[int(result_deg.values['latence'])-70]) ==True :
            print('lala')
            start_anti = np.nan
            v_anti = np.nan
            latence = np.nan
            tau = np.nan
            maxi = np.nan
        else :
            axs.bar(latence, 80, bottom=-40, color='r', width=6, linewidth=0)
            axs.text(latence+25, -35, "Latence", color='r', fontsize=14)#,  weight='bold')

        axs.text(StimulusOn_s+(StimulusOf_s-StimulusOn_s)/2, 31, "FIXATION", color='k', fontsize=16, ha='center', va='bottom')
        axs.text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", color='r', fontsize=16, ha='center', va='bottom')
        axs.text(TargetOn_s+(TargetOff_s-TargetOn_s)/2, 31, "POURSUITE", color='k', fontsize=16, ha='center', va='bottom')
        axs.text(StimulusOn_s+15, 18, "start_anti: %s \nv_anti: %s"%(start_anti, v_anti), color='k', fontsize=14, va='bottom')
        axs.text(StimulusOn_s+15, -18, "latence: %s \ntau: %s \nmaxi: %s"%(latence, tau, maxi), color='k', fontsize=14, va='top')

        axs.set_xlabel('Temps (ms)', fontsize=12)
        axs.set_ylabel('Vitesse', fontsize=12)
        plt.show()

        return fig, axs, result_deg.fit_report()


    def plot_experiment(self, mode=None, fig=None, axs=None, fig_width=15, t_titre=35, t_label=20):

        import matplotlib.pyplot as plt

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']
        ec = 0.2
        
        if fig is None:
            fig_width= fig_width
            fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_width/1.6180))

        for i_layer, label in enumerate(['Target Direction', 'Probability', 'Switch']) :
            for i_block in range(N_blocks):
                axs[i_layer].step(range(N_trials), p[:, i_block, i_layer]+i_block+ec*i_block, lw=1, c='k', alpha=.3)
                axs[i_layer].fill_between(range(N_trials), i_block+np.zeros_like(p[:, i_block, i_layer])+ec*i_block, i_block+p[:, i_block, i_layer]+ec*i_block,
                                          lw=.5, alpha=.3, facecolor='k', step='pre')

            #------------------------------------------------
            # Barre Pause
            #------------------------------------------------
            axs[i_layer].bar(49, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)
            axs[i_layer].bar(99, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)
            axs[i_layer].bar(149, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)

            #------------------------------------------------
            # affiche les numéro des block sur le côté gauche
            #------------------------------------------------
            ax_block = axs[i_layer].twinx()
            if i_layer==0 :
                ax_block.set_ylabel('Block', fontsize=t_label/1.5, rotation='horizontal', ha='left', va='bottom')
                ax_block.yaxis.set_label_coords(1.01, 1.08)

            ax_block.set_ylim(-.05, N_blocks + .05)
            ax_block.set_yticks(np.arange(N_blocks)+0.5)
            ax_block.set_yticklabels(np.arange(N_blocks)+1, fontsize=t_label/1.5)
            ax_block.yaxis.set_tick_params(width=0, pad=(t_label/1.5)+10)

            #------------------------------------------------
            # cosmétique
            #------------------------------------------------
            axs[i_layer].set_xlim(-1, N_trials)
            
            if i_layer==2 :
                axs[i_layer].set_xticks([-1, 49, 99,149])
                axs[i_layer].set_xticklabels([0, 50, 100, 150], ha='left',fontsize=t_label/2)
                axs[i_layer].yaxis.set_tick_params(width=0)
                axs[i_layer].xaxis.set_ticks_position('bottom')
            else :
                axs[i_layer].set_xticks([])

            axs[i_layer].set_ylim(-(ec/2), N_blocks +ec*3-(ec/2))
            axs[i_layer].set_ylabel(label, fontsize=t_label)
            axs[i_layer].set_yticks([0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2])
            axs[i_layer].yaxis.set_label_coords(-0.05, 0.5)
            axs[i_layer].yaxis.set_tick_params(direction='out')
            axs[i_layer].yaxis.set_ticks_position('left')

        #-------------------------------------------------------------------------------------------------------------
        if mode == 'pari' :
            results = (self.exp['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
            for block in range(N_blocks):
                if block == 0 :
                    axs[1].step(range(N_trials), block+results[:, block]+ec*block, lw=1, alpha=.9, color='darkred', label='Results Bet')
                else :
                    axs[1].step(range(N_trials), block + results[:, block], alpha=.9, color='darkred')
            axs[0].set_title('Bet results', fontsize=t_titre, x=0.5, y=1.1)

        #------------------------------------------------
        elif mode == 'enregistrement' :
            v_anti = self.param['v_anti']
            for block in range(N_blocks):
                if block == 0 :
                    axs[1].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block, color='k', lw=1, alpha=1, label='Eye movement')
                else :
                    axs[1].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block, color='k', lw=1, alpha=1)
            axs[0].set_title('Eye movements recording results', fontsize=t_titre, x=0.5, y=1.1)

        #------------------------------------------------
        elif mode=='deux':
            results = (self.exp['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
            v_anti = self.param['v_anti']
            for block in range(N_blocks):
                if block == 0 :
                    axs[1].step(range(N_trials), block+results[:, block]+ec*block, lw=1, alpha=.9, color='darkred', label='Results Bet')
                    axs[1].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block, color='k', lw=1, alpha=1, label='Eye movement')
                else :
                    axs[1].step(range(N_trials), block+results[:, block]+ec*block, lw=1, alpha=.9, color='darkred')
                    axs[1].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,color='k', lw=1, alpha=1)
            axs[0].set_title('Eye movements recording results', fontsize=t_titre, x=0.5, y=1.1)

        #------------------------------------------------
        elif mode is None :
            axs[0].set_title('Experiment', fontsize=t_titre, x=0.5, y=1.1)
        #-------------------------------------------------------------------------------------------------------------

        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        axs[0].set_yticklabels(['left','right','left','right','left','right'],fontsize=t_label/2)
        axs[1].set_yticklabels(['0','1','0','1','0','1'],fontsize=t_label/2)
        axs[2].set_yticklabels(['No','Yes','No','Yes','No','Yes'],fontsize=t_label/2)
        axs[-1].set_xlabel('Trials', fontsize=t_label)

        fig.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        #------------------------------------------------

        return fig, axs, p

    def plot_experiment_plus(self, sujet=[0], fig_width=15, t_titre=35, t_label=20) :

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(sujet)+1, 1, figsize=(fig_width, ((len(sujet)+1)*fig_width/3)/(1.6180)))
        
        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']
        ec = 0.2

        for i_block in range(N_blocks):
            axs[0].step(range(N_trials), p[:, i_block, 0]+i_block+ec*i_block, lw=1, c='k', alpha=.3)
            axs[0].fill_between(range(N_trials), i_block+np.zeros_like(p[:, i_block, 0])+ec*i_block,
                                      i_block+p[:, i_block, 0]+ec*i_block,
                                      lw=.5, alpha=.3, facecolor='k', step='pre')

            for s in range(len(sujet)) :
                axs[s+1].step(range(N_trials), p[:, i_block, 1]+i_block+ec*i_block, lw=1, c='k', alpha=.3)
                axs[s+1].fill_between(range(N_trials), i_block+np.zeros_like(p[:, i_block, 1])+ec*i_block, i_block+p[:, i_block, 1]+ec*i_block,
                                          lw=.5, alpha=.3, facecolor='k', step='pre')

                axs[s+1].set_yticklabels(['0','1','0','1','0','1'],fontsize=t_label/2)

        for i_layer in range(len(sujet)+1) :

            if i_layer != 0 :

                print('sujet', sujet[i_layer-1], '=', self.PARI[sujet[i_layer-1]]['observer'])
                p = self.PARI[sujet[i_layer-1]]['p']
                results = (self.PARI[sujet[i_layer-1]]['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
                v_anti = self.ENREGISTREMENT[sujet[i_layer-1]]['v_anti']
                axs[i_layer].set_ylabel('Subject %s'%(sujet[i_layer-1]), fontsize=t_label)

                for block in range(N_blocks):

                    if block == 0 :
                        axs[i_layer].step(range(N_trials), block+results[:, block]+ec*block, lw=1, alpha=.9,
                                    color='r', label='Individual guess')
                        axs[i_layer].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block, 
                                    color='k', lw=1, alpha=1, label='Eye movements')
                    else :
                        axs[i_layer].step(range(N_trials), block+results[:, block]+ec*block, lw=1, alpha=.9,
                                          color='r')
                        axs[i_layer].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                                    color='k', lw=1, alpha=1)

            #------------------------------------------------
            # Barre Pause
            #------------------------------------------------
            axs[i_layer].bar(49, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)
            axs[i_layer].bar(99, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)
            axs[i_layer].bar(149, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)

            #------------------------------------------------
            # affiche les numéro des block sur le côté gauche
            #------------------------------------------------
            ax_block = axs[i_layer].twinx()
            if i_layer==0 :
                ax_block.set_ylabel('Block', fontsize=t_label/1.5, rotation='horizontal', ha='left', va='bottom')
                ax_block.yaxis.set_label_coords(1.01, 1.08)

            ax_block.set_ylim(0, N_blocks)
            ax_block.set_yticks(np.arange(N_blocks)+0.5)
            ax_block.set_yticklabels(np.arange(N_blocks)+1, fontsize=t_label/1.5)
            ax_block.yaxis.set_tick_params(width=0, pad=(t_label/1.5)+10)

            #------------------------------------------------
            # cosmétique
            #------------------------------------------------
            if i_layer==len(sujet)+1 :
                axs[i_layer].set_xticks([-1, 49, 99,149])
                axs[i_layer].set_xticklabels([0, 50, 100, 150], ha='left',fontsize=t_label/2)
                axs[i_layer].yaxis.set_tick_params(width=0)
                axs[i_layer].xaxis.set_ticks_position('bottom')
            else :
                axs[i_layer].set_xticks([])
            axs[i_layer].set_xlim(-1, N_trials)

            axs[i_layer].set_ylim(-(ec/2), N_blocks +ec*3-(ec/2))
            axs[i_layer].set_yticks([0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2])
            axs[i_layer].yaxis.set_label_coords(-0.05, 0.5)
            axs[i_layer].yaxis.set_tick_params(direction='out')
            axs[i_layer].yaxis.set_ticks_position('left')

        axs[1].legend(fontsize=t_label/1.3, bbox_to_anchor=(0., 2.1, 1, 0.), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        axs[0].set_title('Eye movements recording results', fontsize=t_titre, x=0.5, y=1.25,va='bottom')
        axs[0].set_ylabel('Target Direction', fontsize=t_label)
        axs[0].set_yticklabels(['left','right','left','right','left','right'],fontsize=t_label/2)

        axs[-1].set_xlabel('Trials', fontsize=t_label)

        fig.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        #------------------------------------------------

        return fig, axs

    def plot_fit_fonction(self, fig_width=15, t_titre=35, t_label=20) :

        import matplotlib.pyplot as plt
        from lmfit import  Model, Parameters
        from edfreader import read_edf

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']


        fig, axs = plt.subplots(1, 1, figsize=(fig_width, (fig_width/2)/1.6180))

        block = 0
        trial = 0

        trial_data = trial + N_trials*block

        data_x = data[trial_data]['x']
        data_y = data[trial_data]['y']
        trackertime = data[trial_data]['trackertime']

        StimulusOn = data[trial_data]['events']['msg'][10][0]
        StimulusOf = data[trial_data]['events']['msg'][14][0]
        TargetOn = data[trial_data]['events']['msg'][15][0]
        TargetOff = data[trial_data]['events']['msg'][16][0]
        saccades = data[trial_data]['events']['Esac']
        trackertime_0 = data[trial_data]['trackertime'][0]

        gradient_x = np.gradient(data_x) # gradient en px/ms
        gradient_deg = gradient_x * 1/self.exp['px_per_deg'] * 1000 # gradient en deg/sec

        ##################################################
        # SUPPRESSION DES SACCADES
        ##################################################
        gradient_deg_NAN = []
        for x_data in range(len(data_x)):
            saccade = None
            for s in range(len(saccades)) :
                if x_data in np.arange((saccades[s][0]-trackertime_0), (saccades[s][1]-trackertime_0+10)) :
                    gradient_deg_NAN.append(np.nan)#gradient_deg_NAN[x_data-1])#'nan')
                    saccade = 'yes'
            if not saccade :
                gradient_deg_NAN.append(gradient_deg[x_data])
            saccade = None 
            
        stop_latence = []    
        for s in range(len(saccades)) :
            if (saccades[s][0]-trackertime_0) >= (TargetOn-trackertime_0+100) :
                stop_latence.append((saccades[s][0]-trackertime_0))
        if stop_latence==[] :
            stop_latence.append(len(trackertime))
        ##################################################

        start = TargetOn

        StimulusOn_s = StimulusOn - start
        StimulusOf_s = StimulusOf - start
        TargetOn_s = TargetOn - start
        TargetOff_s = TargetOff - start
        trackertime_s = trackertime - start
                
        # FIT
        model = Model(exponentiel)
        bino=p[trial, block, 0]
        params = Parameters()

        params.add('tau', value=15., min=13., max=80.)#, vary=False)
        params.add('maxi', value=15., min=1., max=40.)#, vary=False)
        params.add('latence', value=TargetOn-trackertime_0+100, min=TargetOn-trackertime_0+50, max=stop_latence[0])
        params.add('start_anti', value=TargetOn-trackertime_0-100, min=StimulusOf-trackertime_0, max=TargetOn-trackertime_0-50)
        params.add('v_anti', value=-25, min=-40., max=40.)
        params.add('bino', value=bino, min=0, max=1, vary=False)

        result_deg = model.fit(gradient_deg_NAN, params, x=trackertime, fit_kws={'nan_policy': 'omit'})

        # -------------------------------------------------------------------------------------
        # COSMETIQUE
        # -------------------------------------------------------------------------------------
        axs.plot(trackertime_s[:TargetOn-trackertime_0-100], result_deg.init_fit[:TargetOn-trackertime_0-100], 'k', linewidth=2)
        axs.plot(trackertime_s[TargetOn-trackertime_0+250:], result_deg.init_fit[TargetOn-trackertime_0+250:], 'k', linewidth=2)

        # ---------------------------------
        axs.text(StimulusOf_s+(TargetOn_s-StimulusOf_s)/2, 31, "GAP", color='k', fontsize=t_label, ha='center', va='bottom')
        axs.text((StimulusOf_s-750)/2, 31, "FIXATION", color='k', fontsize=t_label, ha='center', va='bottom')
        axs.text((750-TargetOn_s)/2, 31, "PURSUIT", color='k', fontsize=t_label, ha='center', va='bottom')

        axs.axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
        axs.axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
        axs.axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)

        # ---------------------------------
        # Anticipation
        # ---------------------------------
        axs.text(TargetOn_s, 15, "Anticipation", color='r', fontsize=t_label/1.5, ha='center')

        # ---------------------------------
        # V_a
        # ---------------------------------
        axs.plot(trackertime_s[TargetOn-trackertime_0-100:TargetOn-trackertime_0+100], result_deg.init_fit[TargetOn-trackertime_0-100:TargetOn-trackertime_0+100], c='r', linewidth=2)
        axs.text(TargetOn_s-50, -5, r"A$_a$", color='r', fontsize=t_label/1.5, ha='center', va='top')
        axs.annotate('', xy=(TargetOn_s+100, result_deg.init_fit[TargetOn-trackertime_0+100]-3), xycoords='data', fontsize=t_label/1.2,
                    xytext=(TargetOn_s-95, result_deg.init_fit[TargetOn-trackertime_0-95]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))

        # ---------------------------------
        # Start_a
        # ---------------------------------
        axs.bar(TargetOn_s-100, 80, bottom=-40, color='k', width=4, linewidth=0, alpha=0.7)
        axs.text(TargetOn_s-100-25, -35, "Start anticipation", color='k', fontsize=t_label/1.5, alpha=0.7, ha='right')

        # ---------------------------------
        # latence
        # ---------------------------------

        axs.bar(TargetOn_s+99, 80, bottom=-40, color='firebrick', width=4, linewidth=0, alpha=1)
        axs.text(TargetOn_s+99+25, -35, "Latency", color='firebrick', fontsize=t_label/1.5)

        # ---------------------------------
        # tau
        # ---------------------------------
        axs.plot(trackertime_s[TargetOn-trackertime_0+100:TargetOn-trackertime_0+250], result_deg.init_fit[TargetOn-trackertime_0+100:TargetOn-trackertime_0+250], c='darkred', linewidth=2)
        axs.annotate(r'$\tau$', xy=(TargetOn_s+140, result_deg.init_fit[TargetOn-trackertime_0+140]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                    xytext=(TargetOn_s+170, result_deg.init_fit[TargetOn-trackertime_0]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))

        # ---------------------------------
        # Max
        # ---------------------------------
        axs.plot(trackertime_s[TargetOn-trackertime_0+100:], np.ones(len(trackertime_s[TargetOn-trackertime_0+100:]))*result_deg.init_fit[TargetOn-trackertime_0+100], '--k', linewidth=1, alpha=0.5)
        axs.plot(trackertime_s[TargetOn-trackertime_0+100:], np.ones(len(trackertime_s[TargetOn-trackertime_0+100:]))*result_deg.init_fit[TargetOn-trackertime_0+250], '--k', linewidth=1, alpha=0.5)
        axs.text(TargetOn_s+400+25, ((result_deg.init_fit[TargetOn-trackertime_0+100]+result_deg.init_fit[TargetOn-trackertime_0+250])/2),
                 'Max', color='k', fontsize=t_label/1.5, va='center')
        axs.annotate('', xy=(TargetOn_s+400, result_deg.init_fit[TargetOn-trackertime_0+100]), xycoords='data', fontsize=t_label/1.5,
                    xytext=(TargetOn_s+400, result_deg.init_fit[TargetOn-trackertime_0+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))

        # -------------------------------------------------------------------------------------
        axs.axis([-750, 750, -40, 40])      
        axs.xaxis.set_ticks_position('bottom')
        axs.xaxis.set_tick_params(labelsize=t_label/2)
        axs.yaxis.set_ticks_position('left')
        axs.yaxis.set_tick_params(labelsize=t_label/2)

        axs.set_xlabel('Time (ms)', fontsize=t_label)
        axs.set_ylabel('Velocity (°/s)', fontsize=t_label)
        axs.set_title('Fit Function', fontsize=t_titre, x=0.5, y=1.05)

        return fig, axs

    def plot_velocity_fit(self, block=0, liste=[19,71,83], fig_width=15, t_titre=35, t_label=20):

        import matplotlib.pyplot as plt
        from lmfit import  Model, Parameters
        from edfreader import read_edf
        
        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']
        
        fig, axs = plt.subplots(len(liste), 1, figsize=(fig_width, (fig_width*(len(liste)/2)/1.6180)))

        x = 0
        for trial in liste :

            trial_data = trial + N_trials*block
            data_x = data[trial_data]['x']
            data_y = data[trial_data]['y']
            trackertime = data[trial_data]['trackertime']

            StimulusOn = data[trial_data]['events']['msg'][10][0]
            StimulusOf = data[trial_data]['events']['msg'][14][0]
            TargetOn = data[trial_data]['events']['msg'][15][0]
            TargetOff = data[trial_data]['events']['msg'][16][0]
            saccades = data[trial_data]['events']['Esac']

            trackertime_0 = data[trial_data]['trackertime'][0]

            gradient_x = np.gradient(data_x)
            gradient_deg = gradient_x * 1/self.exp['px_per_deg'] * 1000 # gradient en deg/sec

            ##################################################
            # SUPPRESSION DES SACCADES
            ##################################################
            gradient_deg_NAN = []
            for x_data in range(len(data_x)):
                saccade = None
                for s in range(len(saccades)) :
                    if x_data in np.arange((saccades[s][0]-trackertime_0), (saccades[s][1]-trackertime_0+10)) :
                        gradient_deg_NAN.append(np.nan)
                        saccade = 'yes'
                if not saccade :
                    gradient_deg_NAN.append(gradient_deg[x_data])
                saccade = None        
            ##################################################

            start = TargetOn

            StimulusOn_s = StimulusOn - start
            StimulusOf_s = StimulusOf - start
            TargetOn_s = TargetOn - start
            TargetOff_s = TargetOff - start
            trackertime_s = trackertime - start

            ##################################################
            # FIT
            ##################################################
            model = Model(exponentiel)
            bino=p[trial, block, 0]
            params = Parameters()
            params.add('tau', value=15., min=13., max=70.)#, vary=False)
            params.add('maxi', value=15., min=10., max=40.)#, vary=False)
            params.add('latence', value=TargetOn-trackertime_0+100, min=TargetOn-trackertime_0+50, max=len(trackertime))
            params.add('start_anti', value=TargetOn-trackertime_0-100, min=StimulusOf-trackertime_0+100, max=TargetOn-trackertime_0-50)
            params.add('v_anti', value=0., min=-100., max=100.)
            params.add('bino', value=bino, min=0, max=1, vary=False)

            result_deg = model.fit(gradient_deg_NAN, params, x=trackertime, fit_kws={'nan_policy': 'omit'})
            ##################################################

            axs[x].plot(trackertime_s, gradient_deg_NAN, color='k', alpha=0.4)
            #axs[x].plot(trackertime_s, result_deg.best_fit, color='k', linewidth=2)

            # ---------------------------------
            debut  = TargetOn - trackertime_0 # TargetOn - temps_0
            start_anti = result_deg.values['start_anti']
            v_anti = result_deg.values['v_anti']
            latence = result_deg.values['latence']
            tau = result_deg.values['tau']
            maxi = result_deg.values['maxi']
            ##################################################
            
            # -------------------------------------------------------------------------------------
            # COSMETIQUE
            # -------------------------------------------------------------------------------------

            axs[x].plot(trackertime_s[:int(start_anti)], result_deg.best_fit[:int(start_anti)], 'k', linewidth=2)
            axs[x].plot(trackertime_s[int(latence)+250:], result_deg.best_fit[int(latence)+250:], 'k', linewidth=2)

            # ---------------------------------
            axs[x].axvspan(StimulusOn_s, StimulusOf_s, color='k', alpha=0.2)
            axs[x].axvspan(StimulusOf_s, TargetOn_s, color='r', alpha=0.2)
            axs[x].axvspan(TargetOn_s, TargetOff_s, color='k', alpha=0.15)
            for s in range(len(saccades)) :
                axs[x].axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.15)

            # ---------------------------------
            # V_a
            # ---------------------------------
            axs[x].plot(trackertime_s[int(start_anti):int(latence)], result_deg.best_fit[int(start_anti):int(latence)], c='r', linewidth=2)
            axs[x].text((trackertime_s[int(start_anti)]+trackertime_s[int(latence)])/2, result_deg.best_fit[int(start_anti)]-15,
                        r"A$_a$ = %0.2f °/s$^2$"%(v_anti), color='r', fontsize=t_label/1.5, ha='center')
            axs[x].annotate('', xy=(trackertime_s[int(latence)], result_deg.best_fit[int(latence)]-3), xycoords='data', fontsize=t_label/1.5,
                    xytext=(trackertime_s[int(start_anti)], result_deg.best_fit[int(start_anti)]-3), textcoords='data', arrowprops=dict(arrowstyle="->", color='r'))

            # ---------------------------------
            # Start_a
            # ---------------------------------
            axs[x].bar(trackertime_s[int(start_anti)], 80, bottom=-40, color='k', width=4, linewidth=0, alpha=0.7)
            axs[x].text(trackertime_s[int(start_anti)]-25, -35, "Start anticipation = %0.2f ms"%(start_anti-debut),
                        color='k', alpha=0.7, fontsize=t_label/1.5, ha='right')

            # ---------------------------------
            # latence
            # ---------------------------------
            axs[x].bar(trackertime_s[int(latence)], 80, bottom=-40, color='firebrick', width=4, linewidth=0, alpha=1)
            axs[x].text(trackertime_s[int(latence)]+25, -35, "Latency = %0.2f ms"%(latence-debut),
                        color='firebrick', fontsize=t_label/1.5, va='center')
            
            # ---------------------------------
            # tau
            # ---------------------------------
            axs[x].plot(trackertime_s[int(latence):int(latence)+250],
                        result_deg.best_fit[int(latence):int(latence)+250], c='darkred', linewidth=2)
            axs[x].annotate(r'$\tau$', xy=(trackertime_s[int(latence)]+50, result_deg.best_fit[int(latence)+50]), xycoords='data', fontsize=t_label/1., color='darkred', va='bottom',
                    xytext=(trackertime_s[int(latence)]+70, result_deg.best_fit[int(latence)]), textcoords='data', arrowprops=dict(arrowstyle="->", color='darkred'))
            axs[x].text(trackertime_s[int(latence)]+70+t_label, (result_deg.best_fit[int(latence)]),
                        r"= %0.2f"%(tau), color='darkred',va='bottom', fontsize=t_label/1.5)
            
            # ---------------------------------
            # Max
            # ---------------------------------
            axs[x].plot(trackertime_s[int(latence):], np.ones(len(trackertime_s[int(latence):]))*result_deg.best_fit[int(latence)], '--k', linewidth=1, alpha=0.5)
            axs[x].plot(trackertime_s[int(latence):], np.ones(len(trackertime_s[int(latence):]))*result_deg.best_fit[int(latence)+250], '--k', linewidth=1, alpha=0.5)
            axs[x].text(TargetOn_s+450+25, (result_deg.best_fit[int(latence)]+result_deg.best_fit[int(latence)+250])/2,
                        "Max = %0.2f °/s"%(-maxi), color='k', va='center', fontsize=t_label/1.5)
            axs[x].annotate('', xy=(TargetOn_s+450, result_deg.best_fit[int(latence)]), xycoords='data', fontsize=t_label/1.5,
                        xytext=(TargetOn_s+450, result_deg.best_fit[int(latence)+250]), textcoords='data', arrowprops=dict(arrowstyle="<->"))

            # -------------------------------------------------------------------------------------
            #axs[x].axis([StimulusOn_s-10, TargetOff_s+10, -40, 40])
            axs[x].axis([-750, 750, -39.5, 39.5])    
            axs[x].xaxis.set_ticks_position('bottom')
            axs[x].xaxis.set_tick_params(labelsize=t_label/2)
            axs[x].yaxis.set_ticks_position('left')
            axs[x].yaxis.set_tick_params(labelsize=t_label/2)

            axs[x].set_xlabel('Time (ms)', fontsize=t_label)
            if x == int((len(liste)-1)/2) :
                axs[x].set_ylabel('Velocity (°/s)', fontsize=t_label)
            if x!= (len(liste)-1) : 
                axs[x].set_xticklabels([])
            if x==0 :
                axs[x].set_title('Velocity Fit', fontsize=t_titre, x=0.5, y=1.05)

            x=x+1

        plt.tight_layout() # pour supprimer les marge trop grande
        plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

        return fig, axs

    def plot_bcp(self, N_scan=100, pause=None, mode='expectation', max_run_length=150, fig_width=15, t_titre=35, t_label=20):
        
        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        from scipy.stats import beta

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']
        tau = N_trials/5.
        h = 1/tau

        #---------------------------------------------------------------------------
        # SCORE
        #---------------------------------------------------------------------------
        hs = h*np.logspace(-1, 1, N_scan)
        modes = ['expectation', 'max']
        score = np.zeros((len(modes), N_scan, N_blocks))
        for i_mode, m in enumerate(modes):
            for i_block in range(N_blocks):
                o = p[:, i_block, 0]
                for i_scan, h_ in enumerate(hs):
                    p_bar, r, beliefs = bcp.inference(o, h=h_, p0=.5)
                    p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=m)
                    score[i_mode, i_scan, i_block] = np.mean(np.log2(1.e-12+bcp.likelihood(o, p_hat, r_hat)))
        #---------------------------------------------------------------------------

        for block in range(N_blocks) :

            fig, axs = plt.subplots(3, 1, figsize=(fig_width, (fig_width)/((1.6180*6)/2)))
            axs[0] = plt.subplot(221)
            axs[1] = plt.subplot(223)
            axs[2] = plt.subplot(143)
            plt.suptitle('Block %s'%(block), fontsize=t_label/2, y=1.05, x=0, ha='left')

            #---------------------------------------------------------------------------
            # affiche la proba réel et les mouvements de la cible
            #---------------------------------------------------------------------------
            o = p[:, block, 0]
            p_true = p[:, block, 1]
            
            axs[0].step(range(N_trials), o, lw=1, alpha=.2, c='k')
            axs[0].step(range(N_trials), p_true, lw=1, alpha=.5, c='k')
            axs[0].fill_between(range(N_trials), np.zeros_like(o), o, lw=.5, alpha=.2, facecolor='k', step='pre')
            axs[0].fill_between(range(N_trials), np.zeros_like(p_true), p_true, lw=.5, alpha=.2, facecolor='k', step='pre')

            #---------------------------------------------------------------------------
            # P_HAT
            #---------------------------------------------------------------------------
            if pause is not None :
                liste = [0,50,100,150,200]
                for a in range(len(liste)-1) :
                    p_bar, r, beliefs = bcp.inference(p[liste[a]:liste[a+1], block, 0], h=h, p0=.5)
                    p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=mode)
                    p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
                    for i_trial in range(50):#N_trials):
                        p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])
                    axs[0].plot(np.arange(liste[a], liste[a+1]), p_hat, c='r',  lw=1)
                    axs[0].plot(np.arange(liste[a], liste[a+1]), p_sup, 'r--', lw=1)
                    axs[0].plot(np.arange(liste[a], liste[a+1]), p_low, 'r--', lw=1)
                    axs[0].fill_between(np.arange(liste[a], liste[a+1]), p_sup, p_low, lw=.5, alpha=.2, facecolor='r')

                    axs[1].imshow(np.log(beliefs[:max_run_length, :]+ 1.e-5), cmap='Greys',
                                  extent=(liste[a],liste[a+1], np.max(r), np.min(r)))
                    axs[1].plot(np.arange(liste[a], liste[a+1]), r_hat, lw=1, alpha=.9, c='r')

                for a in range(2):
                    axs[a].bar(50, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=0, linewidth=2)
                    axs[a].bar(100, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=0, linewidth=2)
                    axs[a].bar(150, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=0, linewidth=2)

            #---------------------------------------------------
            else :
                p_bar, r, beliefs = bcp.inference(o, h=h, p0=.5)
                p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=mode)

                p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
                for i_trial in range(N_trials):
                    p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

                axs[0].plot(range(N_trials), p_hat, lw=1, alpha=.9, c='r')
                axs[0].plot(range(N_trials), p_sup, 'r--', lw=1, alpha=.9)
                axs[0].plot(range(N_trials), p_low, 'r--', lw=1, alpha=.9)
                axs[0].fill_between(range(N_trials), p_low, p_sup, lw=.5, alpha=.2, facecolor='r')

                axs[1].imshow(np.log(beliefs[:max_run_length, :] + 1.e-5 ), cmap='Greys')
                axs[1].plot(range(N_trials), r_hat, lw=1, alpha=.9, c='r')

            #---------------------------------------------------------------------------
            # affiche SCORE
            #---------------------------------------------------------------------------
            if mode=='expectation' :
                i_mode = 0
            else :
                i_mode = 1

            axs[2].plot(hs, np.mean(score[i_mode, ...], axis=1), c='r', label=mode)
            axs[2].fill_between(hs,np.std(score[i_mode, ...], axis=1)+np.mean(score[i_mode, ...], axis=1), -np.std(score[i_mode, ...], axis=1)+np.mean(score[i_mode, ...], axis=1),  lw=.5, alpha=.2, facecolor='r', step='mid')

            axs[2].vlines(h, ymin=np.nanmin(score), ymax=np.nanmax(score), lw=2, label='true')
            axs[2].set_xscale("log")

            #---------------------------------------------------------------------------
            # cosmétique
            #---------------------------------------------------------------------------
            for i_layer, label in zip(range(2), ['$\hat{P}$ +/- CI', 'belief on r=p(r)']):
                axs[i_layer].set_xlim(0, N_trials)
                axs[i_layer].axis('tight')
                axs[i_layer].set_ylabel(label, fontsize=t_label/2)

            axs[0].set_ylim(-.05, 1 + .05)
            axs[0].set_yticks(np.arange(0, 1 + .05, 1/2))
            axs[0].set_xticks([])
            axs[0].set_xticklabels([])

            axs[1].set_ylim(-.05*140, 140 + (.05*140))
            axs[1].set_yticks(np.arange(0, 140 + (.05*140), 140/2))
            axs[1].set_xlabel('trials', fontsize=t_label/2);
            axs[1].set_xticks([-1, 49, 99,149])
            axs[1].set_xticklabels([0, 50, 100, 150], ha='left')

            axs[2].set_xlabel('Hazard rate', fontsize=t_label/2)
            axs[2].set_ylabel('Mean log-likelihood (bits)', fontsize=t_label/2)
            axs[2].legend(frameon=False, loc="lower left")

            for i_layer in range(len(axs)) :      
                axs[i_layer].xaxis.set_ticks_position('bottom')
                axs[i_layer].yaxis.set_ticks_position('left')

            fig.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            #---------------------------------------------------------------------------

            plt.show()

        return fig, axs

    def plot_bcp_2(self, block, trial=50, max_run_length=150, fig_width=15, t_titre=35, t_label=25):
        
        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        import matplotlib.gridspec as gridspec
        from scipy.stats import beta

        print('Block', block)
        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        tau = N_trials/5.
        h = 1/tau

        p = self.exp['p']
        o = p[:, block, 0]
        p_true = p[:, block, 1]

        #------------------------------------------------
        fig, axs = plt.subplots(5, 1, figsize=(fig_width, (fig_width)/((1.6180))), sharex=True)

        gs1 = gridspec.GridSpec(2, 1)
        gs1.update(left=0, bottom=1/2, right=1, top=1., hspace=0.05)
        axs[0] = plt.subplot(gs1[0])
        axs[1] = plt.subplot(gs1[1])

        gs2 = gridspec.GridSpec(2, 1)
        gs2.update(left=0, bottom=-0.16, right=1, top=(1/2)-0.16, hspace=0.05)
        axs[2] = plt.subplot(gs2[0])
        axs[3] = plt.subplot(gs2[1])

        gs3 = gridspec.GridSpec(1, 1)
        gs3.update(left=0, bottom=-(0.82)/1.5, right=1, top=-0.32, wspace=0.05)
        axs[4] = plt.subplot(gs3[0])
        #------------------------------------------------

        for x, mode in enumerate(['expectation', 'max']) :

            if x == 0 :
                a=0
            else:
                a=1

            #------------------------------------------------
            # affiche la proba réel et les mouvements de la cible
            #------------------------------------------------
            axs[a+x].step(range(N_trials), o, lw=1, alpha=.15, c='k')
            axs[a+x].step(range(N_trials), p_true, lw=1, alpha=.13, c='k')
            axs[a+x].fill_between(range(N_trials), np.zeros_like(o), o, lw=0, alpha=.15, facecolor='k', step='pre')
            axs[a+x].fill_between(range(N_trials), np.zeros_like(p_true), p_true, lw=0, alpha=.13, facecolor='k', step='pre')

            #------------------------------------------------
            # P_HAT
            #------------------------------------------------
            p_bar, r, beliefs = bcp.inference(o, h=h, p0=.5)
            p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=mode)
            
            p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
            for i_trial in range(N_trials):
                p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

            axs[a+x].plot(range(N_trials), p_hat, lw=1.5, c='darkred')
            axs[a+x].plot(range(N_trials), p_sup, c='darkred', ls='--', lw=1.2)
            axs[a+x].plot(range(N_trials), p_low, c='darkred', ls='--', lw=1.2)
            axs[a+x].fill_between(range(N_trials), p_low, p_sup, lw=.5, alpha=.11, facecolor='darkred')

            axs[(a+1)+x].imshow(np.log(beliefs[:max_run_length, :] + 1.e-5 ), cmap='Greys')

            #------------------------------------------------
            # Belief on r for trial view_essai
            #------------------------------------------------
            r_essai = (beliefs[:, trial])
            axs[4].plot(r_essai, c='k')
            axs[4].spines['top'].set_color('none')
            axs[4].spines['right'].set_color('none')
            
            axs[(a+1)+x].plot(range(N_trials), r_hat, lw=1.5, c='r')

            #---------------------------------------------------------------------------
            # cosmétique
            #---------------------------------------------------------------------------
            for i_layer, label in zip(range(2), ['$\hat{P}$ $\pm$ CI', 'belief on r = p(r)']) :
                axs[i_layer+a+x].set_xlim(0, N_trials)
                axs[i_layer+a+x].axis('tight')
                axs[i_layer+a+x].set_ylabel(label, fontsize=t_label/1.5)

                axs[i_layer+a+x].xaxis.set_ticks_position('bottom')
                axs[i_layer+a+x].yaxis.set_ticks_position('left')

            axs[(a+1)+x].bar(trial-1, 140 + (.05*140)+.05*140, bottom=-.05*140, color='firebrick', width=0.5, linewidth=0, alpha=1)
            axs[(a+1)+x].set_ylim(-.05*140, 140 + (.05*140))
            axs[(a+1)+x].set_yticks(np.arange(0, 140 + (.05*140), 140/2))
            axs[(a+1)+x].yaxis.set_tick_params(labelsize=t_label/2)
            axs[(a+1)+x].set_xlabel('Trials', fontsize=t_label);

            axs[a+x].set_ylim(-.05, 1 + .05)
            axs[a+x].set_yticks(np.arange(0, 1 + .05, 1/2))
            axs[a+x].yaxis.set_tick_params(labelsize=t_label/2)

            axs[(a+1)+x].set_xticks([-1, 49, 99,149])
            axs[(a+1)+x].set_xticklabels([0, 50, 100, 150], ha='left', fontsize=t_label/2)
            axs[a+x].set_xticks([])
            axs[a+x].set_xticklabels([])

            if mode == 'expectation' :
                axs[a+x].set_title('Bayesian change point : expectation $\sum_{r=0}^\infty r p(r)$', x=0.5, y=1.20, fontsize=t_titre)
            else :
                axs[a+x].set_title('Bayesian change point : max(p(r))', x=0.5, y=1.05, fontsize=t_titre)

        for i_layer in range(len(axs)) :
            axs[i_layer].xaxis.set_ticks_position('bottom')
            axs[i_layer].yaxis.set_ticks_position('left')

        axs[4].set_xscale('log')
        axs[4].set_xlim(0, max_run_length)

        axs[4].set_xlabel('r$_{%s}$'%(trial), fontsize=t_label/1.5)
        axs[4].set_ylabel('p(r$_{%s}$)'%(trial), fontsize=t_label/1.5)
        axs[4].set_title('Belief on r for trial %s'%(trial), x=0.5, y=1., fontsize=t_titre/1.2)
        axs[4].xaxis.set_tick_params(labelsize=t_label/1.9)
        axs[4].yaxis.set_tick_params(labelsize=t_label/1.9)
        #---------------------------------------------------------------------------

        return fig, axs

    def plot_results_2(self, mode, kde=None, tau=40., sujet=[6], fig_width=15, t_titre=35, t_label=25) :

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import bayesianchangepoint as bcp
        from scipy import stats

        colors = ['black','dimgrey','grey','darkgrey','silver','rosybrown','lightcoral','indianred','firebrick','brown','darkred','red']
        nb_sujet = len(self.PARI)
        full_proba, full_bino, full_results, full_va, proba_sujet, bino_sujet, results_sujet, va_sujet, full_p_hat_e, full_p_hat_m, p_hat_sujet_e, p_hat_sujet_m = liste_tout(self.PARI, self.ENREGISTREMENT, P_HAT=True)

        fig, axs = plt.subplots(len(sujet)+2, 1, figsize=(fig_width, fig_width/(1.6180)))
        fig.subplots_adjust(left = 0, bottom = -1/2+(((len(sujet))*2/3)-0.16), right = 1, top =len(sujet))

        gs1 = gridspec.GridSpec(len(sujet), 1)
        gs1.update(left=0, bottom=(len(sujet))*2/3, right=1, top=len(sujet), hspace=0.05)
        for s in range(len(sujet)) :
            axs[s] = plt.subplot(gs1[s])

        gs2 = gridspec.GridSpec(1, 2)
        gs2.update(left=0, bottom=-1/2+(((len(sujet))*2/3)-0.16), right=1, top=((len(sujet))*2/3)-0.16, wspace=0.2)
        axs[len(sujet)] = plt.subplot(gs2[0])
        axs[len(sujet)+1] = plt.subplot(gs2[1])

        ec = 0.2 # pour l'écart entre les différents blocks
        for s in range(len(sujet)) :
            print(sujet[s], '=', self.PARI[sujet[s]]['observer'])
            N_trials = self.PARI[sujet[s]]['N_trials']
            N_blocks = self.PARI[sujet[s]]['N_blocks']
            p = self.PARI[sujet[s]]['p']
            # tau = N_trials/5.
            h = 1./tau 
            results = (self.PARI[sujet[s]]['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
            v_anti = self.ENREGISTREMENT[sujet[s]]['v_anti']

            for block in range(N_blocks) :
                #----------------------------------------------------------------------------------
                liste = [0,50,100,150,200]
                for a in range(len(liste)-1) :
                    p_bar, r, beliefs = bcp.inference(p[liste[a]:liste[a+1], block, 0], h=h, p0=.5)
                    p_hat, r_hat = bcp.readout(p_bar, r, beliefs,mode=mode)
                    p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
                    for i_trial in range(50):
                        p_low[i_trial], p_sup[i_trial] = stats.beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

                    # Pour éviter d'avoir 36 légendes
                    if block == 0 :
                        if a == 0 :
                            axs[s].plot(np.arange(liste[a], liste[a+1]), block+p_hat+ec*block,
                                        c='darkred', alpha=.9, lw=1.5, label='$\hat{p}_{%s}$'%(mode))
                        else :
                            axs[s].plot(np.arange(liste[a], liste[a+1]), block+p_hat+ec*block,
                                        c='darkred', lw=1.5)
                    else :
                        axs[s].plot(np.arange(liste[a], liste[a+1]), block+p_hat+ec*block,
                                    c='darkred', lw=1.5)

                    axs[s].plot(np.arange(liste[a], liste[a+1]), block+p_sup+ec*block,
                                c='darkred', ls='--', lw=1.2)
                    axs[s].plot(np.arange(liste[a], liste[a+1]), block+p_low+ec*block,
                                c='darkred', ls= '--', lw=1.2)
                    axs[s].fill_between(np.arange(liste[a], liste[a+1]), block+p_sup+ec*block,
                                        block+p_low+ec*block, lw=.5, alpha=.11,
                                        facecolor='darkred')

                #----------------------------------------------------------------------------------
                axs[s].step(range(N_trials), block+p[:, block, 1]+ec*block, lw=1, alpha=0.13, c='k')
                axs[s].fill_between(range(N_trials), block+np.zeros_like(p[:, block, 1])+ec*block,
                                    block+p[:, block, 1]+ec*block,
                                    lw=0, alpha=.13, facecolor='black', step='pre')

                # Pour éviter d'avoir 36 légendes
                if block == 0 :
                    axs[s].step(range(N_trials), block+results[:, block]+ec*block, color='r',
                                lw=1.2, label='Individual guess')
                    axs[s].step(range(N_trials),
                                block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                                color='k', lw=1.2, label='Eye movements')
                else :
                    axs[s].step(range(N_trials), block+results[:, block]+ec*block, lw=1.2, color='r')
                    axs[s].step(range(N_trials),
                                block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                                color='k', lw=1.2)

            #------------------------------------------------
            # Barre Pause
            #------------------------------------------------
            axs[s].bar(49, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)
            axs[s].bar(99, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)
            axs[s].bar(149, 3+ec*3, bottom=-ec/2, color='k', width=0, linewidth=2)

            #------------------------------------------------
            # affiche les numéro des block sur le côté gauche
            #------------------------------------------------
            ax_block = axs[s].twinx()
            if s == 0 :
                ax_block.set_ylabel('Block', fontsize=t_label/1.5, rotation='horizontal', ha='left', va='bottom')
            ax_block.yaxis.set_label_coords(1.01, 1.08)
            ax_block.set_ylim(0, N_blocks)
            ax_block.set_yticks(np.arange(N_blocks)+0.5)
            ax_block.set_yticklabels(np.arange(N_blocks)+1, fontsize=t_label/1.5)
            ax_block.yaxis.set_tick_params(width=0, pad=(t_label/1.5)+10)

            #------------------------------------------------
            axs[s].set_yticks([0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2])
            axs[s].set_yticklabels(['0','1','0','1','0','1'],fontsize=t_label/2)
            axs[s].yaxis.set_label_coords(-0.02, 0.5)
            axs[s].set_ylabel('Subject %s'%(sujet[s]), fontsize=t_label)
            axs[s].set_ylim(-(ec/2), N_blocks +ec*3-(ec/2))
            #------------------------------------------------

        if mode=='expectation' :
            p_hat_sujet = p_hat_sujet_e
            full_p_hat = full_p_hat_e
        elif mode=='max' :
            p_hat_sujet = p_hat_sujet_m
            full_p_hat = full_p_hat_m
        
        #------------------------------------------------
        # SCATTER Plot
        #------------------------------------------------
        if kde is None :
            for x, color in enumerate(colors[:nb_sujet]):
                axs[len(sujet)].scatter(p_hat_sujet[x], results_sujet[x], c=color, alpha=0.5, linewidths=0)
                axs[len(sujet)+1].scatter(p_hat_sujet[x], va_sujet[x], c=color, alpha=0.5, linewidths=0)
        
        #------------------------------------------------
        # KDE
        #------------------------------------------------
        else :
            x = full_p_hat
            y = full_results
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            xmin, xmax = np.min(x), np.max(x)# -0.032, 1.032
            ymin, ymax =  np.min(y), np.max(y)#-0.032, 1.032
            xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)
            axs[len(sujet)].contourf(xx, yy, f, cmap='Greys', N=25)
            
            # masque les essais qui où full_va = NAN
            full_p_hat_nan = np.ma.masked_array(full_p_hat, mask=np.isnan(full_va)).compressed()
            full_va_nan = np.ma.masked_array(full_va, mask=np.isnan(full_va)).compressed()

            x = full_p_hat_nan
            y = full_va_nan
            values = np.vstack([x, y])
            kernel = stats.gaussian_kde(values)
            xmin, xmax = np.min(x), np.max(x)#-0.032, 1.032
            ymin, ymax = np.min(y), np.max(y)#-21.28, 21.28
            xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)
            axs[len(sujet)+1].contourf(xx, yy, f, cmap='Greys', N=25)
        
        #------------------------------------------------
        # LINREGRESS
        #------------------------------------------------
        # RESULTS
        slope, intercept, r_, p_value, std_err = stats.linregress(full_p_hat, full_results)
        x_test = np.linspace(np.min(full_p_hat), np.max(full_p_hat), 100)
        fitLine = slope * x_test + intercept
        axs[len(sujet)].plot(x_test, fitLine, c='k', linewidth=2)
        axs[len(sujet)].text(0.75,-0.032+(1.032--0.032)/10, 'r = %0.3f'%(r_), fontsize=t_label/1.2)

        # VA
        # masque les essais qui où full_va = NAN
        full_p_hat_nan = np.ma.masked_array(full_p_hat, mask=np.isnan(full_va)).compressed()
        full_va_nan = np.ma.masked_array(full_va, mask=np.isnan(full_va)).compressed()
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(full_p_hat_nan, full_va_nan)
        x_test = np.linspace(np.min(full_p_hat), np.max(full_p_hat), 100)
        fitLine = slope * x_test + intercept
        axs[len(sujet)+1].plot(x_test, fitLine, c='k', linewidth=2)
        axs[len(sujet)+1].text(0.75,-21.28+(21.28--21.28)/10, 'r = %0.3f'%(r_value), fontsize=t_label/1.2)

        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        axs[len(sujet)].axis([-0.032, 1.032, -0.032, 1.032])
        axs[len(sujet)].set_ylabel('Probability Bet', fontsize=t_label/1.2)
        axs[len(sujet)].set_title("Probability Bet", fontsize=t_titre/1.2, x=0.5, y=1.05)
        axs[len(sujet)].set_xlabel('$\hat{P}_{%s}$'%(mode), fontsize=t_label/1) 
        
        axs[len(sujet)+1].axis([-0.032, 1.032, -21.28, 21.28])
        axs[len(sujet)+1].set_ylabel('Acceleration of anticipation (°/s$^2$)', fontsize=t_label/1.2)
        axs[len(sujet)+1].set_title("Acceleration", fontsize=t_titre/1.2, x=0.5, y=1.05)
        axs[len(sujet)+1].set_xlabel('$\hat{P}_{%s}$'%(mode), fontsize=t_label/1)    

        for i_layer in range(len(axs)) :
            axs[i_layer].xaxis.set_ticks_position('bottom')
            axs[i_layer].yaxis.set_ticks_position('left')
            axs[i_layer].xaxis.set_tick_params(labelsize=t_label/1.8)
            axs[i_layer].yaxis.set_tick_params(labelsize=t_label/1.8)
            if i_layer < len(sujet)-1 :
                axs[i_layer].set_xticks([])
            elif i_layer == len(sujet)-1 :
                axs[i_layer].set_xlabel('Trials', fontsize=t_label)
                axs[i_layer].set_xticks([-1, 49, 99,149])
                axs[i_layer].set_xticklabels([0, 50, 100, 150], ha='left',fontsize=t_label/1.8)

        axs[0].legend(fontsize=t_label/1.2, bbox_to_anchor=(0., 1.05, 1, 0.), loc=4, ncol=3,
                  mode="expand", borderaxespad=0.)
        axs[0].set_title('Results bayesian change point %s'%(mode), fontsize=t_titre, x=0.5, y=1.3)
        #------------------------------------------------

        return fig, axs



if __name__ == '__main__':

    try:
        mode = sys.argv[1]
    except:
        mode = 'pari'
        #mode = 'enregistrement'
    try:
        timeStr = sys.argv[4]
    except:
        import time
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        #timeStr = '2017-06-22_102207'

    e = aSPEM(mode, timeStr)
    if not mode is 'model':
        print('Starting protocol')
        e.run_experiment()
