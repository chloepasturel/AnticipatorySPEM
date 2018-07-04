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

        #logging.console.setLevel(logging.WARNING)
        #if verb: print('launching experiment')
        #logging.console.setLevel(logging.WARNING)
        #if verb: print('go!')

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






#############################################################################
######################### ANALYSIS ##########################################
#############################################################################

def full_liste(PARI, ENREGISTREMENT, P_HAT=None):

    import pandas as pd
    pd.set_option('mode.chained_assignment', None)

    if P_HAT is not None :
        import bayesianchangepoint as bcp
        full = pd.DataFrame(index=np.arange(len(PARI)*600),columns=('sujet', 'proba','bino','results','va','p_hat_e','p_hat_m', 'p_hat_f'))
    else :
        full = pd.DataFrame(index=np.arange(len(PARI)*600),columns=('sujet', 'proba','bino','results','va'))

    for x in range(len(PARI)):

        N_trials = PARI[x]['N_trials']
        N_blocks = PARI[x]['N_blocks']

        p = PARI[x]['p']
        results = (PARI[x]['results']+1)/2
        v_anti = ENREGISTREMENT[x]['v_anti']

        for block in range(N_blocks):

            nb = x*N_trials*N_blocks
            a = nb + N_trials*block
            b = (nb + N_trials*(block+1))

            full['sujet'][a:b] = PARI[x]['observer']
            full['proba'][a:b] = p[:, block, 1]
            full['bino'][a:b] = p[:, block, 0]
            full['results'][a:b] = results[:, block]
            full['va'][a:b] = v_anti[block]

            if P_HAT is not None :
                tau = N_trials/5.
                h = 1./tau
                liste = [0,50,100,150,200]
                p_hat_block_e = []
                p_hat_block_m = []
                p_hat_block_f = []

                for s in range(len(liste)-1) :
                    p_bar, r, beliefs = bcp.inference(p[liste[s]:liste[s+1], block, 0], h=h, p0=.5)
                    p_hat_e, r_hat_e = bcp.readout(p_bar, r, beliefs, mode='expectation')
                    p_hat_m, r_hat_m = bcp.readout(p_bar, r, beliefs, mode='max')
                    p_hat_f, r_hat_f = bcp.readout(p_bar, r, beliefs, mode='fixed')

                    p_hat_block_e.extend(p_hat_e)
                    p_hat_block_m.extend(p_hat_m)
                    p_hat_block_f.extend(p_hat_f)

                full['p_hat_e'][a:b] = p_hat_block_e
                full['p_hat_m'][a:b] = p_hat_block_m
                full['p_hat_f'][a:b] = p_hat_block_f
    return full

def mutual_information(hgram):
    """ Mutual information for joint histogram
    https://matthew-brett.github.io/teaching/mutual_information.html"""
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))



def scatter(self, ax, full, p_hat, colors, nb_sujet, result):
    #------------------------------------------------
    # SCATTER Plot
    #------------------------------------------------
    for x, color in enumerate(colors[:nb_sujet]):
        s = self.PARI[x]['observer']
        if result=='bet' :
            ax.scatter(full[full.sujet==s][p_hat], full[full.sujet==s]['results'], c=color, alpha=0.5, linewidths=0)
        elif result=='acceleration' :
            ax.scatter(full[full.sujet==s][p_hat], full[full.sujet==s]['va'], c=color, alpha=0.5, linewidths=0)
    return ax

def KDE(ax, x, y, xmin, xmax, ymin, ymax, mode_kde) :
    from scipy import stats
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    if mode_kde=='normal':
        ax.contourf(xx, yy, f, cmap='Greys', N=25)
    elif mode_kde=='mean':
        fmean = []
        for x in range(len(f)):
            fmean.append([])
            for y in range(len(f[x])):
                fmean[x].append(f[x][y]/np.sum(f[x]))
        ax.contourf(xx, yy, fmean, cmap='Greys')
    return ax

def regress(ax, p, data, y1, y2, t_label) :
    from scipy import stats
    slope, intercept, r_, p_value, std_err = stats.linregress(p, data)
    x_test = np.linspace(np.min(p), np.max(p), 100)
    fitLine = slope * x_test + intercept
    ax.plot(x_test, fitLine, c='k', linewidth=2)
    ax.text(0.75,y1+(y2-y1)/10, 'r = %0.3f'%(r_), fontsize=t_label/1.2)

    hist, x_edges, y_edges = np.histogram2d(p, data ,bins=20)
    ax.text(0.75,y1+2*(y2-y1)/10, 'MI = %0.3f'%(mutual_information(hist)), fontsize=t_label/1.2)

    return ax


def scatter_KDE(self, ax, mode_bcp, plot, mode_kde, result, t_titre, t_label) :

    colors = ['black','dimgrey','grey','darkgrey','silver','rosybrown','lightcoral','indianred','firebrick','brown','darkred','red']
    nb_sujet = len(self.PARI)
    full = full_liste(self.PARI, self.ENREGISTREMENT, P_HAT=True)

    if mode_bcp=='expectation' :
        p_hat = 'p_hat_e'
        full_p_hat = full['p_hat_e']
    elif mode_bcp=='max' :
        p_hat = 'p_hat_m'
        full_p_hat = full['p_hat_m']
    elif mode_bcp=='fixed' :
        p_hat = 'p_hat_f'
        full_p_hat = full['p_hat_f']
    elif mode_bcp=='reel' :
        p_hat = 'proba'
        full_p_hat = full['proba']

    if plot=='scatter' :
        ax = scatter(self, ax, full, p_hat, colors, nb_sujet, result)

    if plot=='kde':
        #------------------------------------------------
        # KDE
        #------------------------------------------------
        xmin, xmax = -0.032, 1.032 # np.min(x), np.max(x)
        if result=='bet' :
            x = full_p_hat.values.tolist()
            y = full['results'].values.tolist()
            ymin, ymax =  -0.032, 1.032 #np.min(y), np.max(y)
            ax = KDE(ax, x, y, xmin, xmax, ymin, ymax, mode_kde)

        elif result=='acceleration' :
            # masque les essais qui où full_va = NAN
            x = np.ma.masked_array(full_p_hat.values.tolist(), mask=np.isnan(full['va'].values.tolist())).compressed()
            y = np.ma.masked_array(full['va'].values.tolist(), mask=np.isnan(full['va'].values.tolist())).compressed()
            ymin, ymax = -21.28, 21.28 #np.min(y), np.max(y)
            ax = KDE(ax, x, y, xmin, xmax, ymin, ymax, mode_kde)

    #------------------------------------------------
    # LINREGRESS
    #------------------------------------------------
    if result=='bet' :
        # RESULTS
        p = full_p_hat.values.tolist()
        data = full['results'].values.tolist()
        y1 = -0.032
        y2 = 1.032
        ax = regress(ax, p, data, y1, y2, t_label)

        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        ax.set_ylabel('Probability Bet', fontsize=t_label/1.2)
        ax.set_title("Probability Bet", fontsize=t_titre/1.2, x=0.5, y=1.05)

    elif result=='acceleration' :
        # VA
        # masque les essais qui où full_va = NAN
        p = np.ma.masked_array(full_p_hat.values.tolist(), mask=np.isnan(full['va'].values.tolist())).compressed()
        data = np.ma.masked_array(full['va'].values.tolist(), mask=np.isnan(full['va'].values.tolist())).compressed()
        y1 = -21.28
        y2 = 21.28
        ax = regress(ax, p, data, y1, y2, t_label)

        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        ax.set_ylabel('Acceleration of anticipation (°/s$^2$)', fontsize=t_label/1.2)
        ax.set_title("Acceleration", fontsize=t_titre/1.2, x=0.5, y=1.05)

    ax.axis([-0.032, 1.032, y1, y2])
    ax.set_xlabel('$\hat{P}_{%s}$'%(mode_bcp), fontsize=t_label/1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=t_label/1.8)
    ax.yaxis.set_tick_params(labelsize=t_label/1.8)
    #------------------------------------------------

    return ax


def results_sujet(self, ax, sujet, s, mode_bcp, tau, t_label):

    import bayesianchangepoint as bcp
    from scipy import stats

    color = [['k', 'k'], ['r', 'r'], ['k','w']]
    alpha = [[.35,.15],[.35,.15],[1,0]]
    lw = 1.3
    ec = 0.2 # pour l'écart entre les différents blocks


    print('Subject', sujet[s], '=', self.PARI[sujet[s]]['observer'])
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
            p_hat, r_hat = bcp.readout(p_bar, r, beliefs,mode=mode_bcp)
            p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
            for i_trial in range(50):
                p_low[i_trial], p_sup[i_trial] = stats.beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

            # Pour éviter d'avoir 36 légendes
            if block == 0 and a== 0 :
                ax.plot(np.arange(liste[a], liste[a+1]), block+p_hat+ec*block, c='darkred', alpha=.9, lw=1.5, label='$\hat{p}_{%s}$'%(mode_bcp))
            else :
                ax.plot(np.arange(liste[a], liste[a+1]), block+p_hat+ec*block, c='darkred', lw=1.5)

            ax.plot(np.arange(liste[a], liste[a+1]), block+p_sup+ec*block, c='darkred', ls='--', lw=1.2)
            ax.plot(np.arange(liste[a], liste[a+1]), block+p_low+ec*block, c='darkred', ls= '--', lw=1.2)
            ax.fill_between(np.arange(liste[a], liste[a+1]), block+p_sup+ec*block, block+p_low+ec*block, lw=.5, alpha=.11, facecolor='darkred')

        #----------------------------------------------------------------------------------
        ax.step(range(N_trials), block+p[:, block, 1]+ec*block, lw=1, alpha=alpha[1][0], c=color[1][0])
        ax.fill_between(range(N_trials), block+np.zeros_like(p[:, block, 1])+ec*block, block+p[:, block, 1]+ec*block,
                            lw=0, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
        ax.fill_between(range(N_trials), block+np.ones_like(p[:, block, 1])+ec*block, block+p[:, block, 1]+ec*block,
                            lw=0, alpha=alpha[1][1], facecolor=color[1][1], step='pre')
        # Pour éviter d'avoir 36 légendes
        if block == 0 :
            ax.step(range(N_trials), block+results[:, block]+ec*block, color='r', lw=1.2, label='Individual guess')
            ax.step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                        color='k', lw=1.2, label='Eye movements')
        else :
            ax.step(range(N_trials), block+results[:, block]+ec*block, lw=1.2, color='r')
            ax.step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                        color='k', lw=1.2)


    #------------------------------------------------
    # affiche les numéro des block sur le côté gauche
    #------------------------------------------------
    ax_block = ax.twinx()
    if s == 0 :
        ax_block.set_ylabel('Block', fontsize=t_label/1.5, rotation='horizontal', ha='left', va='bottom')
    ax_block.yaxis.set_label_coords(1.01, 1.08)
    ax_block.set_ylim(0, N_blocks)
    ax_block.set_yticks(np.arange(N_blocks)+0.5)
    ax_block.set_yticklabels(np.arange(N_blocks)+1, fontsize=t_label/1.5)
    ax_block.yaxis.set_tick_params(width=0, pad=(t_label/1.5)+10)

    #------------------------------------------------
    ax.set_yticks([0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2])
    ax.set_yticklabels(['0','1','0','1','0','1'],fontsize=t_label/2)
    ax.yaxis.set_label_coords(-0.02, 0.5)
    ax.set_ylabel('Subject %s'%(sujet[s]), fontsize=t_label)
    ax.set_ylim(-(ec/2), N_blocks +ec*3-(ec/2))
    ax.set_xlim(0, N_trials)
    #------------------------------------------------

    #------------------------------------------------
    # Barre Pause
    #------------------------------------------------
    ax.bar(50, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
    ax.bar(100, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
    ax.bar(150, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
    #ax.bar(49, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)

    return ax



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
                try :
                    with open(a, 'rb') as fichier :
                        b = pickle.load(fichier, encoding='latin1')
                        self.ENREGISTREMENT.append(b)
                except :
                    print('/!\ Le fichier param Fit n\'existe pas pour %s !'%(liste[x][1]))
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


    def plot_position(self, fig=None, axs=None, fig_width=10, t_label=20, file_fig=None) :

        # from pygazeanalyser.edfreader import read_edf
        from ANEMO import read_edf
        #from edfreader import read_edf
        from ANEMO import ANEMO

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
        bino = p[:,:,0]

        if file_fig is None :
            file_fig = 'figures/enregistrement_%s'%(self.observer)

        ANEMO.plot_position(data=data, N_trials=N_trials, N_blocks=N_blocks, bino=bino, V_X=V_X, RashBass=RashBass, stim_tau=stim_tau,
                            screen_width_px=screen_width_px, screen_height_px=screen_height_px, fig=fig, axs=axs, fig_width=fig_width, t_label=t_label, file_fig=file_fig)

        #        for f in range(len(fixations)) :
        #            axs[trial]. axvspan(fixations[f][0]-start, fixations[f][1]-start, color='r', alpha=0.1)
        #        for s in range(len(saccades)) :
        #            axs[trial]. axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)

        #    plt.tight_layout() # pour supprimer les marge trop grande
        #    plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

        #    plt.savefig('figures/enregistrement_%s_%s.pdf'%(self.observer, block+1))
        #plt.close()
        #return fig, axs

    def plot_velocity(self, block=0, trials=0, report=None, fig_width=15, t_titre=35, t_label=20):
        import matplotlib.pyplot as plt
        from ANEMO import read_edf
        from ANEMO import ANEMO

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        fig, axs = ANEMO.plot_velocity(data=data, trials=trials, block=block, N_trials=self.exp['N_trials'], px_per_deg=self.exp['px_per_deg'],
                                       fig_width=fig_width, t_titre=t_titre, t_label=t_label)

        return fig, axs

    def Fit (self, plot=None, fig_width=12, file_save=None, t_label=20, t_text=14, file_fig=None, list_events=None, sup=True, time_sup=-280, param_fit=None, stop_recherche_misac=None, step_fit=2) :

        import matplotlib.pyplot as plt
        from ANEMO import read_edf
        from ANEMO import ANEMO

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')
        
        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']
        px_per_deg = self.exp['px_per_deg']

        if file_fig is None :
            file_fig='figures/Fit_%s'%(self.observer)

        param = ANEMO.Fit (data=data, N_trials=N_trials, N_blocks=N_blocks, binomial=p[:,:,0], px_per_deg=px_per_deg, list_events=list_events,
                           sup=sup, time_sup=time_sup, observer=self.observer, plot=plot, fig_width=fig_width, t_label=t_label, t_text=t_text,
                           file_fig=file_fig, param_fit=param_fit, stop_recherche_misac=stop_recherche_misac, step_fit=step_fit)

        if file_save is None :
            file = os.path.join('parametre', 'param_Fit_' + self.observer + '.pkl')
        else :
            file = file_save

        with open(file, 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(param)

        print('FIN !!!')

    def plot_Fit(self, plot='fonction', block=0, trials=0, report=None, fig_width=15, t_titre=35, t_label=20):

        import matplotlib.pyplot as plt
        from ANEMO import read_edf
        #from edfreader import read_edf
        from ANEMO import ANEMO

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        N_trials = self.exp['N_trials']
        p = self.exp['p']
        px_per_deg = self.exp['px_per_deg']
        bino = p[:,:,0]

        if report is None :
            fig, axs = ANEMO.plot_Fit(data=data, bino=bino, trials=trials, block=block, N_trials=N_trials, px_per_deg=px_per_deg,
                                      plot=plot, fig_width=fig_width, t_titre=t_titre, t_label=t_label, report=report)
            return fig, axs
        else :
            fig, axs, results = ANEMO.plot_Fit(data=data, bino=bino, trials=trials, block=block, N_trials=N_trials, px_per_deg=px_per_deg,
                                               plot=plot, fig_width=fig_width, t_titre=t_titre, t_label=t_label, report=report)
            return fig, axs, results


    def plot_experiment(self, sujet=[0], mode_bcp='expectation', tau=40, direction=True, p=None, num_block=None, mode=None, fig=None, axs=None, fig_width=15, titre='Experiment', t_titre=35, t_label=25, return_proba=None):

        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        from scipy import stats
        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        h = 1./tau
        ec = 0.2
        if p is None :
            p = self.exp['p']
        if num_block is None :
            BLOCK = range(N_blocks)
        else:
            BLOCK = [num_block]

        if fig is None:
            fig_width= fig_width
            if len(sujet)==1 :
                fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_width/1.6180))
            else :
                if direction is True :
                    fig, axs = plt.subplots(len(sujet)+1, 1, figsize=(fig_width, ((len(sujet)+1)*fig_width/3)/(1.6180)))
                else :
                    fig, axs = plt.subplots(len(sujet), 1, figsize=(fig_width, ((len(sujet)+1)*fig_width/3)/(1.6180)))

        #color = [['r', 'b'], ['orange', 'g'], ['k','w']]
        #alpha = [[.2,.2],[.2,.2],[.2,.2]]
        #lw=1
        color = [['k', 'k'], ['r', 'r'], ['k','w']]
        alpha = [[.35,.15],[.35,.15],[1,0]]
        lw = 1.3

        for i_block in BLOCK:
            if len(sujet)==1 :
                for i_layer, label in enumerate(['Target Direction', 'Probability', 'Switch']) :
                    if label == 'Switch' : axs[i_layer].step(range(N_trials), p[:, i_block, i_layer]+i_block+ec*i_block, lw=1, c=color[i_layer][0], alpha=alpha[i_layer][0])
                    axs[i_layer].fill_between(range(N_trials), i_block+np.zeros_like(p[:, i_block, i_layer])+ec*i_block, i_block+p[:, i_block, i_layer]+ec*i_block,
                                              lw=.5, alpha=alpha[i_layer][0], facecolor=color[i_layer][0], step='pre')
                    axs[i_layer].fill_between(range(N_trials), i_block+np.ones_like(p[:, i_block, i_layer])+ec*i_block, i_block+p[:, i_block, i_layer]+ec*i_block,
                                              lw=.5, alpha=alpha[i_layer][1], facecolor=color[i_layer][1], step='pre')

                    axs[i_layer].set_ylabel(label, fontsize=t_label)
            else :
                if direction is True :
                    axs[0].step(range(N_trials), p[:, i_block, 0]+i_block+ec*i_block, lw=1, c=color[0][0], alpha=alpha[0][0])
                    axs[0].fill_between(range(N_trials), i_block+np.zeros_like(p[:, i_block, 0])+ec*i_block,
                                              i_block+p[:, i_block, 0]+ec*i_block,
                                              lw=.5, alpha=alpha[0][0], facecolor=color[0][0], step='pre')
                    axs[0].fill_between(range(N_trials), i_block+np.ones_like(p[:, i_block, 0])+ec*i_block,
                                              i_block+p[:, i_block, 0]+ec*i_block,
                                              lw=.5, alpha=alpha[0][1], facecolor=color[0][1], step='pre')


                    axs[0].set_ylabel('Target Direction', fontsize=t_label)
                for s in range(len(sujet)) :
                    if direction is True :
                        a = s+1
                    else :
                        a = s
                    axs[a].step(range(N_trials), p[:, i_block, 1]+i_block+ec*i_block, lw=1, c=color[1][0], alpha=alpha[1][0])
                    axs[a].fill_between(range(N_trials), i_block+np.zeros_like(p[:, i_block, 1])+ec*i_block, i_block+p[:, i_block, 1]+ec*i_block,
                                              lw=.5, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
                    axs[a].fill_between(range(N_trials), i_block+np.ones_like(p[:, i_block, 1])+ec*i_block, i_block+p[:, i_block, 1]+ec*i_block,
                                              lw=.5, alpha=alpha[1][1], facecolor=color[1][1], step='pre')
                    axs[a].set_yticklabels(['0','1','0','1','0','1'],fontsize=t_label/2)
                    axs[a].set_ylabel('Subject %s'%(sujet[s]), fontsize=t_label)

        #-------------------------------------------------------------------------------------------------------------
        for s in range(len(sujet)) :
            if direction is True :
                a = s+1
            else :
                a = s

            if len(sujet)==1:
                results = (self.exp['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
                v_anti = self.param['v_anti']
                print('sujet =', self.exp['observer'])
                y_t = 1.1
            else :
                p = self.PARI[sujet[s]]['p']
                results = (self.PARI[sujet[s]]['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
                v_anti = self.ENREGISTREMENT[sujet[s]]['v_anti']
                print('sujet', sujet[s], '=', self.PARI[sujet[s]]['observer'])
                y_t = 1.25
            #-------------------------------------------------------------------------------------------------------------
            if mode == 'pari' :
                for block in BLOCK:
                    if block == 0 :
                        axs[a].step(range(N_trials), block+results[:, block]+ec*block, lw=lw, alpha=1,
                                      color='r', label='Individual guess')
                    else :
                        axs[a].step(range(N_trials), block+results[:, block]+ec*block, lw=lw, alpha=1, color='r')
                axs[0].set_title('Bet results', fontsize=t_titre, x=0.5, y=y_t)

            #------------------------------------------------
            elif mode == 'enregistrement' :
                for block in BLOCK:
                    if block == 0 :
                        axs[a].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                                      color='k', lw=lw, alpha=1, label='Eye movement')
                    else :
                        axs[a].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                                      color='k', lw=lw, alpha=1)
                axs[0].set_title('Eye movements recording results', fontsize=t_titre, x=0.5, y=y_t)

            #------------------------------------------------
            elif mode=='deux':
                for block in BLOCK:
                    if block == 0 :
                        axs[a].step(range(N_trials), block+results[:, block]+ec*block, lw=lw, alpha=1,
                                      color='r', label='Individual guess')
                        axs[a].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,
                                      color='k', lw=lw, alpha=1, label='Eye movement')
                    else :
                        axs[a].step(range(N_trials), block+results[:, block]+ec*block, lw=lw, alpha=1, color='r')
                        axs[a].step(range(N_trials), block+((np.array(v_anti[block])-np.nanmin(v_anti))/(np.nanmax(v_anti)-np.nanmin(v_anti)))+ec*block,color='k', lw=lw, alpha=1)
                axs[0].set_title('Bet + Eye movements results', fontsize=t_titre, x=0.5, y=y_t)

            #------------------------------------------------
            elif mode is None and titre is not None :
                axs[0].set_title(titre, fontsize=t_titre, x=0.5, y=y_t)
            #-------------------------------------------------------------------------------------------------------------



        for i_layer in range(len(axs)):
            if num_block is None :
                #------------------------------------------------
                # Barre Pause
                #------------------------------------------------
                axs[i_layer].bar(49, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
                axs[i_layer].bar(99, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
                axs[i_layer].bar(149, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
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

            if i_layer==(len(axs)-1) :
                axs[i_layer].set_xticks([-1, 49, 99, 149, 199])
                axs[i_layer].set_xticklabels([0, 50, 100, 150, 200], ha='left', fontsize=t_label/2)
                axs[i_layer].yaxis.set_tick_params(width=0)
                axs[i_layer].xaxis.set_ticks_position('bottom')
            else :
                axs[i_layer].set_xticks([])

            axs[i_layer].set_ylim(-(ec/2), len(BLOCK) +ec*len(BLOCK)-(ec/2))

            y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]

            axs[i_layer].set_yticks(y_ticks[:len(BLOCK)*2])
            axs[i_layer].yaxis.set_label_coords(-0.05, 0.5)
            axs[i_layer].yaxis.set_tick_params(direction='out')
            axs[i_layer].yaxis.set_ticks_position('left')

        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        if len(sujet)==1 :
            axs[0].set_yticklabels(['left','right']*len(BLOCK),fontsize=t_label/2)
            axs[1].set_yticklabels(['0','1']*len(BLOCK),fontsize=t_label/2)
            axs[2].set_yticklabels(['No','Yes']*len(BLOCK),fontsize=t_label/2)
        else :
            if direction is True :
                axs[0].set_yticklabels(['left','right']*len(BLOCK),fontsize=t_label/2)
            else :
                axs[1].legend(fontsize=t_label/1.3, bbox_to_anchor=(0., 2.1, 1, 0.), loc=3, ncol=2, mode="expand", borderaxespad=0.)

        axs[-1].set_xlabel('Trials', fontsize=t_label)
        try:
            fig.tight_layout()
        except:
            print('tight_layout failed :-(')
        plt.subplots_adjust(hspace=0.05)
        #------------------------------------------------

        if return_proba is None :
            return fig, axs
        else :
            return fig, axs, p

    def plot_bcp(self, plot='normal', block=[0,1,2], trial=50, N_scan=100, pause=None, mode=['expectation', 'max'], max_run_length=150, fig_width=15, t_titre=35, t_label=20):

        '''plot='normal' -> bcp, 'detail' -> bcp2'''

        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        from scipy.stats import beta

        if type(block) is not list :
            block = [block]
        if type(mode) is not list :
            mode = [mode]

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']
        tau = N_trials/5.
        h = 1/tau

        if plot=='detail' :

            import matplotlib.gridspec as gridspec
            block=[block[0]]
            mode = ['expectation', 'max']
            print('Block', block[0])

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

        if plot=='normal' :
            #---------------------------------------------------------------------------
            # SCORE
            #---------------------------------------------------------------------------
            hs = h*np.logspace(-1, 1, N_scan)
            modes = ['expectation', 'max']
            score = np.zeros((len(modes), N_scan, N_blocks))
            for i_block in range(N_blocks):
                o = p[:, i_block, 0]
                for i_scan, h_ in enumerate(hs):
                    p_bar, r, beliefs = bcp.inference(o, h=h_, p0=.5)
                    for i_mode, m in enumerate(modes):
                        p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=m)
                        score[i_mode, i_scan, i_block] = np.mean(np.log2(1.e-12+bcp.likelihood(o, p_hat, r_hat)))
            #---------------------------------------------------------------------------

        for b in block :
            for x, m in enumerate(mode) :
                if plot=='normal':
                    #------------------------------------------------
                    fig, axs = plt.subplots(3, 1, figsize=(fig_width, (fig_width)/((1.6180*6)/2)))
                    axs[0] = plt.subplot(221)
                    axs[1] = plt.subplot(223)
                    axs[2] = plt.subplot(143)
                    plt.suptitle('Mode %s Block %s'%(m, (b+1)), fontsize=t_label, y=1.1, x=0.125, ha='left')
                    #------------------------------------------------
                    A=2
                    num = 0

                if plot=='detail' :
                    A=4
                    num=2*x

                #---------------------------------------------------------------------------
                # affiche la proba réel et les mouvements de la cible
                #---------------------------------------------------------------------------
                o = p[:, b, 0]
                p_true = p[:, b, 1]
                axs[num].step(range(N_trials), o, lw=1, alpha=.15, c='k')
                axs[num].step(range(N_trials), p_true, lw=1, alpha=.13, c='k')
                axs[num].fill_between(range(N_trials), np.zeros_like(o), o, lw=0, alpha=.15, facecolor='k', step='pre')
                axs[num].fill_between(range(N_trials), np.zeros_like(p_true), p_true, lw=0, alpha=.13, facecolor='k', step='pre')

                #---------------------------------------------------------------------------
                # P_HAT
                #---------------------------------------------------------------------------
                if pause is not None :
                    liste = [0,50,100,150,200]
                    for a in range(len(liste)-1) :
                        p_bar, r, beliefs = bcp.inference(p[liste[a]:liste[a+1], b, 0], h=h, p0=.5)
                        p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=m)
                        p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
                        for i_trial in range(50):#N_trials):
                            p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])
                        axs[num].plot(np.arange(liste[a], liste[a+1]), p_hat, c='darkred',  lw=1.5, alpha=.9)
                        axs[num].plot(np.arange(liste[a], liste[a+1]), p_sup, c='darkred', ls='--', lw=1.2)
                        axs[num].plot(np.arange(liste[a], liste[a+1]), p_low, c='darkred', ls='--', lw=1.2)
                        axs[num].fill_between(np.arange(liste[a], liste[a+1]), p_sup, p_low, lw=.5, alpha=.11, facecolor='darkred')
                        axs[num+1].imshow(np.log(beliefs[:max_run_length, :]+ 1.e-5), cmap='Greys',
                                      extent=(liste[a],liste[a+1], np.max(r), np.min(r)))
                        axs[num+1].plot(np.arange(liste[a], liste[a+1]), r_hat, lw=1.5, alpha=.9, c='r')

                    for a in range(A):
                        axs[a].bar(50, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=.2, linewidth=0)
                        axs[a].bar(100, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=.2, linewidth=0)
                        axs[a].bar(150, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=.2, linewidth=0)

                else :
                    p_bar, r, beliefs = bcp.inference(o, h=h, p0=.5)
                    p_hat, r_hat = bcp.readout(p_bar, r, beliefs, mode=m)

                    p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
                    for i_trial in range(N_trials):
                        p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

                    axs[num].plot(range(N_trials), p_hat, lw=1.5, alpha=.9, c='darkred')
                    axs[num].plot(range(N_trials), p_sup, c='darkred', ls='--', lw=1.2, alpha=.9)
                    axs[num].plot(range(N_trials), p_low, c='darkred', ls='--', lw=1.2, alpha=.9)
                    axs[num].fill_between(range(N_trials), p_low, p_sup, lw=.5, alpha=.11, facecolor='darkred')

                    axs[num+1].imshow(np.log(beliefs[:max_run_length, :] + 1.e-5 ), cmap='Greys')
                    axs[num+1].plot(range(N_trials), r_hat, lw=1.5, alpha=.9, c='r')

                #---------------------------------------------------------------------------
                # affiche SCORE
                #---------------------------------------------------------------------------
                if plot=='normal' :
                    if m=='expectation' :
                        i_mode = 0
                    else :
                        i_mode = 1

                    axs[2].plot(hs, np.mean(score[i_mode, ...], axis=1), c='r', label=m)
                    axs[2].fill_between(hs,np.std(score[i_mode, ...], axis=1)+np.mean(score[i_mode, ...], axis=1), -np.std(score[i_mode, ...], axis=1)+np.mean(score[i_mode, ...], axis=1),  lw=.5, alpha=.2, facecolor='r', step='mid')

                    axs[2].vlines(h, ymin=np.nanmin(score), ymax=np.nanmax(score), lw=2, label='true')
                    axs[2].set_xscale("log")

                #------------------------------------------------
                # Belief on r for trial view_essai
                #------------------------------------------------
                if plot=='detail':
                    r_essai = (beliefs[:, trial])
                    axs[4].plot(r_essai, c='k')
                    axs[4].spines['top'].set_color('none')
                    axs[4].spines['right'].set_color('none')

                    axs[4].set_xscale('log')
                    axs[4].set_xlim(0, max_run_length)

                    axs[4].set_xlabel('r$_{%s}$'%(trial), fontsize=t_label/1.5)
                    axs[4].set_ylabel('p(r) at trial $%s$'%(trial), fontsize=t_label/1.5)
                    axs[4].set_title('Belief on r for trial %s'%(trial), x=0.5, y=1., fontsize=t_titre/1.2)
                    axs[4].xaxis.set_tick_params(labelsize=t_label/1.9)
                    axs[4].yaxis.set_tick_params(labelsize=t_label/1.9)


                #---------------------------------------------------------------------------
                # cosmétique
                #---------------------------------------------------------------------------
                for i_layer, label in zip(range(2), ['$\hat{P}$ +/- CI', 'belief on r=p(r)']):
                    axs[i_layer+num].set_xlim(-1, N_trials)
                    axs[i_layer+num].axis('tight')
                    axs[i_layer+num].set_ylabel(label, fontsize=t_label/1.5)
                    axs[i_layer+num].xaxis.set_ticks_position('bottom')
                    axs[i_layer+num].yaxis.set_ticks_position('left')

                axs[num].set_xlim(-1, N_trials)
                axs[num].set_ylim(-.05, 1 + .05)
                axs[num].set_yticks(np.arange(0, 1 + .05, 1/2))
                axs[num].set_xticks([])
                axs[num].set_xticklabels([])

                axs[num+1].set_ylim(-.05*140, 140 + (.05*140))
                axs[num+1].set_yticks(np.arange(0, 140 + (.05*140), 140/2))
                axs[num+1].set_xlabel('trials', fontsize=t_label);
                axs[num+1].set_xticks([-1, 49, 99,149])
                axs[num+1].set_xticklabels([0, 50, 100, 150], ha='left', fontsize=t_label/2)

                if plot=='normal' :
                    axs[2].set_xlabel('Hazard rate', fontsize=t_label/2)
                    axs[2].set_ylabel('Mean log-likelihood (bits)', fontsize=t_label/2)
                    axs[2].legend(frameon=False, loc="lower left")

                if plot=='detail':
                    axs[num+1].bar(trial-1, 140 + (.05*140)+.05*140, bottom=-.05*140, color='firebrick', width=.5, linewidth=0, alpha=1)
                    axs[num+1].yaxis.set_tick_params(labelsize=t_label/2)
                    axs[num+1].set_xlabel('Trials', fontsize=t_label);

                    axs[num].yaxis.set_tick_params(labelsize=t_label/2)

                    if m == 'expectation' :
                        axs[num].set_title('Bayesian change point : expectation $\sum_{r=0}^\infty r \cdot p(r) \cdot \hat{p}(r) $', x=0.5, y=1.20, fontsize=t_titre)
                    else :
                        axs[num].set_title('Bayesian change point : $\hat{p} ( \mathrm{ArgMax}_r (p(r)) )$', x=0.5, y=1.05, fontsize=t_titre)

                for i_layer in range(len(axs)) :
                    axs[i_layer].xaxis.set_ticks_position('bottom')
                    axs[i_layer].yaxis.set_ticks_position('left')
                #---------------------------------------------------------------------------

                if plot=='normal':
                    fig.tight_layout()
                    plt.subplots_adjust(hspace=0.1)
                    plt.show()

        if plot=='detail':
            plt.show()

        return fig, axs


    def plot_results(self, mode_bcp='expectation', mode_kde=None, tau=40., sujet=[6], fig_width=15, t_titre=35, t_label=25, plot='Full') :

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import bayesianchangepoint as bcp
        from scipy import stats

        colors = ['black','dimgrey','grey','darkgrey','silver','rosybrown','lightcoral','indianred','firebrick','brown','darkred','red']
        nb_sujet = len(self.PARI)
        full = full_liste(self.PARI, self.ENREGISTREMENT, P_HAT=True)

        if plot == 'Full' :
            a = len(sujet)
            b = len(sujet)+1
            print('sujet+scatterKDE')
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

        elif plot == 'sujet' :
            print('sujet')
            fig, axs = plt.subplots(len(sujet), 1, figsize=(fig_width, fig_width/(1.6180)))
            # fig.subplots_adjust(left = 0, bottom = (len(sujet))*2/3, right = 1, top =len(sujet))
            # plt.subplots_adjust(hspace=0.05)

        elif plot == 'scatterKDE' :
            a = 0
            b = 1
            print('scatterKDE')
            fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_width/2.))
            # fig.subplots_adjust(left = 0, bottom = 1/2, right = 1, top =1)
            # fig.subplots_adjust(wspace=0.2)
            # plt.suptitle('Results bayesian change point %s'%(mode_bcp), fontsize=t_titre, x=0.5, y=1.3)

        if plot in ['Full', 'sujet'] :
            # axs[0].set_title('Results bayesian change point %s'%(mode_bcp), fontsize=t_titre, x=0.5, y=1.3)
            for s in range(len(sujet)) :
                axs[s] = results_sujet(self, axs[s], sujet, s, mode_bcp, tau, t_label)

        if plot in ['Full', 'scatterKDE'] :

            #------------------------------------------------
            # SCATTER KDE Plot
            #------------------------------------------------
            opt = {'mode_bcp':mode_bcp, 'mode_kde':mode_kde, 't_titre':t_titre, 't_label':t_label}
            if mode_kde is None :
                axs[a] = scatter_KDE(self, ax=axs[a], plot='scatter', result='bet', **opt)
                axs[b] = scatter_KDE(self, ax=axs[b], plot='scatter', result='acceleration', **opt)
            else :
                axs[a] = scatter_KDE(self, ax=axs[a], plot='kde', result='bet', **opt)
                axs[b] = scatter_KDE(self, ax=axs[b], plot='kde', result='acceleration', **opt)
            #------------------------------------------------

        for i_layer in range(len(axs)) :
            axs[i_layer].xaxis.set_ticks_position('bottom')
            axs[i_layer].yaxis.set_ticks_position('left')
            axs[i_layer].xaxis.set_tick_params(labelsize=t_label/1.8)
            axs[i_layer].yaxis.set_tick_params(labelsize=t_label/1.8)
            if i_layer < len(sujet)-1 :
                axs[i_layer].set_xticks([])
            elif i_layer == len(sujet)-1 :
                axs[i_layer].set_xlabel('Trials', fontsize=t_label)
                axs[i_layer].set_xticks([-1, 49, 99, 149])
                axs[i_layer].set_xticklabels([0, 50, 100, 150], ha='left',fontsize=t_label/1.8)

        if not plot in ['scatterKDE'] :
            axs[0].legend(fontsize=t_label/1.2, bbox_to_anchor=(0., 1.05, 1, 0.), loc=4, ncol=3,
                  mode="expand", borderaxespad=0.)

        #------------------------------------------------

        return fig, axs

    def plot_scatter_KDE(self, mode_bcp='expectation', plot='kde', mode_kde='normal', result='bet', fig=None, axs=None, fig_width=15, t_titre=35, t_label=25) :

        if fig is None:
            fig_width= fig_width
            fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_width)) #/(1.6180)))
        axs = scatter_KDE(self, axs, mode_bcp, plot, mode_kde, result, t_titre, t_label)

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
