#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import sys
import os
import numpy as np
import pickle

def binomial_motion(N_trials, N_blocks, tau, seed, Jeffreys=True, N_layer=3):
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
            # Présente un dialogue pour changer les paramètres
            expInfo = {"Sujet":'', "Age":''}
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
            framerate = 100 #60 #100.for ordi enregistrement
            screen = 0 # 1 pour afficher sur l'écran 2 (ne marche pas pour enregistrement (mac))

            screen_width_cm = 57. # (cm)
            viewingDistance = 57. # (cm) TODO : what is the equivalent viewing distance?
            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi
            #px_per_deg = screen_height_px / screen_width_deg
            px_per_deg = screen_width_px / screen_width_deg

            # ---------------------------------------------------
            # stimulus parameters
            # ---------------------------------------------------
            dot_size = 10 # (0.02*screen_height_px)
            V_X_deg = 20   # deg/s   # 15 for 'enregistrement'
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
            stim_tau = 1 #.35 # in seconds # 1.5 for 'enregistrement'

            gray_tau = .0 # in seconds
            T =  stim_tau + gray_tau
            N_frame_stim = int(stim_tau*framerate)
            # ---------------------------------------------------

            self.exp = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials, p=p, stim_tau =stim_tau,
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
                escape_possible(self.mode)
                win.flip()


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

        for block in range(self.exp['N_blocks']):
            if self.mode == 'pari' :
                score = 0

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

    def plot(self, mode=None, fig=None, axs=None, fig_width=13):

        import matplotlib.pyplot as plt

        N_trials = self.exp['N_trials']
        N_blocks = self.exp['N_blocks']
        p = self.exp['p']

        if fig is None:
            fig_width= fig_width
            fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_width/1.6180))
        stick = np.zeros_like(p)
        stick[:, :, 0] = np.ones((N_trials, 1)) * np.arange(N_blocks)[np.newaxis, :]
        stick[:, :, 1] = np.ones((N_trials, 1)) * np.arange(N_blocks)[np.newaxis, :]
        stick[:, :, 2] = np.ones((N_trials, 1)) * np.arange(N_blocks)[np.newaxis, :]
        corrects = 0

        for i_layer, label in enumerate([r'$\^x_0$', r'$\^p$', r'$\^x_2$']):
            from cycler import cycler
            axs[i_layer].set_prop_cycle(cycler('color', [plt.cm.magma(h) for h in np.linspace(0, 1, N_blocks+1)]))
            _ = axs[i_layer].step(range(N_trials), p[:, :, i_layer]+stick[:, :, i_layer], lw=1, alpha=.9)
            for i_block in range(N_blocks):
                _ = axs[i_layer].fill_between(range(N_trials), i_block + np.zeros_like(p[:, i_block, i_layer]), i_block + p[:, i_block, i_layer], lw=.5, alpha=.1, facecolor='green', step='pre')
                _ = axs[i_layer].fill_between(range(N_trials), i_block + np.ones_like(p[:, i_block, i_layer]), i_block + p[:, i_block, i_layer], lw=.5, alpha=.1, facecolor='red', step='pre')
            axs[i_layer].axis('tight')
            axs[i_layer].set_yticks(np.arange(N_blocks)+.5)
            axs[i_layer].set_yticklabels(np.arange(N_blocks) )
            axs[i_layer].set_ylabel(label, fontsize=14)

        if not mode is None:
            results = (self.exp['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
            for block in range(N_blocks):
                #corrects += (results[:, block] == p[:, block, 0]).sum()
                _ = axs[1].step(range(N_trials), block + results[:, block], alpha=.9, color='r')
            #print('corrects', corrects)
        fig.tight_layout()
        for i in range(2): axs[i].set_ylim(-.05, N_blocks + .05)
        axs[-1].set_xlabel('trials', fontsize=14);

        return fig, axs, p

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
        ##################################
        RashBass = self.exp['RashBass']
        #RashBass = 100
        ##################################
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

                ##################################################
                # TARGET
                ##################################################
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
                ##################################################
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

                for f in range(len(fixations)) :
                    axs[trial]. axvspan(fixations[f][0]-start, fixations[f][1]-start, color='r', alpha=0.1)
                for s in range(len(saccades)) :
                    axs[trial]. axvspan(saccades[s][0]-start, saccades[s][1]-start, color='k', alpha=0.2)
            plt.tight_layout() # pour supprimer les marge trop grande
            plt.subplots_adjust(hspace=0) # pour enlever espace entre les figures

            plt.savefig('figures/%s_%s_block-%s_%s-trials.pdf'%(self.observer, self.timeStr, block+1, N_trials))
        plt.close()
        return fig, axs
        


if __name__ == '__main__':

    try:
        mode = sys.argv[1]
    except:
        mode = 'pari' #'enregistrement' #

    try:
        timeStr = sys.argv[4]
    except:
        import time
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        #timeStr = '2017-06-22_102207'

    e = aSPEM(mode, timeStr)

    if True:
        print('Starting protocol')
        e.run_experiment()
