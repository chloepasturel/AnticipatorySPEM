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
        if Jeffreys:
            p_random = beta.rvs(a=.5, b=.5, size=N_blocks)
        else:
            p_random = np.random.rand(1, N_blocks)
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * p_random # probability
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks) # binomial

    return (trials, p)


class aSPEM(object):
    """ docstring for the aSPEM class. """

    def __init__(self, mode, observer, timeStr) :
        self.mode = mode
        self.observer = observer
        self.timeStr = str(timeStr)

        self.init()


    def init(self) :

        # TODO: use pickle to extract the parameters of an experiment that was already run

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
                #print (self.exp)

        else :
            # width and height of your screen
            # displayed on a 20” Viewsonic p227f monitor with resolution 1024 × 768 at 100 Hz
            #w, h = 1920, 1200
            screen_width_px = 1024
            screen_height_px = 768
            screen_width_px, screen_height_px = 2560, 1440 # iMac 27''
            framerate = 100.
            screen = 0

            screen_width_cm = 57. # (cm)
            viewingDistance = 57. # (cm) TODO : what is the equivalent viewing distance?
            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi
            px_per_deg = screen_height_px / screen_width_deg

            # ---------------------------------------------------
            # stimulus parameters
            # ---------------------------------------------------
            dot_size = (0.02*screen_height_px)            #
            V_X_deg = 40.                                   # deg/s
            V_X = px_per_deg * V_X_deg     # pixel/s
            saccade_px = .618*screen_height_px
            offset = .2*screen_height_px

            # ---------------------------------------------------
            # exploration parameters
            # ---------------------------------------------------
            N_blocks = 2
            seed = 2017
            N_trials = 200
            tau = N_trials/5.
            (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)
            stim_tau = .35 # in seconds

            gray_tau = .0 # in seconds
            T =  stim_tau + gray_tau
            N_frame_stim = int(stim_tau*framerate)

            self.exp = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials, p=p, stim_tau =stim_tau,
                            N_frame_stim=N_frame_stim, T=T,
                            datadir=datadir, cachedir=cachedir,
                            framerate=framerate,
                            screen=screen,
                            screen_width_px=screen_width_px, screen_height_px=screen_height_px,
                            px_per_deg=px_per_deg, offset=offset,
                            dot_size=dot_size, V_X =V_X, saccade_px=saccade_px)

            #self.params_protocol = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials, p=p, stim_tau =stim_tau,
            #                N_frame_stim=N_frame_stim, T=T)

            #self.params_exp = dict(datadir=datadir, cachedir=cachedir,
            #            framerate=framerate,
            #            screen=screen,
            #            screen_width_px=screen_width_px, screen_height_px=screen_height_px,
            #            px_per_deg=px_per_deg)
            #self.params_stim = dict(dot_size=dot_size, V_X =V_X, saccade_px=saccade_px)



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
                _ = axs[1].plot(range(N_trials), block + results[:, block], alpha=.9, color='r')
            #print('corrects', corrects)
        fig.tight_layout()
        for i in range(2): axs[i].set_ylim(-.05, N_blocks + .05)
        axs[-1].set_xlabel('trials', fontsize=14);

        return fig, axs


    def run_experiment(self, verb=True):

        #if verb: print('launching experiment')

        from psychopy import visual, core, event, logging, prefs
        prefs.general['audioLib'] = [u'pygame']
        from psychopy import sound

#        logging.console.setLevel(logging.WARNING)
#        if verb: print('launching experiment')
#        logging.console.setLevel(logging.WARNING)
#        if verb: print('go!')

        # ---------------------------------------------------
        win = visual.Window([self.exp['screen_width_px'], self.exp['screen_height_px']],
                            allowGUI=False, fullscr=True, screen=self.exp['screen'], units='pix')

        win.setRecordFrameIntervals(True)
        win._refreshThreshold = 1/self.exp['framerate'] + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance

        # ---------------------------------------------------
        if verb: print('FPS = ',  win.getActualFrameRate() , 'framerate=', self.exp['framerate'])

        # ---------------------------------------------------
        #target = visual.Circle(win, lineColor='white', size=self.exp['dot_size'], lineWidth=2)
        target = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.exp['dot_size'])

        #fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.exp['dot_size'])
        fixation = visual.TextStim(win, text = u"+", units='pix', height=self.exp['dot_size']*4, color='white',
                                pos=[0., self.exp['offset']], alignHoriz='center', alignVert='center' )

        ratingScale = visual.RatingScale(win, scale=None, low=-1, high=1, precision=100, size=.4, stretch=2.5,
                        labels=('Left', 'unsure', 'Right'), tickMarks=[-1, 0., 1], tickHeight=-1.0,
                        marker='triangle', markerColor='black', lineColor='White', showValue=False, singleClick=True,
                        showAccept=False)

        #scorebox = visual.TextStim(win, text = u"0", units='norm', height=0.05, color='white', pos=[0., .5], alignHoriz='center', alignVert='center' )

        Bip_pos = sound.Sound('2000', secs=0.05)
        Bip_neg = sound.Sound('200', secs=0.5)

        # ---------------------------------------------------
        # fonction pause avec possibilité de quitter l'expérience
        msg_pause = visual.TextStim(win, text=u"\n\n\nTaper sur une touche pour continuer\n\nESCAPE pour arrêter l'expérience",
                                    font='calibri', height=25,
                                    alignHoriz='center')#, alignVert='top')

        def pause() :
            msg_pause.draw()
            win.flip()

            allKeys=event.waitKeys()
            for thisKey in allKeys:
                if thisKey in ["escape", "Q", "a"]:
                    win.close()
                    core.quit()

        def escape_possible() :
            if event.getKeys(keyList=["escape", "Q", "a"]):
                core.quit()
                #import sys
                #sys.exit()

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
                target.setPos((dir_sign * self.exp['V_X']*np.float(clock.getTime()/self.exp['stim_tau']), self.exp['offset']))
                target.draw()
                win.flip()

        # ---------------------------------------------------
        # EXPERIMENT
        # ---------------------------------------------------

        if self.mode == 'psychophysique' :

            results = np.zeros((self.exp['N_trials'], self.exp['N_blocks'] ))

            for block in range(self.exp['N_blocks']):

                score = 0
                pause()
                #print block
                for trial in range(self.exp['N_trials']):

                    ratingScale.reset()
                    while ratingScale.noResponse :

                        #scorebox.setText(str(score))
                        #scorebox.draw()

                        fixation.draw()
                        ratingScale.draw()
                        escape_possible()
                        win.flip()

                    ans = ratingScale.getRating()
                    results[trial, block] = ans

                    dir_bool = self.exp['p'][trial, block, 0]
                    presentStimulus_move(dir_bool)
                    win.flip()

                    score_trial = ans * (dir_bool * 2 - 1)
                    #print(score_trial)
                    if score_trial > 0 :
                        Bip_pos.setVolume(score_trial)
                        Bip_pos.play()
                    else :
                        Bip_neg.setVolume(-score_trial)
                        Bip_neg.play()
                    core.wait(0.1)

                    score += score_trial

            self.exp['results'] = results

            with open(self.exp_name(), 'wb') as fichier:
                f = pickle.Pickler(fichier)
                f.dump(self.exp)

        elif self.mode == 'enregistrement': # see for Eyelink
            for block in range(self.exp['N_blocks']):

                for trial in range(self.exp['N_trials']):

                    clock.reset()
                    t = clock.getTime()

                    fixation.draw()
                    escape_possible()
                    win.flip()
                    core.wait(np.random.uniform(0.4, 0.8))

                    # GAP
                    win.flip()
                    core.wait(0.3)

                    presentStimulus_move(self.exp['p'][trial, block, 0])
                    escape_possible()

                    win.flip()


        else :
            print ('mode incorect')


        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()

        core.quit()


if __name__ == '__main__':

    try:
        mode = sys.argv[1]
    except:
        mode = 'psychophysique'

    try:
        observer = sys.argv[2]
    except:
        observer = 'laurent'

    try:
        timeStr = sys.argv[4]
    except:
        import time
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        #timeStr = '2017-06-22_102207'

    e = aSPEM(mode, observer, timeStr)

    if True:
        print('Starting protocol')
        e.run_experiment()
