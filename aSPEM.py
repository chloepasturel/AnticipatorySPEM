#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import sys
import os
import numpy as np

# displayed on a 20” Viewsonic p227f monitor with resolution 1024 × 768 at 100 Hz


def binomial_motion(N_trials, N_blocks, tau=25., seed=420, N_layer=3):

    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)
    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau # switch
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * np.random.rand(1, N_blocks) # probability
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks) # binomial

    return (trials, p)


class aSPEM(object):
    """ docstring for the aSPEM class. """

    def __init__(self):
        # super(, self).__init__()
        self.init()


    def init(self):

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
        for dir_ in [datadir, cachedir]:
            try:
                os.mkdir(dir_)
            except:
                pass

        # width and height of your screen
        # displayed on a 20” Viewsonic p227f monitor with resolution 1024 × 768 at 100 Hz
        #w, h = 1920, 1200
        #w, h = 2560, 1440 # iMac 27''
        screen_width_px = 1024
        screen_height_px = 768
        framerate = 100.
        screen = 0

        screen_width_cm = 57. # (cm)
        viewingDistance = 57. # (cm) TODO : what is the equivalent viewing distance?
        screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi
        px_per_deg = screen_height_px / screen_width_deg

        self.params_exp = dict(datadir=datadir, cachedir=cachedir,
                    framerate=framerate,
                    screen=screen,
                    screen_width_px=screen_width_px, screen_height_px=screen_height_px,
                    px_per_deg=px_per_deg)

        # ---------------------------------------------------
        # exploration parameters
        # ---------------------------------------------------
        N_blocks = 2
        seed = 1973
        N_trials = 80
        tau = N_trials/4.
        (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)
        stim_tau = .35 # in seconds

        gray_tau = .0 # in seconds
        T =  stim_tau + gray_tau
        N_frame_stim = int(stim_tau*framerate)

        self.params_protocol = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials, p=p, stim_tau =stim_tau,
                        N_frame_stim=N_frame_stim, T=T)

        # ---------------------------------------------------
        # stimulus parameters
        # ---------------------------------------------------
        dot_size = (0.05*screen_height_px)            # 
        V_X_deg = 20.                                   # deg/s
        V_X = px_per_deg * V_X_deg     # pixel/s
        saccade_px = .618/2*screen_height_px
        self.params_stim = dict(dot_size=dot_size, V_X =V_X, saccade_px=saccade_px)


    def print_protocol(self):
        if True: #try:
            N_blocks = self.params_protocol['N_blocks']
            N_trials = self.params_protocol['N_trials']
            N_frame_stim = self.params_protocol['N_frame_stim']
            T = self.params_protocol['T']
            return "TODO"
    #         return """
    # ##########################
    # #  PROTOCOL  #
    # ##########################
    #
    # We used a two alternative forced choice (2AFC) paradigm. In each trial, a gray fixation screen with a small dark fixation spot was followed by a moving target during {stim_tau} second each. Different trials are separated by an uniformly gray {gray_tau}  inter-stimulus interval. Before each trial, a gray screen appears asking the participant to report in which direction he thinks the target will go.
    #
    #  * Presentation of stimuli at {framerate} Hz on the {screen_width_px}x{screen_width_px} array during {T} s
    #
    #  * fixed parameters:
    #          - dot_size = {dot_size} dot's size,
    #          - XXX
    #
    # Fro each condition (blockid), we used {N_blocks} blocks of {N_trials} trials.
    #
    #  * parameters:
    #          - N_blocks = {N_blocks} different blocks: seed={seed} and their {N_blocks} increments to generate different movies within one block
    #          - N_trials = {N_trials} number of trials within a block
    #
    #         Grand total for one block is
    #          - {N_trials} trials ⨉
    #          - {N_blocks} blocks x
    #          - {T}s
    #
    #         That is,
    #          - One block=  {N_trials}  trials
    #          - One block=  {total_frames}  frames
    #          - One block= {time} seconds
    #          - {N_blocks} repetitions of each block= {total_time} seconds
    #
    #
    #  # and now... let's
    #     """.format(**self.params_protocol, **self.params_stim, **self.params_exp,
    #                time=N_trials * T,
    #                N_conditions=N_blocks * N_trials,
    #                total_frames=N_blocks * N_trials * N_frame_stim,
    #                total_time=N_blocks * N_trials * T)

        # except:
        #     return 'blurg'


    def exp_name(self, mode, observer, block, timeStr):
        return os.path.join(self.params_exp['datadir'], mode + '_' + observer + '_' + str(block) + '_' + timeStr + '.npy')


    def load(self, mode, observer, block, timeStr):
        return np.load(self.exp_name(mode, observer, block, timeStr))


    def plot(self, mode, observer, N_trials, N_blocks, p, timeStr, fig_width):
        import matplotlib.pyplot as plt

        fig_width= fig_width
        fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_width/1.6180))
        stick = np.zeros_like(p)
        stick[:, :, 0] = np.ones((N_trials, 1)) * np.arange(N_blocks)[np.newaxis, :]
        stick[:, :, 1] = np.ones((N_trials, 1)) * np.arange(N_blocks)[np.newaxis, :]
        stick[:, :, 2] = np.ones((N_trials, 1)) * np.arange(N_blocks)[np.newaxis, :]
        corrects = 0

        for i_layer, label in enumerate([r'$\^x_1$', r'$\^p$', r'$\^x_3$']):
            from cycler import cycler
            axs[i_layer].set_prop_cycle(cycler('color', [plt.cm.magma(h) for h in np.linspace(0, 1, N_blocks+1)]))
            _ = axs[i_layer].step(range(N_trials), p[:, :, i_layer]+stick[:, :, i_layer], lw=.5, alpha=.9)
            axs[i_layer].axis('tight')
            axs[i_layer].set_ylabel(label, fontsize=14)

        for block in range(N_blocks):
            results = self.load(mode, observer, block, timeStr)
            corrects += (results == p[:, block, 0]).sum()
            _ = axs[1].plot(range(N_trials), block + results, alpha=.9, color='r')

        fig.tight_layout()
        for i in range(2): axs[i].set_ylim(-.05, N_blocks + .05)
        axs[-1].set_xlabel('time', fontsize=14);

        return corrects


    def run_experiment(self, mode, observer, block, timeStr, verb=True):

        if verb: print('launching experiment')

        from psychopy import visual, core, event, logging

        logging.console.setLevel(logging.DEBUG)
        if verb: print('launching experiment')
        logging.console.setLevel(logging.DEBUG)
        if verb: print('go!')

        # ---------------------------------------------------
        win = visual.Window([self.params_exp['screen_width_px'], self.params_exp['screen_height_px']],
                            allowGUI=False, fullscr=True, screen=self.params_exp['screen'], units='pix')

        win.setRecordFrameIntervals(True)
        win._refreshThreshold = 1/self.params_exp['framerate'] + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance

        # ---------------------------------------------------
        if verb: print('FPS = ',  win.getActualFrameRate() , 'framerate=', self.params_exp['framerate'])

        # ---------------------------------------------------
        #target = visual.Circle(win, lineColor='white', size=self.params_stim['dot_size'], lineWidth=2)
        target = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.params_stim['dot_size'])

        #fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.params_stim['dot_size'])
        fixation = visual.TextStim(win,
                                text = u"+", units='norm', height=0.15, color='BlanchedAlmond',
                                pos=[0., -0.], alignHoriz='center', alignVert='center' )

        ratingScale = visual.RatingScale(win, scale=None, low=-1, high=1, precision=100, size=.4, stretch=2.5,
                        labels=('bet Left', 'unsure...', 'bet Right'), tickMarks=[-1, 0., 1], tickHeight=-1.0,
                        marker='triangle', markerColor='black', lineColor='White', showValue=False, singleClick=True,
                        showAccept=False, flipVert=True)

        scorebox = visual.TextStim(win,
                                text = u"0", units='norm', height=0.15, color='BlanchedAlmond',
                                pos=[0., .5], alignHoriz='center', alignVert='center' )


        # ---------------------------------------------------
        def escape_possible() :
            if event.getKeys(keyList=["escape", "Q", "a"]):
                core.quit()
                import sys
                sys.exit()


        def presentStimulus_fixed(dir_bool):
            dir_sign = dir_bool * 2 - 1
            target.setPos((dir_sign * (self.params_stim['saccade_px']), 0))
            target.draw()
            win.flip()
            core.wait(0.3)

        clock = core.Clock()
        myMouse = event.Mouse(win=win)

        def presentStimulus_move(dir_bool):
            clock.reset()
            myMouse.setVisible(0)
            dir_sign = dir_bool * 2 - 1
            while clock.getTime() < self.params_protocol['stim_tau']:
                target.setPos((dir_sign * self.params_stim['V_X']*np.float(clock.getTime()/self.params_protocol['stim_tau']), 0))
                target.draw()
                win.flip()

        # ---------------------------------------------------
        # EXPERIMENT
        # ---------------------------------------------------

        results = np.zeros((self.params_protocol['N_trials'], ))

        if mode == 'psychophysique' :
            score = 0

            for trial in range(self.params_protocol['N_trials']):
                scorebox.setText(str(score))
                scorebox.draw()

                ratingScale.reset()
                while ratingScale.noResponse :
                    fixation.draw()
                    ratingScale.draw()
                    escape_possible()
                    win.flip()

                ans = ratingScale.getRating()
                results[trial] = ans

                dir_bool = self.params_protocol['p'][trial, block, 0]
                presentStimulus_fixed(dir_bool)
                win.flip()
                
                score += ans * (dir_bool * 2 - 1)


        elif mode == 'enregistrement': # see for Eyelink

            for trial in range(self.params_protocol['N_trials']):

                clock.reset()
                t = clock.getTime()

                fixation.draw()
                escape_possible()
                win.flip()
                core.wait(np.random.uniform(0.4, 0.8))

                # GAP
                win.flip()
                core.wait(0.3)

                presentStimulus_move(self.params_protocol['p'][trial, block, 0])
                escape_possible()

                win.flip()


        else :
            print ('mode incorect')


        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()

        #save data
        np.save(self.exp_name(mode, observer, block, timeStr), results)
        core.quit()


if __name__ == '__main__':
    e = aSPEM()
    print('Starting protocol')

    try:
        mode = sys.argv[1]
    except:
        mode = 'psychophysique'

    try:
        observer = sys.argv[2]
    except:
        observer = 'anna'

    try:
        block = int(sys.argv[3])
    except:
        block = 0

    try:
        timeStr = sys.argv[4]
    except:
        import time, datetime
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    if True:
        e.run_experiment(mode, observer, block,  timeStr)