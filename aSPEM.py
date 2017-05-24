#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Using psychopy to perform an experiment on the role of a bias in the direction


"""

import sys
import os
import pickle
import numpy as np
# displayed on a 20” Viewsonic p227f monitor with resolution 1024 × 768 at 100 Hz


def binomial_motion(N_trials, N_blocks, tau=25., seed=420, N_layer=3):
    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)
    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] *  np.random.rand(1, N_blocks)
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks)

    return (trials, p)

class aSPEM(object):
    """docstring for the aSPEM class.
    """
    def __init__(self):
        # super(, self).__init__()
        self.init()

    def init(self):
        self.dry_run = True
        self.dry_run = False
        self.experiment = 'aSPEM'
        self.instructions = """

        TODO

        """

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
        # setup values
        framerate = 100.
        screen = 0
        screen_width_cm = 57. # (cm)
        viewingDistance = 57. # (cm) TODO : what is the equivalent viewing distance?
        screen_width_deg = 2.*np.arctan(screen_width_cm/2/viewingDistance)*180/np.pi
        screen_height_deg =  screen_width_deg*screen_height_px/screen_width_px
        deg_per_px = screen_height_deg/screen_height_px
        #info['stim_diameter_height'] = stim_diameter_deg / screen_height_deg
        #stim_diameter_deg =  40 # deg
        stim_diameter_deg =  512./screen_height_px*screen_height_deg

        self.params_exp = dict(datadir=datadir, cachedir=cachedir, screen_width_px=screen_width_px, screen_height_px=screen_height_px,
        framerate=framerate, screen=screen, screen_width_cm=screen_width_cm,
        viewingDistance=viewingDistance, screen_width_deg=screen_width_deg,
        screen_height_deg=screen_height_deg, deg_per_px=deg_per_px, stim_diameter_deg=stim_diameter_deg)


        # exploration parameters
        N_blocks = 2 #

        seed = 1973
        N_trials = 80
        tau = N_trials/4.

        (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)

        total_conditions = N_trials *  N_blocks
        stim_tau = .35 # in seconds
        gray_tau = .0 # in seconds
        T =  stim_tau + gray_tau
        #N_frame_total'] = 128 # a full period. in time frames
        #info['N_frame'] = int(info['framerate']/stim_tau) # length of the presented period. in time frames
        N_frame_stim = int(stim_tau*framerate)
        N_frame =  int(T*framerate)

        self.params_protocol = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials,
                               stim_tau=stim_tau, gray_tau=gray_tau, T=T, N_frame=N_frame, N_frame_stim=N_frame_stim,
                               total_conditions=total_conditions, p=p)


        # stimulus parameters

        N_X, N_Y = int(screen_width_px), int(screen_width_px)
        dot_size = 0.01
        V_X = .5
        V_Y = 0.

        self.params_stim = dict(N_X=N_X, N_Y=N_Y, V_X=V_X, V_Y=V_Y, dot_size=dot_size)

    def print_protocol(self):
        if True: #try:
            N_blocks = self.params_protocol['N_blocks']
            N_trials = self.params_protocol['N_trials']
            N_frame_stim = self.params_protocol['N_frame_stim']
            T = self.params_protocol['T']
            return "TODO"
    #         return """
    # ##########################
    # #  🐭 🐁 PROTOCOL 🐭 🐁 #
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
    #  # and now... let's 💃
    #     """.format(**self.params_protocol, **self.params_stim, **self.params_exp,
    #                time=N_trials * T,
    #                N_conditions=N_blocks * N_trials,
    #                total_frames=N_blocks * N_trials * N_frame_stim,
    #                total_time=N_blocks * N_trials * T)

        # except:
        #     return 'blurg'

    def exp_name(self, observer, block, timeStr):
        return os.path.join(self.params_exp['datadir'], observer + '_' + str(block) + '_' + timeStr + '.npy')

    def load(self, observer, block, timeStr):
        return np.load(self.exp_name(observer, block, timeStr))

    def run_experiment(self, observer, block, timeStr, verb=True):

        if verb: print('launching experiment')
        from psychopy import visual, core, event, logging, misc

        logging.console.setLevel(logging.DEBUG)
        if verb: print('launching experiment')
        logging.console.setLevel(logging.DEBUG)
        if verb: print('go!')
        # http://www.psychopy.org/general/monitors.html

        win = visual.Window([self.params_exp['screen_width_px'], self.params_exp['screen_height_px']],
                            allowGUI=False, fullscr=True, screen=self.params_exp['screen'])
        win.setRecordFrameIntervals(True)
        win._refreshThreshold=1/self.params_exp['framerate']+0.004 #i've got 50Hz monitor and want to allow 4ms tolerance
        #set the log module to report warnings to the std output window (default is errors only)
        # log.console.setLevel(log.WARNING)
        myMouse = event.Mouse(win=win)


        if verb: print('FPS = ',  win.getActualFrameRate() , 'framerate=', self.params_exp['framerate'])
        target = visual.Circle(win, lineColor='black',  units='height',
                                radius = self.params_stim['dot_size'],
                                interpolate = True, fillColor='Navy',
                                autoLog=False)

        wait_for_response = visual.TextStim(win,
                                text = u"+", units='norm', height=0.15, color='BlanchedAlmond',
                                pos=[0., -0.], alignHoriz='center', alignVert='center' )
        # wait_for_next = visual.TextStim(win,
        #                         text = u"+", units='norm', height=0.15, color='BlanchedAlmond',
        #                         pos=[0., -0.], alignHoriz='center', alignVert='center' )

        def getResponse():
            event.clearEvents() # clear the event buffer to start with
            resp = None # initially
            while True: # forever until we return a keypress
                # wheel_dX, wheel_dY = myMouse.getWheelRel()
                #    #get mouse events
                #    mouse_dX,mouse_dY = myMouse.getRel()
                #    mouse1, mouse2, mouse3 = myMouse.getPressed()
                #    if (mouse1):
                #        wedge.setAngularCycles(mouse_dX, '+')
                #    elif (mouse3):
                #        rotationRate += mouse_dX
                for key in event.getKeys():
                    # quit
                    if key in ['escape', 'q']:
                        win.close()
                        core.quit()
                        return None
                    # valid response - check to see if correct
                    elif key in ['left', 'right']:
                        if key in ['left'] :return -1
                        else: return 1
                    else:
                        print ("hit LEFT or RIGHT (or Esc) (You hit %s)" %key)

        clock = core.Clock()

        def presentStimulus(dir_bool):
            """Present stimulus
            """
            clock.reset()
            myMouse.setVisible(0)
            dir_sign = dir_bool * 2 - 1
            while clock.getTime() < self.params_protocol['stim_tau']:
                target.setPos((dir_sign * self.params_stim['V_X']*np.float(clock.getTime()/self.params_protocol['stim_tau']), 0))
                target.draw()
                win.flip()

        results = np.zeros((self.params_protocol['N_trials'], ))
        for trial in range(self.params_protocol['N_trials']):
            clock.reset()
            t = clock.getTime()
            # wait_for_next.draw()
            # core.wait(self.params_protocol['gray_tau'])
            # win.flip()

            wait_for_response.draw()
            win.flip()
            ans = getResponse()
            results[trial] = ans

            presentStimulus(self.params_protocol['p'][trial, block, 0])
            win.flip()


        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()

        #save data

        np.save(self.exp_name(observer, block, timeStr), results)
        core.quit() # quit


if __name__ == '__main__':
    e = aSPEM()
    print('Starting protocol')

    try:
        observer = sys.argv[1]
    except:
        observer = 'anonymous'
    try:
        block = int(sys.argv[2])
    except:
        block = 0
    try:
        # width and height of the stimulus
        timeStr = sys.argv[3]
    except:
        import time, datetime
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    if True:#try:
        e.run_experiment(observer, block,  timeStr)
    # except:
    #     print('not able to launch psychopy, try with python2')
