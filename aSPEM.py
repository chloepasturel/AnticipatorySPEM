#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Using psychopy to perform an experiment on the role of a bias in the direction


"""

import sys
import os
import pickle
import numpy as np
import MotionClouds as mc
# displayed on a 20” Viewsonic p227f monitor with resolution 1024 × 768 at 100 Hz


def binomial_motion(N_trials, N_blocks, tau=25, seed=420, N_layer=3):
    np.random.seed(seed)

    trials = np.arange(N_tN_trialsime)
    p = np.random.rand(N_trials, N_blocks, N_layer)
    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] *  np.random.rand(1, N_blocks)
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks)

    return (trials, p)

class aSPEM(object):
    """docstring for the SpeedDiscrimination class.
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
        if DEBUG:
            N_blocks = 2 #
        else:
            N_blocks = 10

        seed = 1973
        N_trials = 400
        tau = 50

        (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)

        total_conditions = N_trials *  N_blocks
        stim_tau = .25 # in seconds
        gray_tau = .25 # in seconds
        T =  stim_tau + gray_tau
        #N_frame_total'] = 128 # a full period. in time frames
        #info['N_frame'] = int(info['framerate']/stim_tau) # length of the presented period. in time frames
        N_frame_stim = int(stim_tau*framerate)
        N_frame =  int(T*framerate)

        self.params_protocol = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials,
                               stim_tau=stim_tau, gray_tau=gray_tau, T=T, N_frame=N_frame, N_frame_stim=N_frame_stim,
                               total_conditions=total_conditions)


        # stimulus parameters

        N_X, N_Y = int(screen_width_px), int(screen_width_px)
        dot_size = 0.1
        V_X = 1.
        V_Y = 0.

        self.params_stim = dict(N_X=N_X, N_Y=N_Y, V_X=V_X, V_Y=V_Y, dot_size=dot_size)

    def print_protocol(self):
        if True: #try:
            N_seed = self.params_protocol['N_seed']
            N_trials = self.params_protocol['N_trials']
            N_frame_stim = self.params_MC['N_frame_stim']
            T = self.params_protocol['T']
            return """
    ##########################
    #  🐭 🐁 PROTOCOL 🐭 🐁 #
    ##########################

    We used a two alternative forced choice (2AFC) paradigm. In each trial, a gray fixation screen with a small dark fixation spot was followed by two stimulus intervals of {stim_tau} second each, separated by an uniformly gray {gray_tau} second inter-stimulus interval. The first stimulus had parameters (v1, z1) and the second had parameters (v2, z2). At the end of the trial, a gray screen appeared asking the participant to report which one of the two intervals was perceived as moving faster by pressing one of two buttons, that is whether v1 > v2 or v2 > v1.

     * Presentation of stimuli at {framerate} Hz on the {screen_width_px}x{screen_width_px} array during {T} s

     * fixed parameters:
             - dot_size = {dot_size} dot's size,
             - XXX

    Fro each condition (blockid), we used {N_seed} repetitions of each of the {N_combinations} possible combinations of these parameters are made per block of {N_conditions} trials. Finally, {N_trials} such blocks were collected per condition (blockid) tested.

     * parameters:
             - N_seed = {N_seed} different seeds: seed={seed} and their {N_seed} increments to generate different movies within one block
             - N_trials = {N_trials} different repetitions of the same block

            Grand total for one block is
             - {N_combinations} combinations ⨉
             - {N_seed} repetitions x
             - {T}s

            That is,
             - One block=  {N_conditions}  movies
             - One block=  {total_frames}  frames
             - One block= {time} seconds
             - {N_trials} repetitions of one block= {total_time} seconds

             Finally, the grand total time is equal to:

             - {N_blockids} blocks  ⨉ {total_time} seconds = {grand_total_time} seconds


     # and now... let's 💃
        """.format(**self.params_protocol, **self.params_MC,
                   time=N_combinations * N_trials * T,
                   N_conditions=N_combinations * N_seed,
                   total_frames=N_combinations* N_seed * N_frame_stim,
                   total_time= N_seed * N_trials * N_combinations * T,
                   grand_total_time=N_blockids * N_seed * N_trials * N_combinations * T)

        # except:
        #     return 'blurg'

    def exp_name(self, observer, timeStr):
        return os.path.join(self.info['datadir'], self.experiment + observer + '_' + timeStr + '.npy')

    def make_protocol(self, p_diconame='data/protocol.pkl'):
        """
        This function generates the diconame file to freeze the protocol given the global seed.

        """
        self.generate_MC() # to get self.combinations

        try:
            with open(p_diconame, 'rb') as output:
                protocol = pickle.load(output)
        except:
            protocol = {}

            for i_blockid, blockid in enumerate(self.params_protocol['blockids']):
                np.random.seed(self.params_protocol['seed'] + i_blockid)
                number_cases = self.params_protocol['N_combinations']*self.params_protocol['N_seed']
                order = np.random.permutation(range(number_cases))
                ref_firsts = np.random.rand(number_cases) > .5

                conditions = order // self.params_protocol['N_seed']
                seeds  = order % self.params_protocol['N_seed']

                delta_f = self.params_protocol['blockids'][blockid]['delta_f']
                delta_V = self.params_protocol['blockids'][blockid]['delta_V']

                protocol[blockid] = []
                for condition, ref_first, seed_ in zip(conditions, ref_firsts, seeds):
                    df = delta_f[condition // len(delta_f)]
                    dV = delta_V[condition % len(delta_f)]
                    protocol[blockid] += [dict(blockid=blockid, df=df, dV=dV, seed_=seed_, ref_first=ref_first)]

            with open(p_diconame, 'wb') as output: pickle.dump(protocol, output)
        self.protocol = protocol


    def run_experiment(self, observer, blockid,  timeStr, verb=True):
        self.make_protocol()

        if verb: print('launching experiment')
        from psychopy import visual, core, event, logging, misc

        logging.console.setLevel(logging.DEBUG)
        if verb: print('launching experiment')
        logging.console.setLevel(logging.DEBUG)
        if verb: print('go!')
        # http://www.psychopy.org/general/monitors.html

        win = visual.Window([self.info['screen_width_px'], self.info['screen_height_px']],
                            allowGUI=False, fullscr=True, mouseVisible=False, screen=screen)
        win.setRecordFrameIntervals(True)
        win._refreshThreshold=1/framerate+0.004 #i've got 50Hz monitor and want to allow 4ms tolerance
        #set the log module to report warnings to the std output window (default is errors only)
        log.console.setLevel(log.WARNING)
        myMouse = event.Mouse(win=win)


        if verb: print('FPS = ',  win.getActualFrameRate() , 'framerate=', framerate)
        target = visual.Circle(win, lineColor='black', size=0.02, pos=(x, 0))
        #  = visual.GratingStim(win,
        #         size=(self.info['stim_diameter_height'], self.info['stim_diameter_height']), units='height',
        #         interpolate=True,
        #         mask = 'gauss',
        #         autoLog=False)#this stim changes too much for autologging to be useful

        wait_for_response = visual.TextStim(win,
                                text = u"?", units='norm', height=0.15, color='DarkSlateBlue',
                                pos=[0., -0.], alignHoriz='center', alignVert='center' )
        wait_for_next = visual.TextStim(win,
                                text = u"+", units='norm', height=0.15, color='BlanchedAlmond',
                                pos=[0., -0.], alignHoriz='center', alignVert='center' )

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
        def presentStimulus(im):
            """
            Present stimulus

            """
            for i_frame in range(im.shape[-1]): # length of the stimulus
                target.setPos(im[:, :, i_frame])
                stim.draw()
                win.flip()

        nTrials = len(self.protocol[blockid])
        results = []
        for trial in self.protocol[blockid]:
            clock.reset()
            t = clock.getTime()
            fixSpot.draw()

            wait_for_next.draw()
            win.flip()
            # preparing data
            im_ref, im_test
            self.movie_name(trial['blockid'], trial['df'], trial['dV'], trial['seed_'])

            if trial['ref_first']:
                im_A, im_B = im_ref, im_test
            else:
                im_A, im_B = im_test, im_ref
            while clock.getTime() < self.params_protocol['gray_tau']:
                # waiting a bit
                pass
            presentStimulus(im_A)
            core.wait(self.params_protocol['gray_tau'])
            presentStimulus(im_A)
            wait_for_response.draw()
            win.flip()
            ans = getResponse()
            results[0, i_trial] = ans
            results[1, i_trial] = C_A

        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()

        #save data

        numpy.save(fileName, results)
        core.quit() # quit


if __name__ == '__main__':
    e = SpeedDiscrimination()
    print('Starting protocol')
    e.init()
    #e.generate_MC()
    e.make_protocol()

    try:
        observer = sys.argv[1]
    except:
        observer = 'anonymous'
    try:
        seed = int(sys.argv[2])
    except:
        seed = e.params_protocol['seed']
    try:
        i_trial = int(sys.argv[3])
    except:
        i_trial = 0
    try:
        # width and height of the stimulus
        timeStr = sys.argv[4]
    except:
        import time, datetime
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    try:
        e.run_experiment(observer, seed,  i_trial,  timeStr)
    except:
        print('not able to launch psychopy, try with python2')
