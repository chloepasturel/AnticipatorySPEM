#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time

def binomial_motion(N_trials, N_blocks, tau, seed, Jeffreys=True, N_layer=3):
    from scipy.stats import beta
    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)

    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau # switch
        if Jeffreys: p_random = beta.rvs(a=.5, b=.5, size=N_blocks)
        else: p_random = np.random.rand(1, N_blocks)
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * p_random # probability
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks) # Bernouilli trial

    return (trials, p)


def run_experiment():

    #NameVideo = 'eyeMvt'# .mp4
    NameVideo = 'bet'# .mp4

    # ---------------------------------------------------
    # exploration parameters
    # ---------------------------------------------------
    seed = 51
    N_trials = 7#200
    N_blocks = 1#3
    tau = N_trials/5.
    (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)
    stim_tau = .75 # in seconds # 1.5 for 'eyeMvt'


    # ---------------------------------------------------
    # setup values
    # ---------------------------------------------------

    # width and height of your screen
    screen_width_px = 800/1.618 #1920 #1280 for ordi enregistrement
    screen_height_px = 500/1.618 #1080 #1024 for ordi enregistrement
    framerate = 60 #100.for ordi enregistrement

    screen_width_cm = 37 # (cm)
    viewingDistance = 57. # (cm)
    screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi
    px_per_deg = screen_width_px / screen_width_deg

    # ---------------------------------------------------
    # stimulus parameters
    # ---------------------------------------------------
    dot_size = 10 # (0.02*screen_height_px)
    V_X_deg = 15 # deg/s
    V_X = px_per_deg * V_X_deg     # pixel/s

    RashBass  = 100  # ms - pour reculer la cible à t=0 de sa vitesse * latency=RashBass

    saccade_px = .618*screen_height_px
    offset = 0 #.2*screen_height_px

    # ---------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------

    from psychopy import visual, core, event, logging, prefs
    prefs.general['audioLib'] = [u'pygame']
    from psychopy import sound

    # ---------------------------------------------------
    win = visual.Window([screen_width_px, screen_height_px], color=(0, 0, 0), allowGUI=False, fullscr=False, screen=0, units='pix')
    win.setRecordFrameIntervals(True)
    win._refreshThreshold = 1/framerate + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance

    # ---------------------------------------------------
    print('FPS = ',  win.getActualFrameRate() , 'framerate=', framerate)

    # ---------------------------------------------------
    target = visual.Circle(win, lineColor='white', size=dot_size, lineWidth=2)
    fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=dot_size)

    if NameVideo=='bet' :
        ratingScale = visual.RatingScale(win, scale=None, low=-1, high=1, precision=100, size=.7, stretch=2.5,
                        labels=('Left', 'unsure', 'Right'), tickMarks=[-1, 0., 1], tickHeight=-1.0,
                        marker='triangle', markerColor='black', lineColor='White', showValue=False, singleClick=True,
                        showAccept=False, pos=(0, -screen_height_px/3)) #size=.4

    # ---------------------------------------------------

    def escape_possible() :
        if event.getKeys(keyList=['escape', 'a', 'q']):
            win.close()
            core.quit()

    # ---------------------------------------------------

    clock = core.Clock()
    myMouse = event.Mouse(win=win)

    def presentStimulus_move(dir_bool):
        clock.reset()
        #myMouse.setVisible(0)
        dir_sign = dir_bool * 2 - 1
        while clock.getTime() < stim_tau:

            escape_possible()
            # la cible à t=0 recule de sa vitesse * latency=RashBass (ici mis en s)
            target.setPos(((dir_sign * V_X*clock.getTime())-(dir_sign * V_X*(RashBass/1000)), offset))
            target.draw()
            win.flip()
            win.getMovieFrame()
            win.flip()
            win.getMovieFrame()
            escape_possible()
            #win.flip()

    # ---------------------------------------------------
    # EXPERIMENT
    # ---------------------------------------------------

    for block in range(N_blocks):

        for trial in range(N_trials):

            # ---------------------------------------------------
            # FIXATION
            # ---------------------------------------------------
            event.clearEvents()
            if NameVideo=='bet' :
                ratingScale.reset()
                while ratingScale.noResponse :
                    fixation.draw()
                    ratingScale.draw()
                    escape_possible()
                    win.flip()
                    win.getMovieFrame()
                #ans = ratingScale.getRating()

            elif NameVideo=='eyeMvt' :
                duree_fixation = np.random.uniform(0.4, 0.8) # durée du point de fixation (400-800 ms)
                tps_fixation = 0
                tps_start_fix = time.time()
                # ---------------------------------------------------
                while (tps_fixation < duree_fixation) :
                    escape_possible()
                    tps_actuel = time.time()
                    tps_fixation = tps_actuel - tps_start_fix

                    escape_possible()
                    fixation.draw()
                    win.flip()
                    win.getMovieFrame()
                    escape_possible()

            # ---------------------------------------------------
            # GAP
            # ---------------------------------------------------
            win.flip()
            win.getMovieFrame()

            escape_possible()
            core.wait(0.3)

            # ---------------------------------------------------
            # Mouvement cible
            # ---------------------------------------------------
            escape_possible()
            dir_bool = p[trial, block, 0]
            presentStimulus_move(dir_bool)
            escape_possible()
            win.flip()
            win.getMovieFrame()

    win.update()
    core.wait(0.5)

    win.saveFrameIntervals(fileName=None, clear=True)
    from moviepy.editor import ImageSequenceClip
    for n, frame in enumerate(win.movieFrames):
        win.movieFrames[n] = np.array(frame)
    clip = ImageSequenceClip(win.movieFrames, fps=framerate)
    clip.write_videofile('%s.mp4'%NameVideo)

    win.close()
    core.quit()


if __name__ == '__main__':

    print('Starting protocol')
    run_experiment()
