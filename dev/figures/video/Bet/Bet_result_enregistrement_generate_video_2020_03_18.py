#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time



def video():

    NameVideo = '2_results_enregistrement'# .mp4
    import os

    folder = 'results_enregistrement/'
    fileList = os.listdir(folder)


    # ---------------------------------------------------
    # setup values
    # ---------------------------------------------------
    screen_width_px = 800/1.618
    screen_height_px = 500/1.618
    framerate = 60

    # ---------------------------------------------------
    # stimulus parameters
    # ---------------------------------------------------
    wait0 = 0.05
    wait1 = 0.05
    wait2 = 0.05
    num_fig1_bascule = 5
    num_fig2_bascule = 22

    dot_size = 20
    offset = 0
    # ---------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------

    from psychopy import visual, core, event, logging, prefs

    # ---------------------------------------------------
    win = visual.Window([screen_width_px, screen_height_px], color=(-255, -255, -255),
                         allowGUI=False, fullscr=False, screen=0, units='pix')
    print('FPS = ',  win.getActualFrameRate() , 'framerate=', framerate)
    # ---------------------------------------------------

    clock = core.Clock()

    for num_fig in range(1, len(fileList)+1) :
        fig = visual.ImageStim(win, folder + '2_results_enregistrement_%s.png'%num_fig)
        fig.setPos((offset, offset))

        if num_fig<num_fig1_bascule :   wait = wait0
        elif num_fig<num_fig2_bascule : wait = wait1
        else :                          wait = wait2

        tps, start = 0, time.time()
        while (tps < wait) :
            tps = time.time() - start
            fig.draw()
            win.flip()
            win.getMovieFrame()

    tps, start = 0, time.time()
    while (tps < .5) :
        tps = time.time() - start
        fig.draw()
        win.flip()
        win.getMovieFrame()

    win.update()
    #win.getMovieFrame()

    win.saveFrameIntervals(fileName=None, clear=True)
    from moviepy.editor import ImageSequenceClip
    for n, frame in enumerate(win.movieFrames):
        win.movieFrames[n] = np.array(frame)
    clip = ImageSequenceClip(win.movieFrames, fps=framerate)
    clip.write_videofile('%s.mp4'%(NameVideo))

    win.close()
    core.quit()


if __name__ == '__main__':
    video()
