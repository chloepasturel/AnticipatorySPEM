#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time



def video():

    NameVideo = 'BSM'# .mp4
    import os

    folder = 'BSM/proba_fig/'
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
    wait0 = 0.6
    wait1 = 0.4
    wait2 = 0.2
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

    for num_fig in range(len(fileList)) :
        fig = visual.ImageStim(win, folder + 'proba_bsm_%s.png'%num_fig)
        fig.setPos((-screen_width_px/6, offset))

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
    clip.write_videofile('%s/%s.mp4'%(NameVideo, NameVideo))

    win.close()
    core.quit()


if __name__ == '__main__':
    video()
