#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from psychopy import visual

win=visual.Window([800, 800],units='pix')

#ratingScale = visual.RatingScale(win, low=0, high=100, precision=0, acceptKeys='4', scale='ploup, prout')
cercle_1 = visual.Circle(win, size=6)

ratingScale = visual.RatingScale(win, scale=None, choices=None, low=1,
    high=10, precision=1, labels=(), tickMarks=None, tickHeight=1.0,
    marker='triangle', markerStart=None, markerColor=None,
    markerExpansion=1, singleClick=False,
    disappear=False,
    showValue=True, showAccept=True,
    acceptKeys='return', acceptPreText='key,click', acceptText='accept?', acceptSize=1.0, leftKeys='left',
    rightKeys='right', respKeys=(), lineColor='White',
    mouseOnly=False, noMouse=False, size=1.0,
    stretch=1.0, pos=None, minTime=0.4, maxTime=0.0,
    flipVert=False, depth=0, name=None, autoLog=True)

while ratingScale.noResponse:
    ratingScale.draw()
    win.flip()

rating = ratingScale.getRating()
decisionTime = ratingScale.getRT()
choiceHistory = ratingScale.getHistory()

print(rating)
print(decisionTime)
print(choiceHistory)