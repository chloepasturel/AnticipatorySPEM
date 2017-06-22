from psychopy import prefs,core
prefs.general['audioLib'] = [u'pygame']
from psychopy import sound

import numpy as np

Bip_pos = sound.Sound('2000', secs=0.1)
for volume in np.linspace(0, 1., 10):
    Bip_pos.setVolume(volume)
    Bip_pos.play()
    core.wait(.5)

Bip_neg = sound.Sound('3000', secs=0.3)
for volume in np.linspace(0, 1., 10):
    Bip_neg.setVolume(volume)
    Bip_neg.play()
    core.wait(.5)

Bip_pos.play()
core.wait(.5)


