from psychopy import visual, core, event, logging, sound
Bip_pos = sound.Sound('2000', secs=0.2)
Bip_neg = sound.Sound('3000', secs=0.2)
Bip_pos.setVolume(.1)
Bip_pos.play()