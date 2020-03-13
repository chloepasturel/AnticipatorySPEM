def figure():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np



    import os

    folder = 'proba_fig/'
    fileList = os.listdir(folder)
    print(len(fileList))

    for f in fileList:
        filePath = folder + '/' + f
        os.remove(filePath)

    np.random.seed(13)
    nb_trial = 30
    p1, p2, p3 = .5, .2, .8
    p = np.random.rand(nb_trial, 2)

    n_t_bascule = 12

    for trial in range(nb_trial):
        if trial < 10 :   p[trial, 1] = p1
        elif trial < 20 : p[trial, 1] = p2
        else :            p[trial, 1] = p3
        p[trial, 0] =  p[trial, 1] > np.random.rand()

    c='w'

    screen_width_px = (800/1.618)/1.5
    screen_height_px = (500/1.618)
    dpi = 80

    s1 = 1/(nb_trial-(nb_trial/2))
    s2 = 1/nb_trial

    num_f = 0
    for t in range(1, nb_trial) :

        fig, ax = plt.subplots(1,1, figsize=(screen_width_px/dpi, screen_height_px/dpi), dpi=dpi)
        if t<n_t_bascule :
            fig1, ax1 = plt.subplots(1,1, figsize=(screen_width_px/dpi, screen_height_px/dpi), dpi=dpi)

        for p_ in range(0, t) :

            if p[p_+1, 0]==0 : smile = 'smile-red'
            else :             smile = 'smile-green'

            if t<n_t_bascule :
                if p_==(t-1) :
                    newax = ax1.inset_axes([1-((t-(p_-.5))/nb_trial), .7, s1, s1])
                    newax.imshow(mpimg.imread('%s_50.png'%smile))
                    newax.axis('off')

                else :
                    for a in [ax, ax1] :
                        newax = a.inset_axes([1-((t-(p_-.5))/nb_trial), .7, s2, s2])
                        newax.imshow(mpimg.imread('%s_40.png'%smile))
                        newax.axis('off')
            else :
                newax = ax.inset_axes([1-((t-(p_-.5))/nb_trial), .7, s2, s2])
                newax.imshow(mpimg.imread('%s_40.png'%smile))
                newax.axis('off')
        #---------------------------------------------------

        l_ax = [ax]
        if t<n_t_bascule : l_ax.append(ax1)
        for a in l_ax :
            a.step(range(t+1), p[:(t+1), 1], lw=3, c=c)
            a.spines['bottom'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['left'].set_color(c)

            a.tick_params(axis='y', colors=c, labelsize=10)
            a.set_xticks([])
            a.spines['left'].set_bounds(0, 1)
            a.set_yticks([0,.5,1])
            a.axis([-.1-(nb_trial-(t+1)), t+.1, -0, 3])

        fig.savefig('proba_fig/proba_bsm_%s.png'%num_f, transparent=True, dpi=dpi)
        num_f+=1
        if t<n_t_bascule :
            fig1.savefig('proba_fig/proba_bsm_%s.png'%num_f, transparent=True, dpi=dpi)
            num_f+=1


if __name__ == '__main__':

    figure()
