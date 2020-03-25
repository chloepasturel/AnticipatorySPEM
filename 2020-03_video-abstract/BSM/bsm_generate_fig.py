import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 400

def figure():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    import os

    folder = 'proba_fig'
    fileList = os.listdir(folder)
    print(len(fileList))
    if True:
        for f in fileList:
            filePath = os.path.join(folder, f)
            os.remove(filePath)

    np.random.seed(13)
    np.random.seed(51)
    np.random.seed(51+13)
    nb_trial = 30
    p1, p2, p3 = .1, .9, .3

    p = np.random.rand(nb_trial, 2)

    n_t_bascule = 12

    for trial in range(nb_trial):
        if trial < 10 :   p[trial, 1] = p1
        elif trial < 20 : p[trial, 1] = p2
        else :            p[trial, 1] = p3
        p[trial, 0] =  p[trial, 1] > np.random.rand()

    color = 'white'

    screen_width_px = 800 # (800/1.618)#/1.5
    screen_height_px = 500 #(500/1.618)
    dpi = 80

    s1 = 1/(nb_trial-(nb_trial/2))  + 1/(nb_trial*2)
    s2 = 1/nb_trial + 1/(nb_trial*2)

    s2_ = 1/nb_trial


    num_f = 0
    for t in range(1, nb_trial) :

        fig, ax = plt.subplots(1, 1, figsize=(screen_width_px/dpi, screen_height_px/dpi),
                                facecolor='black', dpi=dpi)
        ax.set_facecolor('black')
        if t<n_t_bascule :
            fig1, ax1 = plt.subplots(1, 1, figsize=(screen_width_px/dpi, screen_height_px/dpi),
                                     facecolor='black', dpi=dpi)
            ax.set_facecolor('black')

        for p_ in range(0, t) :

            if p[p_+1, 0]==1 : smile = 'smile-red'
            else :             smile = 'smile-green'

            h_smiley = 1.04
            if t<n_t_bascule :
                if p_==(t-1) :
                    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.inset_axes.html?highlight=inset_axes#matplotlib.axes.Axes.inset_axes
                    # bounds[x0, y0, width, height]
                    newax = ax1.inset_axes([1-((t-(p_-.5))/nb_trial), h_smiley, s1, s1])
                    newax.imshow(mpimg.imread('%s_50.png'%smile))
                    newax.axis('off')

                else :
                    for a in [ax, ax1] :
                        newax = a.inset_axes([1-((t-(p_-.5))/nb_trial), h_smiley, s2, s2])
                        newax.imshow(mpimg.imread('%s_40.png'%smile))
                        newax.axis('off')
            else :
                newax = ax.inset_axes([1-((t-(p_-.5))/nb_trial), h_smiley, s2_, s2_])
                newax.imshow(mpimg.imread('%s_40.png'%smile))
                newax.axis('off')
        #---------------------------------------------------

        l_ax = [ax]
        if t<n_t_bascule : l_ax.append(ax1)
        for a in l_ax :
            a.step(range(t+1), p[:(t+1), 1], lw=3, c=color)
            if t > 10 :
                #a.fill_between(range(9, t+1), p[9:(t+1), 1], 0, color='r')
                a.step(range(9, t+1), p[9:(t+1), 1], lw=3, c='r')
            if t > 20 :
                #a.fill_between(range(19, t+1), p[19:(t+1), 1], 0, color='g')
                a.step(range(19, t+1), p[19:(t+1), 1], lw=3, c='g')

            a.spines['bottom'].set_color(color)#.set_visible(False)
            a.spines['top'].set_color(color)#.set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['left'].set_color(color)

            a.tick_params(axis='y', colors=color, labelsize=10)
            a.set_xticks([])
            a.set_ylabel('Probability', color='white', fontsize=24)
            # a.text(-.5, .5, 'p', color=color, fontsize=24)
            a.spines['left'].set_bounds(0, 1)
            a.set_yticks([0,.5,1])
            a.axis([-.5-(nb_trial-(t+1)), t+.1+1, -0, 1])

        opts = dict(transparent=True, dpi=dpi)#, bbox_inches='tight', pad_inches=.3)
        fig.savefig('proba_fig/proba_bsm_%s.png'%num_f, **opts)
        num_f += 1
        if t<n_t_bascule :
            fig1.savefig('proba_fig/proba_bsm_%s.png'%num_f, **opts)
            num_f += 1


if __name__ == '__main__':

    figure()
