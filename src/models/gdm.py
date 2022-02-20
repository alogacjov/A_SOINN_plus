##################################################################
# A reimplementation of the Growing Dual-Memory                  #
# approach of Parisi et al.:                                     #
#                                                                #
# Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018)          #
# Lifelong Learning of Spatiotemporal Representations            #
# with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966 #
#                                                                #
# Many parts of the original implementation are used here. See:  #
# https://github.com/giparisi/GDM                                #
#                                                                #
# The modifications are only made to make the GDM usable with    #
# our experimental setup                                         #
#                                                                # 
# @author: Aleksej Logacjov (a.logacjov@gmail.com)               #
#                                                                #
##################################################################


from GWR.gammagwr import GammaGWR


class EpisodicGWR(GammaGWR):
    def __init__(self):
        '''Reimplementation of Parisi et al.'s episodic_gwr.EpisodicGWR'''
        super().__init__()


class GDM(EpisodicGWR):
    def __init__(self):
        '''Reimplementation of Parisi et al.'s gdm_demo'''
        pass
