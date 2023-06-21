import sys
sys.path.append('\\Users\\sahil\\code\\Testing\\NN')
from lib import draw_dot
from lib import MLP


o = MLP(2, [4, 4, 1])
out = o([2, 3])
diagram = draw_dot(out)
diagram.render('MLP2[4,4,1]', format='png')
