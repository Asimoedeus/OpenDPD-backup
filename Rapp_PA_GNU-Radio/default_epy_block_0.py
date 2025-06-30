"""
Embedded Python Blocks:
 
Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block"
    """
    Simple Rapp model PA:
        y = x / (1 + (|x|/A_sat)^(2*p))^(1/(2*p))
    When p → ∞ 就变成硬限幅；p 越小转折越圆滑
    """
    def __init__(self, A_sat=1.0, p=2.0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='PA_model_Rapp',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.A_sat = A_sat
        self.p = p

    def work(self, input_items, output_items):
        x = input_items[0]
        y = x / (1.0 + (np.abs(x)/self.A_sat)**(2.0*self.p))**(1.0/(2.0*self.p))
        output_items[0][:] = y
        return len(output_items[0])
