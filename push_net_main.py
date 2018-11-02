from push_net_core import *

if __name__=='__main__':
    [in_img, gl_img] = get_args()
    con = PushController(in_img, gl_img)
