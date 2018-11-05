from push_net_core import *

def get_args(argv=None):
    parser = argparse.ArgumentParser('push_net')
    parser.add_argument('--test-filename', dest='test_filename', type=str,
                        help='Input image filename.')
    parser.add_argument('--goal-filename', dest='goal_filename', type=str,
                        help='Goal image filename')

    args = parser.parse_args(argv)

    in_img = None; gl_img = None;
    if args.test_filename is not None:
      print 'Input image: ' + args.test_filename
      in_img = cv2.imread(args.test_filename)[:,:,0]
    else:
      in_img = cv2.imread('test_book.png')[:,:,0]
    if args.goal_filename is not None:
      print 'Goal image: ' + args.goal_filename
      gl_img = cv2.imread(args.goal_filename)[:,:,0]
    return in_img, gl_img

if __name__=='__main__':
    visualize = True

    in_img, gl_img = get_args()
    Ic = in_img.astype(np.uint8)
    Gc = None
    if gl_img is not None:
      Gc = gl_img.astype(np.uint8)

    con = PushController(visualize)
    best_start, best_end = con.get_best_push(Ic, Gc)

    print 'best_start ' + str(best_start[0]) + ' ' + str(best_start[1])
    print 'best_end ' + str(best_end[0]) + ' ' + str(best_end[1])
