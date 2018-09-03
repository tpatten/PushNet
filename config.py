batch_size = 1 ## adjust this number to fit in memory capacity of your GPU
num_action = 100 ## number of candidates push actions to be sampled from current image, the number should be a multiple of batch_size
test_filename = "/container/Data/MITrenderings/abs_rect1/1_abs_rect1_pn_0.jpg"
goal_filename = "/container/Data/MITrenderings/abs_rect1/1_abs_rect1_pn_1.jpg"
push_vector = [91, 6, 75, 33]
#push_vector = None

### three differetn network architecture for comparison
arch = {
        'simcom':'push_net',
        'sim': 'push_net_sim',
        'nomem': 'push_net_nomem'
       }



