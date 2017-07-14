import os, sys, numpy, random
from sympy import Symbol, solve

T_l = float(sys.argv[1])
T_s = float(sys.argv[2])
cov = float(sys.argv[3])

# of seconds for each tracefile
time_length = int(sys.argv[4])

# get bitrate levels (in Mbps)
min_bitrate = 0.2
max_bitrate = 4.3
steps = 10
bitrate_states_low_var = []
curr = min_bitrate
for x in range(0,steps):
    bitrate_states_low_var.append(curr)
    curr += ((max_bitrate-min_bitrate)/(steps-1))
    x += 1

# list of transition probabilities
transition_probabilities = []
# assume you can go steps-1 states away (we will normalize this to the actual scenario)
eq = -1
x = Symbol("x", positive=True)
for y in range(1, steps-1):
    eq += (1/x**y)
res = solve(eq)
switch_parameter = res[0]
for z in range(1, steps-1):
    transition_probabilities.append(1/switch_parameter*y)

# two variance levels
sigma_low = 1.0
sigma_high = 1.0

# probability of switching variance levels
variance_switch_prob = 0.2

# probability to stay in same state
prob_stay = 1 - 1 / T_l

# takes a state and decides what the next state is
def transition(state, variance):
    transition_prob = random.uniform(0,1)

    # pick next variance first
    variance_switch = random.uniform(0,1)
    next_variance = variance
    if ( variance_switch < variance_switch_prob ):
        if ( next_variance == sigma_low ):
            next_variance = sigma_high
        else:
            next_variance = sigma_low

    if transition_prob < prob_stay: # stay in current state
        return (state, next_variance)
    else: # pick appropriate state!
        next_state = state
        curr_pos = state
        # first find max distance that you can be from current state
        max_distance = max(curr_pos, len(bitrate_states_low_var)-1-curr_pos)
        # cut the transition probabilities to only have possible number of steps
        curr_transition_probabilities = transition_probabilities[0:max_distance]
        trans_sum = sum(curr_transition_probabilities)
        normalized_transitions = [x/trans_sum for x in curr_transition_probabilities]
        # generate a random number and see which bin it falls in to
        trans_switch_val = random.uniform(0,1)
        running_sum = 0
        num_switches = -1
        for ind in range(0, len(normalized_transitions) ):
            if ( trans_switch_val <= (normalized_transitions[ind] + running_sum) ): # this is the val
                num_switches = ind
                break
            else:
                running_sum += normalized_transitions[ind]

        # now check if there are multiple ways to move this many states away
        switch_up = curr_pos + num_switches
        switch_down = curr_pos - num_switches
        if ( switch_down >= 0 and switch_up <= (len(bitrate_states_low_var)-1) ): # can go either way
            x = random.uniform(0,1)
            if ( x < 0.5 ):
                return (switch_up, next_variance)
            else:
                return(switch_down, next_variance)
        elif switch_down >= 0: # switch down
            return(switch_down, next_variance)
        else: # switch up
            return(switch_up, next_variance)

current_state = random.randint(0,len(bitrate_states_low_var)-1)
current_variance = cov * bitrate_states_low_var[current_state]
time = 0
cnt = 0
while time < time_length:
    # prints timestamp (in seconds) and throughput (in Mbits/s)
    if cnt <= 0:
        noise = numpy.random.normal(0, current_variance, 1)[0]
        cnt = T_s
    gaus_val = max(0.1, bitrate_states_low_var[current_state] + noise)
    cnt -= 1
    print str(time) + " " + str(gaus_val)
    next_vals = transition(current_state, current_variance)
    if current_state != next_vals[0]:
        cnt = 0
    current_state = next_vals[0]
    current_variance = cov * bitrate_states_low_var[current_state]
    time += 1
