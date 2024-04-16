#!/usr/bin/env python3
import rospy
from task4_env.srv import navigate, navigateResponse, pick, pickResponse, place, placeResponse, info, infoResponse
import re,os,csv,subprocess,sys,random

# Define constants and parameters
ALPHA = 0.001  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON = 0.02  # Initial exploration rate
MAX_EPISODES = 1000  # Maximum number of episodes

placed_balls = 0
pick_calls = 0
nav_calls = 0
learning = 1

sum_last_25_total_rewards_array = []
order_locations_toy={1:"green", 2:"blue", 3:"black", 4:"red"} #the order of state, when robot position is 0

q_table= []

# gets a location and handles the navigation to the location
def navigate_to_location(location):
    global nav_calls
    nav_calls += 1
    rospy.wait_for_service('/navigate')
    try:
        navigate_service = rospy.ServiceProxy('/navigate', navigate)
        response = navigate_service(location)
        return response.success
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed:", e)

# handles the pick toy action
def pick_toy():
    global pick_calls
    pick_calls += 1
    rospy.wait_for_service('/pick')
    try:
        state = get_state()
        #rospy.logwarn("the state when trying to pick: {}".format(state))
        robot_location = state[0]
        #rospy.logwarn("the robot location when trying to pick: {}".format(state[0]))
        toy_type = "None"
        for i in range(1,5):
            if robot_location == 4:
                rospy.logwarn("not suppused to pick the child")
                break
            if state[i] == robot_location:
                #rospy.logwarn("the ball that the robot soppused to pick: {}".format(order_locations_toy[i]))
                toy_type = order_locations_toy[i]
        pick_service = rospy.ServiceProxy('/pick', pick)
        response = pick_service(toy_type)
        rospy.loginfo("Picking up toy of type {} completed with success: {}".format(toy_type, response.success))
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Service call to pick failed: {}".format(e))

# handles the place toy action
def place_toy():
    global placed_balls
    rospy.wait_for_service('/place')
    try:
        place_service = rospy.ServiceProxy('/place', place)
        response = place_service()
        rospy.loginfo("Placing toy completed with success: {}".format(response.success))
        if(response.success):
            placed_balls += 1
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Service call to place failed: {}".format(e))

# retrieve all the info
def get_info():
    rospy.wait_for_service('/info')
    try:
        info_service = rospy.ServiceProxy('/info', info)
        response = info_service()
        return response.internal_info
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed:", e)

# helper function for get_state - retrieve the numbers from te info string
def extract_numbers(input_string):
    pattern = r"\b\d+\b"
    numbers = re.findall(pattern, input_string)
    numbers = [int(num) for num in numbers]
    return numbers

# gets a string in base 5 and returns an int in base 10
def base5_to_base10(number_str):
    base10_number = 0

    # Iterate through each digit of the number
    for i in range(len(number_str)):
        # Multiply the digit by 5 raised to the power of its position from the right
        base10_number += int(number_str[-(i + 1)]) * (5 ** i)

    return base10_number

# call /info and retrieved the state of the world
def get_state():
    info_string = get_info()
    state = extract_numbers(info_string)
    state = state[:5]
    if 5 in state:
        state.append(1)
    else:
        state.append(0)
    return state

# get a state of the world and returns the index of the state in q_table
def get_index_of_state(state):
    # Concatenate each number in the array into a single string
    number_str = ''
    for num in range(0,5):
        str_num = str(state[num])
        number_str += str_num
    index = base5_to_base10(number_str)
    if state[5] == 1:
        if state[0] != 4:
            return 3125 + state[0]
        else:
            index_of_ball = state.index(5) -1
            return 3125 + state[0] + index_of_ball
    return index

# get a row index in q_table that represent a state and return the max in that row(the best action to do)
def find_max_action_index(index):
    # Get the row at the specified index
    row = q_table[index]

    # Find the maximum value and its index in the row
    max_value = max(row)
    max_column_index = row.index(max_value)

    return max_column_index

# gets the state of the world, and returns the best action number for the robot to do based on q_table or random action in prob EPSILON
def choose_action(state_index):
    if learning == 1:
        options = ['random', 'by_policy']
        probabilities = [EPSILON, 1-EPSILON]
        choice = random.choices(options,probabilities,k=1)[0]
    else:
        choice = 'by_policy'
    if choice == 'random':
        action_index = random.randint(0,6)
    else:
        action_index = find_max_action_index(state_index)
    #print("choosing option: ", choice)
    return action_index

# gets an index of an action, calls the right wrapper of the function and performes it
def do_action(action_index):
    if(action_index >= 0 and action_index <= 4):
        if(nav_calls < 8):
            success = navigate_to_location(action_index)
            return True
        else:
            return False
    elif action_index == 5:
            if not terminate():
                success = pick_toy()
                return True
            else:
                rospy.logwarn("got to max nav/pick calls -> cant do action, did already {} pick and {} navigation.. GAME OVER!".format(pick_calls,nav_calls))
                return False
    elif action_index == 6:
        success = place_toy()
        return True
    else:
        print("Invalid action index")

# extract the last reward from the info
def get_lookup_str_from_info(lookup_str):
    input_string = get_info()
    # Find the last instance of "lookup_str:" in the input string
    last_reward_index = input_string.rfind(lookup_str)

    # If lookup_str is found, retrieve the number that comes after it
    if last_reward_index != -1:
        # Find the substring starting from the character after lookup_str
        substring_after_reward = input_string[last_reward_index +
                                              len(lookup_str):]

        # Use regular expression to extract the number
        import re
        match = re.search(r'-?\d+', substring_after_reward)
        if match:
            return int(match.group())

    # Return None if lookup_str is not found or if no number is found after it
    return None

# gets the indexes of the table that need to be updated and does that according to the q learning formula
def update_table(state_index, action_index):
    global q_table
    reward = get_lookup_str_from_info("reward:")
    current_value = q_table[state_index][action_index]
    next_state = get_state()
    next_state_index = get_index_of_state(next_state)
    next_action_index = find_max_action_index(next_state_index)
    next_best_action_value = q_table[next_state_index][next_action_index]
    q_table[state_index][action_index] = ((1-ALPHA) * current_value) + (ALPHA * (reward + (GAMMA * next_best_action_value)))

def write_csv_table():
    # Get the directory of the current file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate one directory up
    parent_directory = os.path.dirname(current_directory)

    # Specify the file name
    file_name = "q_table.csv"

    # Concatenate the current directory with the file name to get the CSV file path
    csv_file_path = os.path.join(parent_directory, file_name)

    # Write the q-table to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(q_table)

def read_csv_table():
    # Get the directory of the current file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate one directory up
    parent_directory = os.path.dirname(current_directory)

    # Specify the file name
    file_name = "q_table.csv"

    # Concatenate the current directory with the file name to get the CSV file path
    csv_file_path = os.path.join(parent_directory, file_name)

    # Read the Q-table data from the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each row of strings to floats and append to q_table
            q_table.append([float(value) for value in row])

# return true if we have reached the limit of navigation or pick calls
def terminate():
    if (nav_calls >= 8 or pick_calls >= 6):
        return True
    return False

def game():
    while placed_balls != 4: 
        state = get_state()
        state_index = get_index_of_state(state)
        action_index = choose_action(state_index)
        tried_to_do_action = do_action(action_index)
        if not tried_to_do_action:
            break
        update_table(state_index, action_index)

def loop():
    global learning, EPSILON, ALPHA, placed_balls, pick_calls, nav_calls, MAX_EPISODES
    try:
        learning = int(sys.argv[1])
    except:
        rospy.logwarn("by defualt set to learning mode")
    sum_last_X_total_rewards = 0
    if learning == 0:
        MAX_EPISODES = 10
        rospy.logwarn("ready to play {} games without learning".format(MAX_EPISODES))
    else:
        rospy.logwarn("ready to play {} games on learning mode".format(MAX_EPISODES))
    experiment_counter = 1
    read_csv_table()
    
    while(experiment_counter <= MAX_EPISODES):

        placed_balls = 0
        pick_calls = 0
        nav_calls = 0

        # Terminate skills_server.py
        os.system("rosnode kill skills_server_node")

        # Run the command in a separate process
        command = "rosrun task4_env skills_server.py"
        process_skills = subprocess.Popen(command, shell=True)

        # Call navigate to location 4
        success_nav = navigate_to_location(4)

        # Call navigate to location 4 again
        success_nav = navigate_to_location(4)

        # Terminate skills_server.py
        placed_balls = 0
        pick_calls = 0
        nav_calls = 0
        os.system("rosnode kill skills_server_node")

        # Launch skills_server.py
        # Run the command in a separate process
        command = "rosrun task4_env skills_server.py"
        process_skills = subprocess.Popen(command, shell=True)

        # start one experiment
        game()

        sum_last_X_total_rewards += get_lookup_str_from_info("total rewards:")
        if experiment_counter % 25 == 0 and learning == 1 :
            sum_last_25_total_rewards_array.append(sum_last_X_total_rewards)
            print("progress of sum last 25 iterations:\n", sum_last_25_total_rewards_array)
            sum_last_X_total_rewards = 0
            EPSILON -= 0.000002
            ALPHA -= 0.000001
            write_csv_table()
        rospy.logwarn("finished game number {}".format(experiment_counter))
        experiment_counter += 1
        
        
    if(learning == 0):
        rospy.logwarn("average of {}  in {} games: ".format(sum_last_X_total_rewards / MAX_EPISODES,MAX_EPISODES))
        
    # Close the skills_server.py subprocess at the end of the loop
    process_skills.kill()
    

if __name__ == "__main__":
    rospy.loginfo("Starting q_control node...")
    rospy.init_node('q_control')
    loop()
    # value of sum of 25 games of the 1000 first games:  [-51, 158, -1, 179, 339, 312, 489, 555, 623 
    #                                                      567, 790, 532, 666, 635, 655, 730, 1007, 931,
    #                                                      1018, 944, 847, 1081, 1017, 1013, 848, 901, 1004,
    #                                                      1034, 1118, 1196, 1318, 1235, 1296, 1169, 1138, 1288,
    #                                                      1161, 1120, 1434, 1309]