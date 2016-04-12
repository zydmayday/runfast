import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import main

IMAGE = {'dn': 'test_dn.png', 'dn_wm': 'test_dn_wm.png', 'nn': 'test_nn.png', 'nn_wm': 'test_nn_wm.png'}

if __name__ == '__main__':
    type = raw_input('input the TYPE of the test: ')
    test_file = main.TEST[type]
    test_data = pickle.load(open(test_file))
    # print test_data
    x_axis_list = test_data.keys()
    x_axis_list.sort()
    y1_list = []
    y2_list = []
    y3_list = []
    for k in x_axis_list:
        item = test_data[k]
        match_num = item[main.PLAYER_LIST[type][0]]['win'] + item[main.PLAYER_LIST[type][1]]['win'] + item[main.PLAYER_LIST[type][2]]['win']
        y1_list.append(float(item[main.PLAYER_LIST[type][0]]['win']))
        y2_list.append(float(item[main.PLAYER_LIST[type][1]]['win']))
        y3_list.append(float(item[main.PLAYER_LIST[type][2]]['win']))

    # red_patch = mpatches.Patch(color='red', label='trainer1')
    # blue_patch = mpatches.Patch(color='blue', label='trainer2')
    # green_patch = mpatches.Patch(color='green', label='trainer3')
    # plt.legend(handles=[red_patch])
    # plt.plot(x_axis_list, y1_list, 'r-', x_axis_list, y2_list, 'b-', x_axis_list, y3_list, 'g-')
    plt.plot(x_axis_list, y1_list, 'r-', label='agent1')
    plt.plot(x_axis_list, y2_list, 'b-', label='agent2')
    plt.plot(x_axis_list, y3_list, 'g-', label='agent3')
    legend = plt.legend(loc='upper center', shadow=True)
    # plt.show()
    plt.savefig(IMAGE[type])