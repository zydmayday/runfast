import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':
    type = 3
    PLAYER_KEY = ['nn_train1', 'nn_train2', 'nn_train3']
    test_file = 'test_winners_nn'
    if type == 2:
        PLAYER_KEY = ['dn_train1', 'dn_train2', 'dn_train3']
        test_file = 'test_winners_dn'
    elif type == 3:
        PLAYER_KEY = ['dn_memory_train1', 'dn_memory_train2', 'dn_memory_train3']
        test_file = 'test_winners_dn_with_memory'
    test_data = pickle.load(open(test_file))
    # print test_data
    x_axis_list = test_data.keys()
    x_axis_list.sort()
    y1_list = []
    y2_list = []
    y3_list = []
    for k in x_axis_list:
        item = test_data[k]
        match_num = item[PLAYER_KEY[0]]['win'] + item[PLAYER_KEY[1]]['win'] + item[PLAYER_KEY[2]]['win']
        y1_list.append(float(item[PLAYER_KEY[0]]['win']) / match_num)
        y2_list.append(float(item[PLAYER_KEY[1]]['win']) / match_num)
        y3_list.append(float(item[PLAYER_KEY[2]]['win']) / match_num)

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
    if type == 1:
        plt.savefig('nn_test_win_percent.png')
    elif type  == 2:
        plt.savefig('dn_test_win_percent.png')
    elif type == 3:
        plt.savefig('dn_memory_test_win_percent.png')