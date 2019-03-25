import data
import math

def get_action_score(action_type):
    """
    Given the action_type of a session return a score to assign to the reference id house
    :param action_type: string of the action type
    :return: score
    """
    dict = {
        'clickout item': 5,
        'interaction item rating': 1,
        'interaction item info': 1,
        'interaction item image': 1,
        'interaction item deals': 1,
        'search for item': 10,
        'search for destination': 'reset',
        'change of sort order': None,
        'filter selection': None,
        'search for poi': None
    }

    if action_type not in dict.keys():
        print("error: WRONG ACTION TYPE")
        exit(0)
    return dict[action_type]


def time_weight(weight_function, session_length):
    """
    :param weight_function:
    :param session_lenght:
    :return:
    """
    weight_array = []
    if weight_function == 'exp':
        for i in range(session_length):
            weight_array.append(((i+1)/session_length)**3)
        return weight_array
    if weight_function == 'lin':
        for i in range(session_length):
            weight_array.append((i+1)/session_length)
        return weight_array
    print('error: weight function not defined')
    exit(0)



if __name__ == '__main__':
    #print(get_action_score('clickout_item'))
    print(time_weight('exp', 100))
    print(time_weight('lin', 100))

