from utils.parse_images import handle_data


def main():
    data = handle_data('./data.json')
    training_data = data['training']
    test_data = data['test']

    for i in range(0):


main()
