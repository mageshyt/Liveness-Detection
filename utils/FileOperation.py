


def write_file(file_path, data, mode='w'):
    """
    Write data to a file
    :param file_path: Path to the file
    :param data: Data to write to the file
    :param mode: Mode in which the file should be opened ('w' for write, 'a' for append)
    :return: None
    """
    with open(file_path, mode) as file:
        file.write(data)

def read_file(file_path):
    """
    Read data from a file
    :param file_path: Path to the file
    :return: Data read from the file
    """
    with open(file_path, 'r') as file:
        data = file.read()
    return data

