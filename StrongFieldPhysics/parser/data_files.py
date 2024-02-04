# Read data files

def read_header(path):
    with open(path) as f:
        lines = f.readlines()
    header_list = []
    for line in lines:
        if line.startswith('#'):
            header_list.append(line)
        else:
            break
    return header_list

def get_header_info(header_list, key):
    for line in header_list:
        if line.startswith('# '+key):
            return line.split(':')[-1].strip()