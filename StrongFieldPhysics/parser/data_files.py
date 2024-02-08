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

def remove_empty_lines(lines, remove_hash=False): # used after read_header
    new_lines = []
    for line in lines:
        line = line.replace('\n','')
        if line.replace('#', '').replace('\n','').strip() == '':
            continue
        if remove_hash:
            line = line.replace('#', '')
        new_lines.append(line)
    return new_lines

def get_header_info(header_list, key):
    for line in header_list:
        if line.startswith('# '+key):
            return line.split(':')[-1].strip()