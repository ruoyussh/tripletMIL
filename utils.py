

def read_split(split_path):
    file = open(split_path)
    lst = []
    for line in file:
        lst.append(line.strip())
    return lst


def read_patient_labels(patient_list_path):
    file = open(patient_list_path)
    patient_labels = {}
    for line in file:
        line = line.strip()
        line = line.split(' ')
        patient_id = line[0]
        label = int(line[1])
        patient_labels[patient_id] = label
    return patient_labels