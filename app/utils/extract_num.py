import os


def extract_check(filedir):
    numbers = []
    for file_name in os.listdir(filedir):
        number_part = float(file_name.split('_')[1].split('.pth')[0])  # Извлекаем цифру
        numbers.append(number_part)
    max_filename = f'checkpoint_{max(numbers)}.pth'
    return max_filename


def extract_number(filename):
    print(f"load checkpoint: {filename.split('_')[1]}")
    return float(filename.split('_')[1].split('.pth')[0])