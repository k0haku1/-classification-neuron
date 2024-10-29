import os
import shutil
from sklearn.model_selection import train_test_split

# Путь к исходным данным
dataset_dir = 'C:/Users/angro/OneDrive/Рабочий стол/flower_photos'


class_daisy_files = os.listdir(os.path.join(dataset_dir, 'daisy'))
class_dandelion_files = os.listdir(os.path.join(dataset_dir, 'dandelion'))
class_roses_files = os.listdir(os.path.join(dataset_dir, 'roses'))
class_sunflowers_files = os.listdir(os.path.join(dataset_dir, 'sunflowers'))
class_tulips_files = os.listdir(os.path.join(dataset_dir, 'tulips'))

# Разделяем данные на обучающую и валидационную части
train_class_daisy, val_class_daisy = train_test_split(class_daisy_files, test_size=0.2)
train_class_dandelion, val_class_dandelion = train_test_split(class_dandelion_files, test_size=0.2)
train_class_roses, val_class_roses = train_test_split(class_roses_files, test_size=0.2)
train_class_sunflowers, val_class_sunflowers = train_test_split(class_sunflowers_files, test_size=0.2)
train_class_tulips, val_class_tulips = train_test_split(class_tulips_files, test_size=0.2)

# Создаем папки для обучения и валидации
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_daisy', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_dandelion', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_roses_files', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_sunflowers', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_tulips', exist_ok=True)

os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_daisy', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_dandelion', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_roses_files', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_sunflowers', exist_ok=True)
os.makedirs('C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_tulips', exist_ok=True)

# Копируем файлы в соответствующие директории
for file in train_class_daisy:
    shutil.copy(os.path.join(dataset_dir, 'daisy', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_daisy')
for file in val_class_daisy:
    shutil.copy(os.path.join(dataset_dir, 'daisy', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_daisy')

for file in train_class_dandelion:
    shutil.copy(os.path.join(dataset_dir, 'dandelion', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_dandelion')
for file in val_class_dandelion:
    shutil.copy(os.path.join(dataset_dir, 'dandelion', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_dandelion')

for file in train_class_roses:
    shutil.copy(os.path.join(dataset_dir, 'roses', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_roses_files')
for file in val_class_roses:
    shutil.copy(os.path.join(dataset_dir, 'roses', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_roses_files')

for file in train_class_sunflowers:
    shutil.copy(os.path.join(dataset_dir, 'sunflowers', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_sunflowers')
for file in val_class_sunflowers:
    shutil.copy(os.path.join(dataset_dir, 'sunflowers', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_sunflowers')

for file in train_class_tulips:
    shutil.copy(os.path.join(dataset_dir, 'tulips', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/train/class_tulips')
for file in val_class_tulips:
    shutil.copy(os.path.join(dataset_dir, 'tulips', file), 'C:/Users/angro/OneDrive/Рабочий стол/flowers/val/class_tulips')