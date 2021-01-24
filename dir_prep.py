import os
import random
#
# #Po sciagnieciu bazy danych uruchamiamy raz.
#
IMG_DIR = 'archive/images/Images/'
MAIN_DIR = 'archive/images/MAIN/'
TEST_DIR = 'archive/images/TEST/'

i = 1
for root, directories, files in os.walk(IMG_DIR):
    breed = root.split('-', 1)
    try:
        breed = breed[1].lower()
        os.rename(root, os.path.join(IMG_DIR,breed))
    except Exception as e:
        continue

# if not os.path.isdir(NEW_DIR):
#     os.mkdir(NEW_DIR)
# for dirpath in os.walk(IMG_DIR):
#     # print(f'Dirpath{dirpath}')
#     # print(f'Dirnames{dirnames}')
#     # print(f'filenames{filenames}')
#     #dirpath - folder
#     #filenames - tablica plikow
#     print(dirpath)
#         roll = random.randint(1, 3)
#         if roll ==3:
#             if not os.path.isdir(dirpath):
#                 os.mkdir(dirpath)
#             os.rename(os.path.join(dirpath,file), os.path.join(NEW_DIR,file))
# # i = 1
# for (dirpath, dirnames, filenames) in os.walk(IMG_DIR):
#     foo = dirpath.split('/')
#     if foo[3] != '':
#         breed = foo[3]
#         real_breed = f'{i}_{breed.split("-")[1]}'
#         foo1 = dirpath.split('/')
#         os.rename(dirpath,os.path.join(foo1[0],foo1[1],foo1[2],real_breed))
#         i = i + 1
# for (dirpath, dirnames, filenames) in os.walk(IMG_DIR):
#     i = 0
#     foo = dirpath.split('/')[3]
#     if foo != '':
#         temp = foo.split('_')
#         number = temp[0]
#         breed = temp[1]

# if not os.path.isdir(MAIN_DIR):
#     os.mkdir(MAIN_DIR)
#
# for root, directories, files in os.walk(IMG_DIR):
#     i = 1
#     try:
#         breed = root.split('-', 1)[1]
#         breed = breed.lower()
#     except Exception as e:
#         continue
#     for name in files:
#         os.rename(os.path.join(root,name),os.path.join(MAIN_DIR,f'{breed}.{i}.jpg'))
#         i += 1
#
# for root, directories, files in os.walk(MAIN_DIR):
#     for name in files:
#         roll = random.randint(1,3)
#         if roll == 3:
#             if not os.path.isdir(TEST_DIR):
#                 os.mkdir(TEST_DIR)
#             os.rename(os.path.join(root,name), os.path.join(TEST_DIR,name))
# i = 1
# for root, directories, files in os.walk(TEST_DIR):
#     for name in files:
#         new_name = name.split('.')[0]
#         os.rename(os.path.join(root,name), os.path.join(root,f'{new_name}.{i}'))
#         i += 1