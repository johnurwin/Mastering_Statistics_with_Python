import os

print(os.path.dirname(os.path.abspath(__file__)))
print(os.path.abspath(__file__))
current_directory = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(current_directory, '..', 'Python_Generated_Images'))
