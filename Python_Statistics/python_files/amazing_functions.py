# Amazing functions
import os
import matplotlib.pyplot as plt

def back_one_enter_new(folder_name, filename):
# Get the absolute path of the current script's directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory and then enter Python_Generated_Images
    output_directory = os.path.join(current_directory, '..', folder_name)
    output_filename = filename
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path)

