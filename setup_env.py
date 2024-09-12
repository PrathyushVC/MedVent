import os
import subprocess
import sys
import argparse

def create_virtual_environment(env_name='venv'):
    """Create a virtual environment."""
    print(f"Creating virtual environment: {env_name}")
    subprocess.check_call([sys.executable, '-m', 'venv', env_name])

def install_requirements(env_name='venv'):
    """Install requirements from requirements.txt."""
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        print(f"Installing requirements from {requirements_file}")
        pip_executable = os.path.join(env_name, 'Scripts', 'pip') if os.name == 'nt' else os.path.join(env_name, 'bin', 'pip')
        subprocess.check_call([pip_executable, 'install', '-r', requirements_file])
    else:
        print(f"No requirements.txt found in the current directory.")

def add_subdirectories_to_sys_path(base_directory):
    """Add all subdirectories of the base directory to sys.path."""
    for root, dirs, files in os.walk(base_directory):
        for dir in dirs:
            subdirectory = os.path.join(root, dir)
            if subdirectory not in sys.path:
                sys.path.append(subdirectory)
                print(f"Added to sys.path: {subdirectory}")

def main():

    parser = argparse.ArgumentParser(description='Set up a Python virtual environment and install dependencies.')
    parser.add_argument('--env-name', type=str, default='venv', help='Name of the virtual environment (default: venv)')

    args = parser.parse_args()
    env_name = args.env_name
    print()

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or (env_name == 'None'):
        print("You are already in a virtual environment.")
        print(f"Using the existing virtual environment: {sys.prefix}")

        add_subdirectories_to_sys_path(os.getcwd())

    else:
        create_virtual_environment(env_name)
    
    install_requirements(env_name)
    base_directory = os.getcwd()
    add_subdirectories_to_sys_path(base_directory)
    print("Setup complete! Activate your virtual environment and run your script.")

if __name__ == '__main__':
    main()