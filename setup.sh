#!/bin/bash

VENV_NAME=".venv"

if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install it and try again."
    exit 1
fi

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/n): " choice
    case "$choice" in
        y|Y )
            echo "Removing existing virtual environment..."
            rm -rf $VENV_NAME
            echo "Creating new virtual environment..."
            python3 -m venv $VENV_NAME
            ;;
        n|N )
            echo "Using existing virtual environment."
            ;;
        * )
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "Creating new virtual environment..."
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate

pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Requirements installed successfully."
else
    echo "requirements.txt not found. No packages were installed."
fi

echo "Setup complete."
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate
echo "Virtual environment activated. Run 'deactivate' to exit."