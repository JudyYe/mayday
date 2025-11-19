#!/bin/bash

set -x

# Check if at least 2 arguments are provided (T and command)
if [ $# -lt 2 ]; then
    echo "Usage: $0 <T> <command> [args...]"
    echo "Runs the command T times, with the first T-1 in background and the last in foreground"
    exit 1
fi

# Get T (number of times to run)
T=$1
shift  # Remove T from arguments, leaving the command and its args

# Validate T is a positive integer
if ! [[ "$T" =~ ^[0-9]+$ ]] || [ "$T" -le 0 ]; then
    echo "Error: T must be a positive integer"
    exit 1
fi

# Run the command T-1 times in background
for ((i=1; i<T; i++)); do
    "$@" &
done

# Run the last one in foreground (no &)
"$@"

