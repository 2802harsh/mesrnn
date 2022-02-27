#!/bin/bash

if [ -d "./save" ]; then
  echo "Save directory already exists"
else
  mkdir save
fi

if [ -d "./log" ]; then
  echo "Log directory already exists"
else
  mkdir log
fi

if [ -d "./plot" ]; then
  echo "Plot directory already exists"
else
  mkdir plot
fi

echo "Created all directories"

