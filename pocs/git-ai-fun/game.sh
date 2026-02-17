#!/bin/bash

choices=("rock" "paper" "scissors")

computer_choice=${choices[$((RANDOM % 3))]}

echo "Rock, Paper, Scissors, Shoot!"
echo ""
echo "Enter your choice (rock, paper, scissors):"
read -r player_choice

player_choice=$(echo "$player_choice" | tr '[:upper:]' '[:lower:]')

if [[ "$player_choice" != "rock" && "$player_choice" != "paper" && "$player_choice" != "scissors" ]]; then
  echo "Invalid choice: $player_choice"
  exit 1
fi

echo ""
echo "You chose:     $player_choice"
echo "Computer chose: $computer_choice"
echo ""

if [[ "$player_choice" == "$computer_choice" ]]; then
  echo "It's a tie!"
elif [[ "$player_choice" == "rock" && "$computer_choice" == "scissors" ]] ||
     [[ "$player_choice" == "paper" && "$computer_choice" == "rock" ]] ||
     [[ "$player_choice" == "scissors" && "$computer_choice" == "paper" ]]; then
  echo "You win!"
else
  echo "Computer wins!"
fi
