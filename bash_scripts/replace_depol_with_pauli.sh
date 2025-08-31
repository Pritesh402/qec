#!/bin/zsh

# This script replaces DEPOLARIZE1 and DEPOLARIZE2 lines in all .stim files in x/ and z/ directories with the specified PAULI_CHANNEL values.

TEMPLATE_DIR="$(dirname "$0")"

for dir in "$TEMPLATE_DIR"/x "$TEMPLATE_DIR"/z; do
  for file in "$dir"/*.stim; do
    if [[ -f "$file" ]]; then
      # Replace DEPOLARIZE1(0.0005)
      sed -i '' 's/DEPOLARIZE1(0.0005)/PAULI_CHANNEL_1(0.0002,0.00016667,0.00016667)/g' "$file"
      # Replace DEPOLARIZE2(0.0005)
      sed -i '' 's/DEPOLARIZE2(0.0005)/PAULI_CHANNEL_2(0.00016658333,0.00016658333,0.00016658333,0.00016658333,0.00000002778,0.00000002778,0.00000002778,0.00016658333,0.00000002778,0.00000002778,0.00000002778,0.00016658333,0.00000002778,0.00000002778,0.00000002778)/g' "$file"
    fi
  done
done
