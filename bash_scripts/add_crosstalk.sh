#!/bin/zsh

# This script adds CORRELATED_ERROR lines to the circuits
# After each PAULI_CHANNEL_2() _______ , it takes the indices of the qubits, and applies:
#   CORRELATED_ERROR(0.0005) X<qubit1> X<qubit2>

# Directory containing with_crosstalk
base_dir="$(dirname "$0")/with_crosstalk"

for subdir in x z; do
  for stimfile in "$base_dir/$subdir"/*.stim; do
    tmpfile="${stimfile}.tmp"
    awk '
      {
        print $0
        # Only process lines containing PAULI_CHANNEL_2
        if (index($0, "PAULI_CHANNEL_2(") > 0) {
          n = split($0, fields, " ")
          # Find indentation
          indent = ""
          for (i = 1; i <= length($0); i++) {
            c = substr($0, i, 1)
            if (c != " " && c != "\t") break
            indent = indent c
          }
          # For each consecutive pair after the PAULI_CHANNEL_2(...) part
          # Find where the indices start (after the last ")")
          idx_start = 0
          for (i = 1; i <= n; i++) {
            if (fields[i] ~ /\)$/) { idx_start = i + 1; break; }
          }
          for (i = idx_start; i < n; i += 2) {
            if ((i+1) <= n) {
              print indent "CORRELATED_ERROR(0.0005) X" fields[i] " X" fields[i+1]
            }
          }
        }
      }
    ' "$stimfile" > "$tmpfile" && mv "$tmpfile" "$stimfile"
  done
done