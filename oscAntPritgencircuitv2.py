# With modifications by Pritesh (2025)

# With modifications by Ant
# Copyright 2022 Oscar Higgott

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import stim
from typing import Callable, Set, List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


def append_anti_basis_error(circuit: stim.Circuit, targets: List[int], p: float, basis: str) -> None:
    if p > 0:
        if basis == "X":
            circuit.append_operation("Z_ERROR", targets, p)
        elif basis == "Z":
            circuit.append_operation("X_ERROR", targets, p)
        elif basis == "Y":
            circuit.append_operation("Y_ERROR", targets, p)  # already added


@dataclass
class CircuitGenParameters:
    code_name: str
    task: str
    rounds: int
    distance: int = None
    x_distance: int = None
    z_distance: int = None
    after_clifford_depolarization: float = 0
    before_round_data_depolarization: float = 0
    before_measure_flip_probability: float = 0
    after_reset_flip_probability: float = 0
    exclude_other_basis_detectors: bool = False

    def append_begin_round_tick(
            self,
            circuit: stim.Circuit,
            data_qubits: List[int]
    ) -> None:
        circuit.append_operation("TICK", [])
        if self.before_round_data_depolarization > 0:
            circuit.append_operation("DEPOLARIZE1", data_qubits, self.before_round_data_depolarization)
            
    def append_unitary_1(
            self,
            circuit: stim.Circuit,
            name: str,
            targets: List[int]
    ) -> None:
        circuit.append_operation(name, targets)
        
        if self.after_clifford_depolarization > 0:
            circuit.append_operation("DEPOLARIZE1", targets, self.after_clifford_depolarization)
            pass
        
    def append_unitary_2(
            self,
            circuit: stim.Circuit,
            name: str,
            targets: List[int],
    ) -> None:
        circuit.append_operation(name, targets)
        if self.after_clifford_depolarization > 0:
            circuit.append_operation("DEPOLARIZE2", targets, self.after_clifford_depolarization)

    def append_reset(
            self,
            circuit: stim.Circuit,
            targets: List[int],
            basis: str = "Z"
    ) -> None:
        circuit.append_operation("R" + basis, targets)
        append_anti_basis_error(circuit, targets, self.after_reset_flip_probability, basis)

    def append_measure(self, circuit: stim.Circuit, targets: List[int], basis: str = "Z") -> None:
        append_anti_basis_error(circuit, targets, self.before_measure_flip_probability, basis)
        circuit.append_operation("M" + basis, targets)

    def append_measure_reset(
            self,
            circuit: stim.Circuit,
            targets: List[int],
            basis: str = "Z"
    ) -> None:
        append_anti_basis_error(circuit, targets, self.before_measure_flip_probability, basis)
        circuit.append_operation("MR" + basis, targets)
        append_anti_basis_error(circuit, targets, self.after_reset_flip_probability, basis)


def finish_surface_code_circuit(
        coord_to_index: Callable[[complex], int],
        data_coords: Set[complex],
        x_measure_coords: Set[complex],   # These are the former "X" ancillas…
        z_measure_coords: Set[complex],
        params: CircuitGenParameters,
        x_order: List[complex],
        z_order: List[complex],
        x_observable: List[complex],
        z_observable: List[complex],
        is_memory_x: bool,  # >>> ZY CHANGE: interpret True == Y memory now
        *,
        exclude_other_basis_detectors: bool = False,
        wraparound_length: Optional[int] = None
) -> stim.Circuit:
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")
    if params.x_distance is not None and (params.x_distance < 2 or
                                          params.z_distance < 2):
        raise ValueError("Need a distance >= 2")

    # >>> ZY CHANGE: alias "X" sets as "Y" sets for clarity/use.
    y_measure_coords = x_measure_coords
    y_measurement_qubits: List[int] = []

    # >>> ZY CHANGE: observables: reuse the prior X-string geometry for Y.
    y_observable = x_observable

    # Choose basis-specific sets/observable (True -> Y, False -> Z)
    chosen_basis_observable = y_observable if is_memory_x else z_observable
    chosen_basis_measure_coords = y_measure_coords if is_memory_x else z_measure_coords

    # Index the measurement qubits and data qubits.
    p2q: Dict[complex, int] = {}
    for q in data_coords:
        p2q[q] = coord_to_index(q)
    for q in y_measure_coords:
        p2q[q] = coord_to_index(q)
    for q in z_measure_coords:
        p2q[q] = coord_to_index(q)

    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}

    data_qubits = [p2q[q] for q in data_coords]
    measurement_qubits = [p2q[q] for q in y_measure_coords]
    measurement_qubits += [p2q[q] for q in z_measure_coords]
    y_measurement_qubits = [p2q[q] for q in y_measure_coords]

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    y_measurement_qubits.sort()

    # Reverse index the measurement order used for defining detectors
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}
    for q in data_qubits:
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)

    # List out gate targets using given interaction orders.
    # Split: Y-stabilizer (replaces previous X) and Z-stabilizer.
    y_cy_targets: List[List[int]] = [[], [], [], []]
    z_cnot_targets: List[List[int]] = [[], [], [], []]

    for k in range(4):
        # Y-stabilizers: ancilla (was x_measure_coords) is control; data is target (use CY).
        for measure in sorted(y_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]  # (order list reused from previous X branch)
            if data in p2q:
                y_cy_targets[k].append(p2q[measure])  # control = ancilla
                y_cy_targets[k].append(p2q[data])     # target  = data
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                y_cy_targets[k].append(p2q[measure])
                y_cy_targets[k].append(p2q[data_wrapped])

        # Z-stabilizers: data is control; ancilla is target (CNOT).
        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                z_cnot_targets[k].append(p2q[data])      # control = data
                z_cnot_targets[k].append(p2q[measure])   # target  = ancilla
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                z_cnot_targets[k].append(p2q[data_wrapped])
                z_cnot_targets[k].append(p2q[measure])

    # Build the repeated actions that make up the surface code cycle
    cycle_actions = stim.Circuit()
    params.append_begin_round_tick(cycle_actions, data_qubits)
    # (Ancilla preparation/cleanup remains H; you only asked to switch data-qubit bases/errors.)
    params.append_unitary_1(cycle_actions, "H", y_measurement_qubits)
    for k in range(4):
        cycle_actions.append_operation("TICK", [])
        # Y stabilizer CYs
        params.append_unitary_2(cycle_actions, "CY", y_cy_targets[k])
        # Z stabilizer CNOTs
        params.append_unitary_2(cycle_actions, "CNOT", z_cnot_targets[k])
    cycle_actions.append_operation("TICK", [])
    params.append_unitary_1(cycle_actions, "H", y_measurement_qubits)
    cycle_actions.append_operation("TICK", [])
    params.append_measure_reset(cycle_actions, measurement_qubits)  # MRZ on ancillas (unchanged)

    # Build the start of the circuit (resets)
    head = stim.Circuit()
    for k, v in sorted(q2p.items()):
        head.append_operation("QUBIT_COORDS", [k], [v.real, v.imag])

    # >>> ZY CHANGE (critical):
    # Use Y basis for data if is_memory_x==True, else Z; also error before/after handled via append_anti_basis_error
    basis_char = 'Y' if is_memory_x else 'Z'
    params.append_reset(head, data_qubits, basis_char)  # RY/Y_ERROR or RZ/X_ERROR as configured
    params.append_reset(head, measurement_qubits)       # ancillas in Z as before
    head += cycle_actions
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[measure])],
            [measure.real, measure.imag, 0.0]
        )

    # Body (detectors comparing to previous cycles)
    body = cycle_actions.copy()
    m = len(measurement_qubits)
    body.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
    for m_index in measurement_qubits:
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            body.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0]
            )

    # End of the circuit (data measurements + detectors + observable)
    tail = stim.Circuit()

    # >>> ZY CHANGE (critical):
    # For the chosen memory basis, measure data in Y (or Z) with the corresponding pre-meas anti-basis error.
    params.append_measure(tail, data_qubits, basis_char)  # MY with Y_ERROR or MZ with X_ERROR

    # Detectors from final data measurements (unchanged geometry)
    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detectors: List[int] = []
        for delta in z_order:
            data = measure + delta
            if data in p2q:
                detectors.append(-len(data_qubits) + data_coord_to_order[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                detectors.append(-len(data_qubits) + data_coord_to_order[data_wrapped])
        detectors.append(-len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure])
        detectors.sort(reverse=True)
        tail.append_operation("DETECTOR", [stim.target_rec(x) for x in detectors], [measure.real, measure.imag, 1.0])

    # Logical observable
    obs_inc: List[int] = []
    for q in chosen_basis_observable:
        obs_inc.append(-len(data_qubits) + data_coord_to_order[q])
    obs_inc.sort(reverse=True)
    tail.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(x) for x in obs_inc], 0.0)

    return head + body * (params.rounds - 1) + tail


def generate_rotated_surface_code_circuit(
        params: CircuitGenParameters,
        is_memory_x: bool,   # >>> ZY CHANGE: interpret True as Y-memory
        x_cnot_order,
        z_cnot_order
) -> stim.Circuit:
    if params.distance is not None:
        x_distance = params.distance
        z_distance = params.distance
    else:
        x_distance = params.x_distance
        z_distance = params.z_distance

    # Place data qubits
    data_coords: Set[complex] = set()
    x_observable: List[complex] = []
    z_observable: List[complex] = []
    for x in [i + 0.5 for i in range(z_distance)]:
        for y in [i + 0.5 for i in range(x_distance)]:
            q = x * 2 + y * 2 * 1j
            data_coords.add(q)
            if y == 0.5:
                z_observable.append(q)
            if x == 0.5:
                x_observable.append(q)

    # Place measurement qubits.
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    for x in range(z_distance + 1):
        for y in range(x_distance + 1):
            q = x * 2 + y * 2j
            on_boundary_1 = x == 0 or x == z_distance
            on_boundary_2 = y == 0 or y == x_distance
            parity = (x % 2) != (y % 2)
            if on_boundary_1 and parity:
                continue
            if on_boundary_2 and not parity:
                continue
            if parity:
                x_measure_coords.add(q)   # will be treated as Y ancillas
            else:
                z_measure_coords.add(q)

    # Default orders converted from indices
    x_order: List[complex] = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]
    z_order: List[complex] = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    clockwise_order: List[complex] = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    x_order = [clockwise_order[i] for i in x_cnot_order]
    z_order = [clockwise_order[i] for i in z_cnot_order]

    def coord_to_idx(q: complex) -> int:
        q = q - math.fmod(q.real, 2) * 1j
        return int(q.real + q.imag * (z_distance + 0.5))

    return finish_surface_code_circuit(
        coord_to_idx,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        x_order,
        z_order,
        x_observable,
        z_observable,
        is_memory_x,  # True means Y in this ZY variant
        exclude_other_basis_detectors=params.exclude_other_basis_detectors
    )


def _generate_unrotated_surface_or_toric_code_circuit(
        params: CircuitGenParameters,
        is_memory_x: bool,  # >>> ZY CHANGE: interpret True as Y-memory
        is_toric: bool,
        z_cnot_order,
        x_cnot_order
) -> stim.Circuit:
    d = params.distance
    assert params.rounds > 0

    # Place qubits
    data_coords: Set[complex] = set()
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    x_observable: List[complex] = []
    z_observable: List[complex] = []
    length = 2 * d if is_toric else 2 * d - 1
    for x in range(length):
        for y in range(length):
            q = x + y * 1j
            parity = (x % 2) != (y % 2)
            if parity:
                if x % 2 == 0:
                    z_measure_coords.add(q)
                else:
                    x_measure_coords.add(q)  # treated as Y ancillas
            else:
                data_coords.add(q)
                if x == 0:
                    x_observable.append(q)
                if y == 0:
                    z_observable.append(q)

    clockwise_order: List[complex] = [-1j, 1, 1j, -1]
    z_order = [clockwise_order[i] for i in z_cnot_order]
    x_order = [clockwise_order[i] for i in x_cnot_order]

    def coord_to_idx(q: complex) -> int:
        return int(q.real + q.imag * length)

    return finish_surface_code_circuit(
        coord_to_idx,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        x_order,
        z_order,
        x_observable,
        z_observable,
        is_memory_x,  # True means Y in this ZY variant
        exclude_other_basis_detectors=params.exclude_other_basis_detectors,
        wraparound_length=2 * d if is_toric else None
    )


def generate_surface_or_toric_code_circuit_from_params(params: CircuitGenParameters, z_cnot_order, x_cnot_order) -> stim.Circuit:
    if params.code_name == "surface_code":
        if params.task == "rotated_memory_x":
            return generate_rotated_surface_code_circuit(params, True, z_cnot_order, x_cnot_order)
        elif params.task == "rotated_memory_z":
            return generate_rotated_surface_code_circuit(params, False, z_cnot_order, x_cnot_order)
        elif params.task == "unrotated_memory_x":
            if params.distance is None:
                raise NotImplementedError('Rectangular unrotated memories are not currently supported')
            return _generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=True,
                is_toric=False,
                z_cnot_order=z_cnot_order,
                x_cnot_order=x_cnot_order)
        elif params.task == "unrotated_memory_z":
            if params.distance is None:
                raise NotImplementedError('Rectangular unrotated memories are not currently supported')
            return _generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=False,
                is_toric=False,
                x_cnot_order=x_cnot_order,
                z_cnot_order=z_cnot_order)
    elif params.code_name == "toric_code":
        if params.distance is None:
            raise NotImplementedError('Rectangular toric codes are not currently supported')
        if params.task == "unrotated_memory_x":
            return _generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=True,
                is_toric=True,
                x_cnot_order=x_cnot_order,
                z_cnot_order=z_cnot_order)
        elif params.task == "unrotated_memory_z":
            return _generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=False,
                is_toric=True,
                x_cnot_order=x_cnot_order,
                z_cnot_order=z_cnot_order)

    raise ValueError(f"Unrecognised task: {params.task}")


def generate_circuit(
        code_task: str,
        *,
        rounds: int,
        distance: int = None,
        x_distance: int = None,
        z_distance: int = None,
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        exclude_other_basis_detectors: bool = False,
        x_cnot_order: list = [2,3,1,0],
        z_cnot_order: list = [2,1,3,0]
) -> stim.Circuit:
    """
    In this ZY variant:
      - `code_task` "...memory_x" means Y-memory (RY/MY on data, Y_ERROR around those ops).
      - Z branch stays Z as before.
    """
    if distance is not None:
        pass
    elif x_distance is not None and z_distance is not None:
        pass
    else:
        raise ValueError('Either the distance parameter or x_distance and z_distance parameters must be specified')
    code_name, task = code_task.split(":")
    if code_name in ["surface_code", "toric_code"]:
        params = CircuitGenParameters(
            code_name=code_name,
            task=task,
            rounds=rounds,
            distance=distance,
            x_distance=x_distance,
            z_distance=z_distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            exclude_other_basis_detectors=exclude_other_basis_detectors,
        )
        return generate_surface_or_toric_code_circuit_from_params(params, x_cnot_order, z_cnot_order)
    else:
        raise ValueError(f"Code name {code_name} not recognised")
