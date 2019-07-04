from .primitives import (CNOT_X_basis,
                         CCNOT_X_basis,
                         majority_gate,
                         unmajority_add_gate,
                         unmajority_add_parallel_gate)

from .ripple_carry_adder import (REG_NAME,
                                 assign_registers_to_line_or_cycle,
                                 get_qubit_registers_for_adder,
                                 adder,
                                 get_n_bit_adder_results,
                                 get_success_probabilities_from_results,
                                 get_error_hamming_distributions_from_results)
