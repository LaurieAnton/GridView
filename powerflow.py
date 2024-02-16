import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

#---- Power Network Class ----#

class power_network:
    def __init__(self, filepath, recip_gamma=1):
        P, Q, V, theta, bus_types, num_buses, A, Y_br, Y_bus, S_base, data = data_grabber(filepath)
        # Main quantities, all vectors
        self.P = P
        self.Q = Q
        self.V = V
        self.theta = theta
        # Other system info
        self.num_buses = num_buses
        self.bus_types = bus_types
        self.A = A
        self.Y_br = Y_br
        self.Y_bus = Y_bus
        self.S_base = S_base
        self.filepath = filepath
        self.data = data
        self.generator_outputs = {}
        self.branch_df = None
        self.bus_df = None
        self.G = None
        self.fig = None
        self.results = {}

        # Hardcoded for now
        self.positions = {
            1: (1,  0),
            2: (2, -3),
            3: (7, -3),
            4: (7,  0),
            5: (3, -1),
            6: (3,  1),
            7: (7.5,  1.5),
            8: (8,  1.5),
            9: (7,  3),
            10: (5.5,  3),
            11: (4.5,  4),
            12: (2,  5),
            13: (3.5,  6),
            14: (5.5,  5)
        }

        # Initialize Richardson state
        # Get the different types of buses
        self.num_slack = 2
        self.num_PV = len(np.where(bus_types == 2)[0])
        self.num_PQ = len(np.where(bus_types == 1)[0])

        # Adjusting the initial state vector to include all necessary elements
        self.state = np.zeros(self.num_slack + 2*self.num_PV + 2*self.num_PQ)
        # Set initial power at slack bus
        self.state[:self.num_slack] = [self.P[0], self.Q[0]]
        # Set initial reactive power at PV buses
        self.state[self.num_slack:self.num_slack + self.num_PV] = self.Q[bus_types == 2]
        # Set initial angles at PV buses
        self.state[self.num_slack + self.num_PV:self.num_slack + 2*self.num_PV] = self.theta[bus_types == 2]
        # Set initial angles at PQ buses
        self.state[self.num_slack + 2*self.num_PV:self.num_slack + 2*self.num_PV + self.num_PQ] = self.theta[bus_types == 1]
        # Set initial voltages at PQ buses
        self.state[self.num_slack + 2*self.num_PV + self.num_PQ:] = self.V[bus_types == 1]
        self.recip_gamma = recip_gamma

        # Dynamics
        self.t = 0
        self.dt = 0
        self.t_hist = []
        self.state_hist = [self.state]

    def reset(self, reset_results=True):
        # Re-grab the data from the original filepath
        P, Q, V, theta, bus_types, num_buses, A, Y_br, Y_bus, S_base, data = data_grabber(self.filepath)

        # Reset main quantities
        self.P = P
        self.Q = Q
        self.V = V
        self.theta = theta

        # Reset other system info
        self.num_buses = num_buses
        self.bus_types = bus_types
        self.A = A
        self.Y_br = Y_br
        self.Y_bus = Y_bus
        self.S_base = S_base
        self.data = data
        self.generator_outputs = {}
        if reset_results: self.results = {}

        # Initialize Richardson state
        # Get the different types of buses
        self.num_slack = 2
        self.num_PV = len(np.where(bus_types == 2)[0])
        self.num_PQ = len(np.where(bus_types == 1)[0])

        # Adjusting the initial state vector to include all necessary elements
        self.state = np.zeros(self.num_slack + 2*self.num_PV + 2*self.num_PQ)
        # Set initial power at slack bus
        self.state[:self.num_slack] = [self.P[0], self.Q[0]]
        # Set initial reactive power at PV buses
        self.state[self.num_slack:self.num_slack + self.num_PV] = self.Q[bus_types == 2]
        # Set initial angles at PV buses
        self.state[self.num_slack + self.num_PV:self.num_slack + 2*self.num_PV] = self.theta[bus_types == 2]
        # Set initial angles at PQ buses
        self.state[self.num_slack + 2*self.num_PV:self.num_slack + 2*self.num_PV + self.num_PQ] = self.theta[bus_types == 1]
        # Set initial voltages at PQ buses
        self.state[self.num_slack + 2*self.num_PV + self.num_PQ:] = self.V[bus_types == 1]

        # Dynamics
        self.t = 0
        self.dt = 0
        self.t_hist = []
        self.dstate_dt = np.zeros(self.state.shape)
        self.state_hist = [self.state]

    # Create a Pre-Solved result dictionary
    def presolved_results(self):
        self.result_dataframes()
        self.create_figure()

        self.results['Presolved'] = {
            'bus_df': self.bus_df,
            'branch_df': self.branch_df,
            'graph': self.G,
            'figure': self.fig
        }

    # Newton powerflow solver
    def newton_raphson_power_flow(self, tol=1e-4, max_iter=10, 
                                  verbose=False, external=False, check_Qlims = True):

        # Load variables in local namespace
        Y_bus = self.Y_bus.copy()
        V = self.V.copy()
        theta = self.theta.copy()
        P = self.P.copy()
        Q = self.Q.copy()
        bus_types = self.bus_types.copy()
        data = self.data.copy()

        # Helper function to calculate -F(x) for Newton solver
        def power_mismatch(Y_bus, V, theta, P, Q, bus_types):
            N = len(V)
            nonslack_idx = np.where(bus_types != 3)[0] # Indices. Bus numbers = idx + 1
            PQ_idx = np.where(bus_types == 1)[0]
            P_calc = np.zeros(len(nonslack_idx))
            Q_calc = np.zeros(len(PQ_idx))

            # Skip k = 0 to avoid the slack bus when calling V, theta, Y_bus.
            for k in range(1,N):
                P_calc[k-1] = V[k]*np.sum([np.abs(Y_bus[k, n])*V[n]*np.cos(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)])

            # k uses the proper indices from V, theta, Y_bus, while k_idx runs through Q_calc.
            k_idx=0
            for k in PQ_idx:
                Q_calc[k_idx] = V[k]*np.sum([np.abs(Y_bus[k, n])*V[n]*np.sin(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)])
                k_idx+=1

            dP = P[nonslack_idx] - P_calc
            dQ = Q[PQ_idx] - Q_calc

            return np.concatenate((dP, dQ)), nonslack_idx, PQ_idx  # [P2 ... PN, Qi ... Qn]

        # Calculate Jacobian from analytical expressions
        def powerflow_jacobian(Y_bus, V, theta, nonslack_idx, PQ_idx):
            '''
            Inputs should be whole state vectors and indices from power mismatches.
            '''
            N = len(V)
            M = len(nonslack_idx) # Number of non-slack buses
            m = len(PQ_idx)       # Number of PQ (Load) buses

            J = np.zeros((M+m, M+m)) # M+m rows, M+m columns

            # Fill in J1 (dP/dtheta)
            k_idx=0
            for k in nonslack_idx:
                j_idx=0
                for j in nonslack_idx:
                    if k != j: # Off-diagonal elements
                        J[k_idx, j_idx] = V[k] *V[j] * np.abs(Y_bus[k, j]) * np.sin(theta[k] - theta[j] - np.angle(Y_bus[k, j]))
                    elif k == j: # Diagonal elements
                        J[k_idx, j_idx] = -V[k] * np.sum(  [V[n] * np.abs(Y_bus[k, n]) * np.sin(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N) if n != k]  )
                    j_idx += 1
                k_idx += 1

            # Fill in J2 (dP/dV)
            k_idx=0
            for k in nonslack_idx:
                j_idx=0
                for j in PQ_idx:
                    if k != j: # Off-diagonal elements
                        J[k_idx, M+j_idx] = V[k] * np.abs(Y_bus[k, j]) * np.cos(theta[k] - theta[j] - np.angle(Y_bus[k, j]))
                    elif k == j: # Diagonal elements
                        J[k_idx, M+j_idx] = V[k] * np.abs(Y_bus[k, k]) * np.cos(np.angle(Y_bus[k, k])) + np.sum(  [V[n] * np.abs(Y_bus[k, n]) * np.cos(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)]  )
                    j_idx += 1
                k_idx += 1

            # Fill in J3 (dQ/dtheta)
            k_idx=0
            for k in PQ_idx:
                j_idx=0
                for j in nonslack_idx:
                    if k != j: # Off-diagonal elements
                        J[M+k_idx, j_idx] = -V[k] * V[j] * np.abs(Y_bus[k, j]) * np.cos(theta[k] - theta[j] - np.angle(Y_bus[k, j]))
                    elif k == j: # Diagonal elements
                        J[M+k_idx, j_idx] = V[k] * np.sum(  [V[n] * np.abs(Y_bus[k, n]) * np.cos(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N) if n != k]  )
                    j_idx += 1
                k_idx += 1

            # Fill in J4 (dQ/dV)
            k_idx=0
            for k in PQ_idx:
                j_idx=0
                for j in PQ_idx:
                    if k != j: # Off-diagonal elements
                        J[M+k_idx, M+j_idx] = V[k] * np.abs(Y_bus[k, j]) * np.sin(theta[k] - theta[j] - np.angle(Y_bus[k, j]))
                    elif k == j: # Diagonal elements
                        J[M+k_idx, M+j_idx] = -V[k] * np.abs(Y_bus[k, k]) * np.sin(np.angle(Y_bus[k, k])) + np.sum([  V[n] * np.abs(Y_bus[k, n]) * np.sin(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)  ])
                    j_idx += 1
                k_idx += 1

            return J # (M+m)x(M+m)

        # Calculate Q at each iteration based on Newton solver results
        def recalculate_Q(Y_bus, V, theta, Q, bus_types, data, check_Qlims):
            N = len(V)
            PV_idx = np.where(bus_types == 2)[0]

            for k in PV_idx:
                Q[k] = V[k]*np.sum([np.abs(Y_bus[k, n])*V[n]*np.sin(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)])

                if check_Qlims:
                    QT =  9999
                    QB = -9999
                    gen_id = 0
                    for subdata in data['generator']:
                        if subdata['I'] == k+1:
                            QT = subdata['QT']/data['case'][0]['SBASE']
                            QB = subdata['QB']/data['case'][0]['SBASE']
                            gen_id = subdata['I']

                    # Check if reactive power limits are reached. If so, change from PV to PQ bus.
                    if Q[k] < QB:
                        Q[k] = QB
                        bus_types[k] = 1
                        print(f'Minimum reactive power limit reached for generator at bus {gen_id}. Changing bus type from PV to PQ.')
                    elif Q[k] > QT:
                        Q[k] = QT
                        bus_types[k] = 1
                        print(f'Maximum reactive power limit reached for generator at bus {gen_id}. Changing bus type from PV to PQ.')

            return Q, bus_types

        # Calculate P, Q for the slack bus once the Newton solver converges
        def recalculate_slack(Y_bus, V, theta, P, Q, bus_types):
            N = len(V)
            slack_idx = np.where(bus_types == 3)[0]

            for k in slack_idx:
                P[k] = V[k]*np.sum([np.abs(Y_bus[k, n])*V[n]*np.cos(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)])
                Q[k] = V[k]*np.sum([np.abs(Y_bus[k, n])*V[n]*np.sin(theta[k] - theta[n] - np.angle(Y_bus[k, n])) for n in range(N)])

            return P, Q

        # Initialize logs
        log = {'P': [], 'Q': [], 'V': [], 'theta': [], 'mismatch': [], 'norm': []}
        for iteration in range(max_iter):

            mismatch, nonslack_idx, PQ_idx = power_mismatch(Y_bus, V, theta, P, Q, bus_types)
            if verbose:
                print(f"Iteration {iteration}:")
                print("  Voltage Magnitudes:", V)
                print("  Voltage Angles:", theta)
                print("  Active Power:", P)
                print("  Reactive Power:", Q)
                print("  Power Mismatch:", mismatch)

            # ||F(x)||2
            norm_error = abs(np.linalg.norm(mismatch, ord=2))

            # Append results at each iteration for analysis
            log['P'].append(P.copy())
            log['Q'].append(Q.copy())
            log['V'].append(V.copy())
            log['theta'].append(theta.copy())
            log['mismatch'].append(mismatch.copy())
            log['norm'].append(norm_error.copy())

            # ||F(x)||2 < epsilon
            if norm_error < tol:
                print("Converged!")
                break
            if iteration == max_iter-1:
                raise RuntimeError('Maximum iterations reached, did not converge. Norm error:', norm_error)

            # Calculate Jacobian around x_k
            J = powerflow_jacobian(Y_bus, V, theta, nonslack_idx, PQ_idx)

            # Try to solve J * dx = - F
            try:
                correction = np.linalg.solve(J, mismatch) # Will spit out [theta2 ... thetaN, Vi ... Vn]^T
                pass
            except Exception as e:
                print("Solver Error:", e)
                break

            # Apply correction to PQ and PV buses
            theta[nonslack_idx] += correction[:len(nonslack_idx)] # First N-1 variables are theta
            V[PQ_idx] += correction[len(nonslack_idx):] # Next are V

            # Recalculate Q for PV buses
            Q, bus_types = recalculate_Q(Y_bus, V, theta, Q, bus_types, data, check_Qlims)

        # Recalculate slack requirements
        P, Q = recalculate_slack(Y_bus, V, theta, P, Q, bus_types)

        # Update class variables
        self.P = P.copy()
        self.Q = Q.copy()
        self.V = V.copy()
        self.theta = theta.copy()
        self.bus_types = bus_types.copy()
        self.log = log.copy()

        if external: return V, theta, P, Q, bus_types, log

    # Matrix implict solver?
    def matrix_implicit_power_flow(self):
        pass

    # Under decoupled assumptions, ODE version of powerflow, i.e. Richardson method
    def powerflow_decoupled_Richardson(self, states=None):
        
        if states is None:
            state = self.state
        else: state = states

        # Unpack state vector
        P_slack, Q_slack = state[:2] # Slack
        Q_gen  = state[2:2 + self.num_PV]
        theta_PV = state[2 + self.num_PV:2 + 2*self.num_PV]
        theta_PQ = state[2 + 2*self.num_PV:2 + 2*self.num_PV + self.num_PQ]
        V_load = state[2 + 2*self.num_PV + self.num_PQ:]
        
        # Construct full V and theta vectors
        V_mixed = np.copy(self.V)
        theta_mixed = np.copy(self.theta)
        V_mixed[np.where(self.bus_types == 1)[0]] = V_load  # Update V for PQ buses
        theta_mixed[np.where(self.bus_types == 2)[0]] = theta_PV  # Update theta for PV buses
        theta_mixed[np.where(self.bus_types == 1)[0]] = theta_PQ  # Update theta for PQ buses

        # Adjust P_spec and Q_spec for slack and PV buses
        P_mixed = np.copy(self.P)
        Q_mixed = np.copy(self.Q)
        P_mixed[0] = P_slack
        Q_mixed[0] = Q_slack
        Q_mixed[np.where(self.bus_types == 2)[0]] = Q_gen

        # Initialize derivatives
        self.dstate_dt = np.zeros_like(state)
        
        for k in range(len(self.P)):
            # Convert admittance matrix to polar form for bus k
            Y_mag = np.abs(self.Y_bus[k])
            Y_angle = np.angle(self.Y_bus[k])
            
            # Voltage angle differences for bus k
            theta_diff = theta_mixed[k] - theta_mixed
            
            # Calculate the active and reactive powers for bus k
            P_calc = V_mixed[k] * np.sum(V_mixed * Y_mag * np.cos(theta_diff - Y_angle))
            Q_calc = V_mixed[k] * np.sum(V_mixed * Y_mag * np.sin(theta_diff - Y_angle))
            
            # Calculate power mismatches for bus k
            Delta_P_k = P_mixed[k] - P_calc
            Delta_Q_k = Q_mixed[k] - Q_calc
            
            # Update derivatives
            if self.bus_types[k] == 3:  # Slack bus
                self.dstate_dt[0] = (P_slack - P_mixed[k]) * self.recip_gamma
                self.dstate_dt[1] = (Q_slack - Q_mixed[k]) * self.recip_gamma
            elif self.bus_types[k] == 2:  # PV bus
                self.dstate_dt[np.where(Q_gen == Q_mixed[k])[0] + 2] = (Q_slack - Q_mixed[k]) * self.recip_gamma
            elif self.bus_types[k] == 1:  # PQ bus
                idx_theta = np.where(theta_PQ == theta_mixed[k])[0] + 2 + 2*self.num_PV
                idx_V = np.where(V_load == V_mixed[k])[0] + 2 + 2*self.num_PV + self.num_PQ
                self.dstate_dt[idx_theta] = Delta_P_k * self.recip_gamma
                self.dstate_dt[idx_V] = Delta_Q_k * self.recip_gamma
        
        return self.dstate_dt

    # Add a dictionary item to results called Dynamics
    def dynamic_results(self):
        solution = np.column_stack(self.state_hist).T
        t = self.t_hist

        # Extracting specific state variables for plotting
        P_states = solution[:, :2]*self.S_base
        Q_states_PV = solution[:, 2:2 + self.num_PV]*self.S_base
        theta_states_PV = solution[:, 2 + self.num_PV:2 + 2*self.num_PV]*180/np.pi
        theta_states_PQ = solution[:, 2 + 2*self.num_PV:2 + 2*self.num_PV + self.num_PQ]*180/np.pi
        V_states = solution[:, -self.num_PQ:]

        # Plotting slack trajectory
        fig_P = go.Figure()
        fig_P.add_trace(go.Scatter(x=t, y=P_states[:, 0], mode='lines', name='P_slack'))
        fig_P.add_trace(go.Scatter(x=t, y=P_states[:, 1], mode='lines', name='Q_slack'))
        fig_P.update_layout(
            title='P, Q Trajectory (Slack Bus)',
            xaxis_title='Time',
            yaxis_title='Power (PU)'
        )
        fig_P.update_yaxes(gridcolor='lightgrey')

        # Plotting Q trajectories for PV buses
        fig_Q_PV = go.Figure()
        for i, q_state in enumerate(Q_states_PV.T):
            fig_Q_PV.add_trace(go.Scatter(x=t, y=q_state, mode='lines', name=f'Q_gen_bus_{np.where(self.bus_types == 2)[0][i] + 1}'))
        fig_Q_PV.update_layout(
            title='Q Trajectory (PV Buses)',
            xaxis_title='Time',
            yaxis_title='Reactive Power (PU)'
        )
        fig_Q_PV.update_yaxes(gridcolor='lightgrey')

        # Plotting Theta states for PV and PQ buses
        fig_theta = go.Figure()
        for i, theta_state_PV in enumerate(theta_states_PV.T):
            fig_theta.add_trace(go.Scatter(x=t, y=theta_state_PV, mode='lines', name=f'theta_bus_{np.where(self.bus_types == 2)[0][i] + 1}'))
        for i, theta_state_PQ in enumerate(theta_states_PQ.T):
            fig_theta.add_trace(go.Scatter(x=t, y=theta_state_PQ, mode='lines', name=f'theta_bus_{np.where(self.bus_types == 1)[0][i] + 1}'))
        fig_theta.update_layout(
            title='Theta States (Voltage Angles)',
            xaxis_title='Time',
            yaxis_title='Angle (Degrees)'
        )
        fig_theta.update_yaxes(gridcolor='lightgrey')

        # Plotting V states for PQ buses
        fig_V = go.Figure()
        for i, v_state in enumerate(V_states.T):
            fig_V.add_trace(go.Scatter(x=t, y=v_state * self.data['bus'][0]['BASKV'], mode='lines', name=f'V_bus_{np.where(self.bus_types == 1)[0][i] + 1}'))
        fig_V.update_layout(
            title='V States (Voltage Magnitudes at PQ Buses)',
            xaxis_title='Time, s',
            yaxis_title='Voltage, kV'
        )
        fig_V.update_yaxes(gridcolor='lightgrey')

        self.results['Dynamics'] = {
            'P_trajectories': fig_P,
            'Q_PV_trajectories': fig_Q_PV,
            'theta_trajectories': fig_theta,
            'V_trajectories': fig_V
        }

    # After powerflow, recover generator outputs -> REQUIRES ONLY 1 GEN PER BUS
    def extract_generator_outputs(self, external=False):
        # Reset at each call
        self.generator_outputs = {}
        for gen_data in self.data['generator']:
            bus_idx = gen_data['I'] - 1
            P_gen = self.P[bus_idx]  # Net active power into the bus
            Q_gen = self.Q[bus_idx]  # Net reactive power into the bus
            # Subtract the load (if any) to get the generator's output
            for load_data in self.data['load']:
                if load_data['I'] - 1 == bus_idx:
                    P_gen += load_data['PL']/self.S_base
                    Q_gen += load_data['QL']/self.S_base
            self.generator_outputs[bus_idx] = {'P': P_gen, 'Q': Q_gen} # Per unit
        if external: return self.generator_outputs
    
    def update_PQVtheta(self):
        # Extract updated values from the state
        P_slack_updated, Q_slack_updated = self.state[:self.num_slack]
        Q_gen_updated = self.state[self.num_slack:self.num_slack + self.num_PV]
        theta_PV_updated = self.state[self.num_slack + self.num_PV:self.num_slack + 2*self.num_PV]
        theta_PQ_updated = self.state[self.num_slack + 2*self.num_PV:self.num_slack + 2*self.num_PV + self.num_PQ]
        V_load_updated = self.state[self.num_slack + 2*self.num_PV + self.num_PQ:]

        # Update the P and Q vectors
        self.P[0] = P_slack_updated
        self.Q[0] = Q_slack_updated
        self.Q[np.where(self.bus_types == 2)[0]] = Q_gen_updated

        # Update the theta vector
        self.theta[np.where(self.bus_types == 2)[0]] = theta_PV_updated
        self.theta[np.where(self.bus_types == 1)[0]] = theta_PQ_updated

        # Update the V vector
        self.V[np.where(self.bus_types == 1)[0]] = V_load_updated

    def calculate_branch_flows(self, external=False):
        self.branch_flows = {}

        # Iterate over both line and transformer data
        for branch_type in ['branch', 'transformer2']:
            for branch_data in self.data[branch_type]:
                bus_i = branch_data['I'] - 1    # From
                bus_j = branch_data['J'] - 1    # To

                # Extract the branch admittance
                Y_ij = self.Y_bus[bus_i, bus_j]

                # Include shunt admittance if available
                B_ij = branch_data.get('B', 0)  # Default to 0 if not available
                Y_ij += 1j * B_ij

                # Calculate the branch impedance
                Z_ij = 1 / Y_ij if Y_ij != 0 else np.inf

                # Format the voltage as a complex phasor
                V_i = self.V[bus_i] * np.exp(1j * self.theta[bus_i])
                V_j = self.V[bus_j] * np.exp(1j * self.theta[bus_j])

                # Calculate current and power flows
                # Negative sign introduced to conform to generator convention for display
                I_ij = -(V_i - V_j) * Y_ij  # Current from 'bus_i' to 'bus_j'
                I_ji = -I_ij  # Current from 'bus_j' to 'bus_i' is the negative

                S_ij = V_i * np.conj(I_ij)  # Complex power from 'bus_i' to 'bus_j'
                S_ji = V_j * np.conj(I_ji)  # Complex power from 'bus_j' to 'bus_i'

                P_ij = S_ij.real
                Q_ij = S_ij.imag
                P_ji = S_ji.real
                Q_ji = S_ji.imag

                # Line losses (considering the line as a simple series impedance)
                P_loss = -(np.abs(I_ij)**2) * Z_ij.real
                Q_loss = -(np.abs(I_ij)**2) * Z_ij.imag

                self.branch_flows[(bus_i+1, bus_j+1)] = {
                    'P_from_to': P_ij,
                    'Q_from_to': Q_ij,
                    'P_to_from': P_ji,
                    'Q_to_from': Q_ji,
                    'P_line_loss': P_loss,
                    'Q_line_loss': Q_loss,
                    'I_line_mag': np.abs(I_ij),
                    'I_line_angle': np.angle(I_ij)
                }

        if external: return self.branch_flows
    
    def result_dataframes(self, external=False):

        generator_outputs = self.extract_generator_outputs(self)
        branch_flows = self.calculate_branch_flows(self)

        # Initialize lists for generator and load data per bus
        P_gen_list = []
        Q_gen_list = []
        P_load_list = []
        Q_load_list = []

        # Populate the lists with generator and load data
        for idx in range(self.num_buses):
            # Initialize generator power to zero
            P_gen = 0
            Q_gen = 0
            
            # If there's generator data for this bus, retrieve it
            if idx in generator_outputs:
                P_gen = generator_outputs[idx]['P'] * self.S_base  # Convert back to MW
                Q_gen = generator_outputs[idx]['Q'] * self.S_base  # Convert back to MVAr
            
            # Append generator data to the list
            P_gen_list.append(P_gen)
            Q_gen_list.append(Q_gen)
            
            # Initialize load power to zero
            P_load = 0
            Q_load = 0
            
            # Check if this bus has load data and if so, subtract to get the net load
            for load_data in self.data['load']:
                if load_data['I'] - 1 == idx:
                    P_load = load_data['PL']
                    Q_load = load_data['QL']
                    break  # Assuming one load entry per bus
            
            # Append load data to the list
            P_load_list.append(P_load)
            Q_load_list.append(Q_load)

        # Create the DataFrame
        self.bus_df = pd.DataFrame({
            'Bus': range(1, self.num_buses + 1),
            'Mag(pu)': self.V,
            'Ang(deg)': np.degrees(self.theta),
            'P_gen(MW)': P_gen_list,
            'Q_gen(MVAr)': Q_gen_list,
            'P_load(MW)': P_load_list,
            'Q_load(MVAr)': Q_load_list,
        })

        # Calculate totals
        totals = pd.DataFrame({
            'Bus': ['Total:'],
            'Mag(pu)': ['-'],
            'Ang(deg)': ['-'],
            'P_gen(MW)': [sum(P_gen_list)],
            'Q_gen(MVAr)': [sum(Q_gen_list)],
            'P_load(MW)': [sum(P_load_list)],
            'Q_load(MVAr)': [sum(Q_load_list)],
        })

        # Concatenate the totals row to the bus_df DataFrame
        self.bus_df = pd.concat([self.bus_df, totals], ignore_index=True)

        # Set the index to the Bus column
        self.bus_df.set_index('Bus', inplace=True)

        # Create an empty list to hold each row's data as a dictionary
        branch_data_list = []

        # Populate the list with branch flow data
        for key, value in branch_flows.items():
            branch_data_list.append({
                'Branch': f"{key[0]}-{key[1]}",
                'From Bus': key[0],
                'To Bus': key[1],
                'P_from_to(MW)': value['P_from_to'] * self.S_base,
                'Q_from_to(MVAr)': value['Q_from_to'] * self.S_base,
                'P_to_from(MW)': value['P_to_from'] * self.S_base,
                'Q_to_from(MVAr)': value['Q_to_from'] * self.S_base,
                'P_loss(MW)': value['P_line_loss'] * self.S_base,
                'Q_loss(MVAr)': value['Q_line_loss'] * self.S_base
            })

        # Create the DataFrame from the list of dictionaries
        self.branch_df = pd.DataFrame(branch_data_list)

        # Calculate total losses
        total_losses = self.branch_df[['P_loss(MW)', 'Q_loss(MVAr)']].sum().to_dict()
        total_losses['Branch'] = 'Total:'
        total_losses['From Bus'] = '-'
        total_losses['To Bus'] = '-'
        total_losses['P_from_to(MW)'] = '-'
        total_losses['Q_from_to(MVAr)'] = '-'
        total_losses['P_to_from(MW)'] = '-'
        total_losses['Q_to_from(MVAr)'] = '-'

        # Create a DataFrame for the total losses and concatenate it with the original branch DataFrame
        totals_df = pd.DataFrame([total_losses])
        self.branch_df = pd.concat([self.branch_df, totals_df], ignore_index=True)

        # Set the index to the Branch column
        self.branch_df.set_index('Branch', inplace=True)

        if external: return self.bus_df, self.branch_df

    # Requires result dataframes
    def initialize_graph(self):
        # Initialize directed graph
        self.G = nx.DiGraph()

        # Add nodes with attributes
        for idx, row in self.bus_df.iterrows():
            if idx != 'Total:':
                self.G.add_node(idx, 
                                voltage=row['Mag(pu)'],
                                angle=row['Ang(deg)'],
                                P_gen=row['P_gen(MW)'],
                                Q_gen=row['Q_gen(MVAr)'],
                                P_load=row['P_load(MW)'],
                                Q_load=row['Q_load(MVAr)']
                )

        # Add edges with attributes
        for idx, row in self.branch_df.iterrows():
            if idx != 'Total:':
                self.G.add_edge(row['From Bus'], row['To Bus'],
                                P_ij=row['P_from_to(MW)'],
                                Q_ij=row['Q_from_to(MVAr)'],
                                P_ji=row['P_to_from(MW)'],
                                Q_ji=row['Q_to_from(MVAr)'],
                                P_loss=row['P_loss(MW)'],
                                Q_loss=row['Q_loss(MVAr)']
                )

    def create_figure(self, V_regulation1=0.05, V_regulation2=0.10, V_ideal=1):

        self.initialize_graph()

        # Define voltage range for coloring
        self.V_regulation1 = V_regulation1
        self.V_regulation2 = V_regulation2
        self.V_ideal = V_ideal

        # Calculate bounds
        V_lower1 = V_ideal * (1 - V_regulation1)
        V_upper1 = V_ideal * (1 + V_regulation1)
        V_lower2 = V_ideal * (1 - V_regulation2)
        V_upper2 = V_ideal * (1 + V_regulation2)

        # Calculate color based on voltage
        def calculate_color(voltage):
            if V_lower2 <= voltage <= V_upper2:
                if V_lower1 <= voltage <= V_upper1:
                    # Within first regulation, green to orange
                    return 0.5 + (voltage - V_ideal) / (V_upper1 - V_ideal) * 0.25
                else:
                    # Within second regulation, orange to red
                    return 0.75 + (voltage - V_upper1) / (V_upper2 - V_upper1) * 0.25
            else:
                # Outside second regulation, red
                return 1

        def calculate_edge_color(S_actual, S_max):
            # Define RGB values for colors
            green = (0, 128, 0)  # Green
            orange = (255, 165, 0) # Orange
            red = (255, 0, 0)  # Red

            # Calculate ratio of actual to max
            ratio = S_actual / S_max if S_max != 0 else 0

            if ratio == 0:
                return 'rgb(128, 128, 128)'  # Mid-gray

            elif ratio < 1:
                # Interpolate between green and orange
                r = green[0] + (orange[0] - green[0]) * ratio
                g = green[1] + (orange[1] - green[1]) * ratio
                b = green[2] + (orange[2] - green[2]) * ratio

            elif 1 <= ratio <= 1.5:
                # Interpolate between orange and red
                excess_ratio = (ratio - 1) / 0.5  # Normalize excess ratio between 0 and 1
                r = orange[0] + (red[0] - orange[0]) * excess_ratio
                g = orange[1] + (red[1] - orange[1]) * excess_ratio
                b = orange[2] + (red[2] - orange[2]) * excess_ratio

            else:
                # Solid red for ratios above 1.5
                r, g, b = red

            return f'rgb({int(r)}, {int(g)}, {int(b)})'

        node_x = []
        node_y = []
        for node in self.G.nodes():
            x, y = self.positions[node]
            node_x.append(x)
            node_y.append(y)

        node_text = []
        node_colors = []
        apparent_powers = []
        for node, adjacencies in enumerate(self.G.adjacency()):
            node_label = adjacencies[0]  # This gets the actual node label
            node_label = adjacencies[0]
            voltage = self.G.nodes[node_label]["voltage"]
            angle = self.G.nodes[node_label]["angle"]
            P_gen = self.G.nodes[node_label]['P_gen']
            Q_gen = self.G.nodes[node_label]['Q_gen']
            P_load = self.G.nodes[node_label]['P_load']
            Q_load = self.G.nodes[node_label]['Q_load']

            # Calculate colors for nodes
            color_value = calculate_color(voltage)
            node_colors.append(color_value)

            # Calculate apparent power for nodes for scaling
            S_gen = np.sqrt(P_gen**2 + Q_gen**2)
            S_load = np.sqrt(P_load**2 + Q_load**2)
            apparent_powers.append(max(S_gen, S_load))

            node_info = (
                f'Node {node_label}:<br>'
                f'Voltage: {voltage:.2f} pu<br>'
                f'Angle: {angle:.2f} degrees<br>'
                f'P_gen: {P_gen:.2f} MW<br>'
                f'Q_gen: {Q_gen:.2f} MVAr<br>'
                f'P_load: {P_load:.2f} MW<br>'
                f'Q_load: {Q_load:.2f} MVAr'
            )
            node_text.append(node_info)

        # Constants for node sizing
        min_size = 30  # Minimum size for nodes
        max_size = 50  # Maximum size for nodes

        max_apparent_power = max(apparent_powers)

        node_labels = [node for node in self.G.nodes()]

        # Scale node sizes
        node_sizes = [min_size + (max_size - min_size) * (S / max_apparent_power) for S in apparent_powers]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',  # Show both markers and text
            text=node_labels,  # Visible labels
            textposition='middle center',  # Position of text
            hoverinfo='text',  # Show text on hover
            hovertext=node_text,  # Text to display on hover
            marker=dict(
                colorscale=[[0, 'red'], [0.25, 'orange'], [0.5, 'green'], [0.75, 'orange'], [1, 'red']],
                size=node_sizes,  # Apply node sizes
                color=node_colors,  # Node colors
                cmin=0,  # Minimum color value
                cmax=1,  # Maximum color value
                opacity=1,
                line_width=2
            ),
            textfont=dict(  # Style for visible text
                size=15,
                color='white'
            ),
            hoverlabel=dict(  # Styling for hover text
                bgcolor='white',  # Background color
                bordercolor='black',  # Border color
                font_size=14,  # Font size
            )
        )

        # Apply node sizes to the scatter plot
        node_trace.marker.size = node_sizes

        # Node trace configuration
        node_trace.marker.color = node_colors
        node_trace.marker.size = node_sizes
        node_trace.marker.colorscale = [[0, 'red'], [0.25, 'orange'], [0.5, 'green'], [0.75, 'orange'], [1, 'red']]  # Custom colorscale
        node_trace.marker.cmin = 0  # Minimum color value
        node_trace.marker.cmax = 1  # Maximum color value
        node_trace.marker.opacity = 1
        node_trace.text = node_labels

        # Create lists to hold midpoint coordinates and hover texts
        midpoint_x = []
        midpoint_y = []
        hover_text = []

        # Iterate over each edge to calculate midpoints and set hover text
        for edge in self.G.edges():
            x0, y0 = self.positions[edge[0]]
            x1, y1 = self.positions[edge[1]]
            midpoint_x.append((x0 + x1) / 2)
            midpoint_y.append((y0 + y1) / 2)

            # Retrieve edge attributes for hover text
            edge_data = self.G.edges[edge]
            hover_info = (
                f'Edge {edge[0]}-{edge[1]}: <br>'
                f'P_ij={edge_data["P_ij"]:.2f} MW, <br>'
                f'Q_ij={edge_data["Q_ij"]:.2f} MVAr, <br>'
                f'P_ji={edge_data["P_ji"]:.2f} MW, <br>'
                f'Q_ji={edge_data["Q_ji"]:.2f} MVAr, <br>'
                f'P_loss={edge_data["P_loss"]:.2f} MW, <br>'
                f'Q_loss={edge_data["Q_loss"]:.2f} MVAr <br>'
            )
            hover_text.append(hover_info)

        # Parameters for edge width scaling
        min_width = 3  # Minimum edge width
        max_width = 12  # Maximum edge width

        # Calculate apparent power and normalize it for edge width
        apparent_power = []
        for edge in self.G.edges():
            edge_data = self.G.edges[edge]
            S_ij = np.sqrt(edge_data["P_ij"]**2 + edge_data["Q_ij"]**2)
            apparent_power.append(S_ij)

        # Normalize the apparent power values to the range [min_width, max_width]
        max_power = max(apparent_power)
        min_power = min(apparent_power)
        normalized_widths = [min_width + (max_width - min_width) * (S - min_power) / (max_power - min_power) for S in apparent_power]

        # Update the edge trace for the plot
        edge_x = []
        edge_y = []
        edge_traces = []
        edge_colors = []

        # Iterate over edges and set color based on power rating
        for edge in self.G.edges():
            branch_key = (edge[0], edge[1])
            
            # Retrieve data for S_actual calculation
            row = self.branch_df.loc[(self.branch_df['From Bus'] == branch_key[0]) & (self.branch_df['To Bus'] == branch_key[1])]
            if not row.empty:
                S_actual = np.sqrt(row['P_from_to(MW)'].values[0]**2 + row['Q_from_to(MVAr)'].values[0]**2)

                # Determine if the edge represents a branch or a transformer
                if any((branch['I'] == branch_key[0] and branch['J'] == branch_key[1]) or 
                    (branch['I'] == branch_key[1] and branch['J'] == branch_key[0]) for branch in self.data['branch']):
                    S_max = next((item['RATEA'] for item in self.data['branch'] if 
                            (item['I'] == branch_key[0] and item['J'] == branch_key[1]) or 
                            (item['I'] == branch_key[1] and item['J'] == branch_key[0])), 0)
                elif any((transformer['I'] == branch_key[0] and transformer['J'] == branch_key[1]) or 
                        (transformer['I'] == branch_key[1] and transformer['J'] == branch_key[0]) for transformer in self.data['transformer2']):
                    S_max = next((item['RATA1'] for item in self.data['transformer2'] if 
                            (item['I'] == branch_key[0] and item['J'] == branch_key[1]) or 
                            (item['I'] == branch_key[1] and item['J'] == branch_key[0])), 0)
                else:
                    edge_color = calculate_edge_color(1, 0) # Will return gray
                    edge_colors.append(edge_color)

                # Calculate color for the edge
                edge_color = calculate_edge_color(S_actual, S_max)
                edge_colors.append(edge_color)
            else:
                edge_color = calculate_edge_color(1, 0) # Will return gray
                edge_colors.append(edge_color)

        for i, edge in enumerate(self.G.edges()):
            x0, y0 = self.positions[edge[0]]
            x1, y1 = self.positions[edge[1]]
            width = normalized_widths[i]

            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=width, color=edge_colors[i]),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        # Create a scatter trace for edge midpoints
        edge_midpoint_trace = go.Scatter(
            x=midpoint_x, y=midpoint_y,
            mode='markers',
            hoverinfo='text',
            text=hover_text,
            marker=dict(
                size=5,
                color='#FFFFFF',  # White color or use 'rgba(0,0,0,0)' for invisible
                opacity=0.5  # Adjust opacity as needed
            )
        )

        edge_color_scale = [[0, 'rgb(0, 128, 0)'], [0.5, 'rgb(255, 165, 0)'], [1, 'rgb(255, 0, 0)']]  # Green to Orange to Red

        dummy_edge_trace = go.Scatter(
            x=[None], 
            y=[None], 
            mode='markers',
            marker=dict(
                colorscale=edge_color_scale,
                cmin=0,
                cmax=1.5,
                colorbar = dict(
                    title='Edge Color',
                    x = 1,  # Adjust this value to position next to the node color bar
                    len = 0.4,
                    yanchor = 'top'
                ),
                color=[0]  # Dummy color value
            ),
            hoverinfo='none'
        )

        dummy_node_trace = go.Scatter(
            x=[None], 
            y=[None], 
            mode='markers',
            marker=dict(
                colorscale=[
                    [0, 'red'],
                    [(V_lower2 - V_lower2) / (V_upper2 - V_lower2), 'red'],  # Lower red limit (0)
                    [(V_lower1 - V_lower2) / (V_upper2 - V_lower2), 'orange'],  # Lower orange limit
                    [(V_ideal - V_lower2) / (V_upper2 - V_lower2), 'green'],  # Ideal voltage (0.5 if V_ideal is midway)
                    [(V_upper1 - V_lower2) / (V_upper2 - V_lower2), 'orange'],  # Upper orange limit
                    [1, 'red']  # Upper red limit (1)
                ],
                cmin=V_lower2,  # Minimum value of the color scale
                cmax=V_upper2,  # Maximum value of the color scale
                colorbar = dict(
                    title='Node Color',
                    x = 1,  # Position to the right of the plot area
                    len = 0.4,  # Length of the color bar (40% of the plot height)
                    yanchor = 'bottom'
                ),
                color=[V_ideal]  # Dummy color value, ideally the middle value
            ),
            hoverinfo='none'
        )

        self.fig = go.Figure(layout=go.Layout(
                            title='Power Network Flow Visualization',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Power flow network",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        self.fig.add_trace(dummy_edge_trace)
        self.fig.add_trace(dummy_node_trace)

        self.fig.add_trace(edge_midpoint_trace)

        # Add all edge traces to the figure
        for trace in edge_traces:
            self.fig.add_trace(trace)

        self.fig.add_trace(node_trace)

    def remove_branch(self, bus_ID_1, bus_ID_2):
        # Find the index of the line to be removed
        line_index = self.get_line_index(bus_ID_1, bus_ID_2)
        if line_index is None:
            print("Line not found")
            return

        # Remove the corresponding entries from A and Y_br matrices
        self.A = np.delete(self.A, line_index, axis=1)
        self.Y_br = np.delete(self.Y_br, line_index, axis=0)
        self.Y_br = np.delete(self.Y_br, line_index, axis=1)

        # Recalculate the Y_bus matrix
        self.Y_bus = self.A @ self.Y_br @ self.A.T

        # Remove the line from self.data
        if line_index < len(self.data['branch']):
            del self.data['branch'][line_index]
        else:
            # Adjust index for transformer data
            transformer_index = line_index - len(self.data['branch'])
            del self.data['transformer2'][transformer_index]

    def remove_load(self, bus_ID):
        # Find and remove the load at the specified bus
        for i, load_data in enumerate(self.data['load']):
            if load_data['I'] == bus_ID:
                self.P[bus_ID-1] += load_data['PL'] / self.S_base  # Revert the load in power flow
                self.Q[bus_ID-1] += load_data['QL'] / self.S_base  # Revert the load in power flow
                del self.data['load'][i]  # Remove the load from data
                break

# If removing slack bus, assign next largest as slack
    def remove_generator(self, bus_ID):
        # Find and remove the generator at the specified bus
        for i, gen_data in enumerate(self.data['generator']):
            if gen_data['I'] == bus_ID:
                self.P[bus_ID-1] -= gen_data['PG'] / self.S_base  # Revert the generator in power flow
                self.Q[bus_ID-1] -= gen_data['QG'] / self.S_base  # Revert the generator in power flow
                del self.data['generator'][i]  # Remove the generator from data
                break

    def get_line_index(self, bus_ID_1, bus_ID_2):
        # Check in branch data
        for idx, branch in enumerate(self.data['branch']):
            if (branch['I'] == bus_ID_1 and branch['J'] == bus_ID_2) or \
            (branch['I'] == bus_ID_2 and branch['J'] == bus_ID_1):
                return idx  # Index in the branch section

        # Check in transformer data, adjusting index to continue after branch data
        transformer_start_idx = len(self.data['branch'])
        for idx, transformer in enumerate(self.data['transformer2']):
            if (transformer['I'] == bus_ID_1 and transformer['J'] == bus_ID_2) or \
            (transformer['I'] == bus_ID_2 and transformer['J'] == bus_ID_1):
                return transformer_start_idx + idx  # Adjusted index in the transformer section

        return None

    def N_1_contingency_screening(self):
        
        # Still need to do this
        self.critical_contingenices = {}

        def try_to_solve(entity, entity_type):
            try:
                # Run power flow analysis
                self.newton_raphson_power_flow()
                self.result_dataframes()
                self.create_figure()

                # Store successful power flow results
                description = f"{entity_type} removal: {entity}"
                self.results[description] = {
                    'bus_df': self.bus_df,
                    'branch_df': self.branch_df,
                    'graph': self.G,
                    'figure': self.fig,
                    'solver log': self.log
                }
            except:
                # Store the failure result if power flow does not converge
                description = f"{entity_type} removal: {entity}"
                self.results[description] = "Failed to converge."

            # Reset the network to its original state
            self.reset(reset_results=False)

        # Iterate over branches for N-1 contingency
        for branch in self.data['branch']:
            branch_description = f"Line {branch['I']}-{branch['J']}"
            self.remove_branch(branch['I'], branch['J'])
            try_to_solve(branch_description, "Branch")

        # Iterate over transformers for N-1 contingency
        for transformer in self.data['transformer2']:
            transformer_description = f"Transformer {transformer['I']}-{transformer['J']}"
            self.remove_branch(transformer['I'], transformer['J'])
            try_to_solve(transformer_description, "Branch")

        # Iterate over loads for N-1 contingency
        for load in self.data['load']:
            load_description = f"Load at Bus {load['I']}"
            self.remove_load(load['I'])
            try_to_solve(load_description, "Load")

        # Iterate over generators for N-1 contingency
        for generator in self.data['generator']:
            generator_description = f"Generator at Bus {generator['I']}"
            self.remove_generator(generator['I'])
            try_to_solve(generator_description, "Generator")

    # Requires that extract_generator_outputs() be called first
    def update_network_from_machine(self, P_machine, Q_machine, V_machine, theta_machine, bus_idx_machine, bus_type_machine):
        # Solve based on bus type
        if bus_type_machine == 1:  # PQ bus, know V, theta
            for bus_idx, gen_output in self.generator_outputs.items():
                if bus_idx == bus_idx_machine:
                    # Subtract the existing output from the machine
                    self.P[bus_idx] -= gen_output['P']
                    self.Q[bus_idx] -= gen_output['Q']
                    # Add in the new output from the machine
                    self.P[bus_idx] += P_machine
                    self.Q[bus_idx] += Q_machine
        elif bus_type_machine == 2:  # PV bus, know theta, Q
            for bus_idx, gen_output in self.generator_outputs.items():
                if bus_idx == bus_idx_machine:
                    # Subtract the existing output from the machine
                    self.P[bus_idx] -= gen_output['P']
                    # Add in the new output from the machine
                    self.P[bus_idx] += P_machine
            self.V[bus_idx_machine] = V_machine
        elif bus_type_machine == 3:  # Slack bus, know P, Q
            self.V[bus_idx_machine] = V_machine
            self.theta[bus_idx_machine] = theta_machine

#---- Parse & Initialize ----#

def data_grabber(filepath):
    # Get data
    data = PSSE33_raw_parser(filepath)
    # Build network matrices
    A, Y_br, Y_bus = network_matrices(data) # Improved

    # System base MVA
    S_base = data['case'][0]['SBASE']

    # Number of buses
    num_buses = len(data['bus'])

    # Initialize arrays to store specified power injections, voltage magnitudes, and bus types
    P = np.zeros(num_buses)
    Q = np.zeros(num_buses)
    V = np.zeros(num_buses)
    theta = np.zeros(num_buses)
    bus_types = np.zeros(num_buses, dtype=int)

    # Extract information from parsed bus data
    for bus_data in data['bus']:
        idx = bus_data['I'] - 1  # Convert 1-based to 0-based indexing
        V[idx] = bus_data['VM']
        theta[idx] = bus_data['VA'] * (np.pi / 180)  # Convert from degrees to radians
        bus_types[idx] = bus_data['IDE']

    # Handle generator data (considering it as negative load)
    for gen_data in data['generator']:
        bus_idx = gen_data['I'] - 1
        P[bus_idx] += gen_data['PG'] / S_base  # Convert from MW to PU
        Q[bus_idx] += gen_data['QG'] / S_base  # Convert from MVAR to PU

    # Handle load data
    for load_data in data['load']:
        bus_idx = load_data['I'] - 1
        P[bus_idx] -= load_data['PL'] / S_base  # Convert from MW to PU
        Q[bus_idx] -= load_data['QL'] / S_base  # Convert from MVAR to PU

    return P, Q, V, theta, bus_types, num_buses, A, Y_br, Y_bus, S_base, data

def network_matrices(data):
    num_buses = len(data['bus'])
    num_branch = len(data['branch']) + len(data['transformer2'])

    # Initialize matrices
    A = np.zeros((num_buses, num_branch), dtype=int)
    Y_br = np.zeros((num_branch, num_branch), dtype=complex)

    # Process branches (lines)
    for idx, branch in enumerate(data['branch']):
        from_bus = branch['I'] - 1
        to_bus = branch['J'] - 1

        # Create incidence matrix
        A[from_bus, idx] = -1
        A[to_bus, idx] = 1

        # Calculate line admittance
        R = branch['R']
        X = branch['X']
        B = branch['B'] / 2  # Half line charging at each end
        Z = complex(R, X)
        Y_line = 1 / Z
        Y_shunt = complex(0, B)  # Shunt admittance due to line charging

        # Admittance matrix for this branch
        Y_br[idx, idx] = Y_line + Y_shunt

    # Process two-winding transformers
    for idx, transformer in enumerate(data['transformer2'], start=len(data['branch'])):
        from_bus = transformer['I'] - 1
        to_bus = transformer['J'] - 1

        A[from_bus, idx] = -1
        A[to_bus, idx] = 1

        R = transformer['R1-2']
        X = transformer['X1-2']
        Z = complex(R, X)

        # Handle transformer tap ratios and phase shifts
        tap = transformer['WINDV1']
        if tap == 0:
            tap = 1.0  # Avoid division by zero
        angle = np.radians(transformer['ANG1'])  # Convert angle to radians
        angle_complex = np.exp(1j * angle)  # Complex exponential for phase shift
        Y_transformer = 1 / (Z * tap**2) * angle_complex  # Adjust admittance for tap and phase shift

        Y_br[idx, idx] = Y_transformer

    # Calculate the nodal admittance matrix
    Y_bus = A @ Y_br @ A.T

    return A, Y_br, Y_bus

# Pulls data from PSS/E 33 .raw format to a dictionary of lists of dictionaries per component type
def PSSE33_raw_parser(filepath):

    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Create a dictionary of data parsed from the .raw file
    data_sections = {
        'case': [],
        'bus': [],
        'load': [],
        'shunt': [],
        'generator': [],
        'branch': [],
        'transformer2': [],
        'area': []
    }

    section_starts = {
        "BEGIN LOAD DATA": 'load',
        "BEGIN FIXED SHUNT DATA": 'shunt',
        "BEGIN GENERATOR DATA": 'generator',
        "BEGIN BRANCH DATA": 'branch',
        "BEGIN TRANSFORMER DATA": 'transformer2',
        "BEGIN AREA DATA": 'area'
    }

    active_section = 'case'

    for idx, line in enumerate(lines):
        if idx == 2:
            active_section = 'bus'
            continue
        else:
            # Check if line starts a new section
            found_section_start = False  # A flag to track if we found a new section start
            for start_line, section in section_starts.items():
                if start_line in line:
                    active_section = section
                    if active_section == 'transformer2':
                        transformer_header_idx = idx
                    found_section_start = True
                    break  # exit the inner loop since we found a match
            if found_section_start:
                continue  # skip the rest of the loop for this iteration

        if active_section == 'case' and idx == 0:
            # Split the line at '/'
            data_part = line.split('/')[0].strip()
            # Split the data part at ','
            values = [val.strip() for val in data_part.split(',')]
            data_sections[active_section].append({
                'IC': int(values[0].strip()),
                'SBASE': float(values[1].strip()),
                'REV': int(values[2].strip()),
                'XFRRAT': float(values[3].strip()),
                'NXFRAT': float(values[4].strip()),
                'BASFRQ': float(values[5].strip()),
            })

        elif active_section == 'bus':
            values = line.split(',')
            data_sections[active_section].append({
                'I': int(values[0].strip()),
                'NAME': values[1].strip().replace("'", "").strip(),
                'BASKV': float(values[2].strip()),
                'IDE': int(values[3].strip()),
                'AREA': int(values[4].strip()),
                'ZONE': int(values[5].strip()),
                'OWNER': int(values[6].strip()),
                'VM': float(values[7].strip()),
                'VA': float(values[8].strip())
            })
        
        elif active_section == 'load':
            values = line.split(',')
            data_sections[active_section].append({
                'I': int(values[0].strip()),
                'ID': values[1].strip().replace("'", "").strip(),
                'STATUS': int(values[2].strip()),
                'AREA': int(values[3].strip()),
                'ZONE': int(values[4].strip()),
                'PL': float(values[5].strip()),
                'QL': float(values[6].strip()),
                'IP': float(values[7].strip()),
                'IQ': float(values[8].strip()),
                'YP': float(values[9].strip()),
                'YQ': float(values[10].strip()),
            })

        elif active_section == 'shunt':
            values = line.split(',')
            data_sections[active_section].append({
                'I': int(values[0].strip()),
                'ID': values[1].strip().replace("'", "").strip(),
                'STATUS': int(values[2].strip()),
                'GL': float(values[3].strip()),
                'BL': float(values[4].strip())
            })

        elif active_section == 'generator':
            values = line.split(',')
            data_sections[active_section].append({
                'I': int(values[0].strip()),
                'ID': values[1].strip().replace("'", "").strip(),
                'PG': float(values[2].strip()),
                'QG': float(values[3].strip()),
                'QT': float(values[4].strip()),
                'QB': float(values[5].strip()),
                'VS': float(values[6].strip()),
                'IREG': float(values[7].strip()),
                'MBASE': float(values[8].strip()),
                'ZR': float(values[9].strip()),
                'ZX': float(values[10].strip()),
                'RT': float(values[11].strip()),
                'XT': float(values[12].strip()),
                'GTAP': float(values[13].strip()),
                'STAT': float(values[14].strip()),
                'RMPCT': float(values[15].strip()),
                'PT': float(values[16].strip()),
                'PB': float(values[17].strip()),
                'Oi': float(values[18].strip()),
                'Fi': float(values[19].strip()),
                'WMOD': float(values[20].strip())
            })

        elif active_section == 'branch':
            values = line.split(',')
            data_sections[active_section].append({
                'I': int(values[0].strip()),
                'J': int(values[1].strip()),
                'CKT': values[2].strip().replace("'", "").strip(),
                'R': float(values[3].strip()),
                'X': float(values[4].strip()),
                'B': float(values[5].strip()),
                'RATEA': float(values[6].strip()),
                'RATEB': float(values[7].strip()),
                'RATEC': float(values[8].strip()),
                'GI': float(values[9].strip()),
                'BI': float(values[10].strip()),
                'GJ': float(values[11].strip()),
                'BJ': float(values[12].strip()),
                'ST': float(values[13].strip()),
                'MET': float(values[14].strip()),
                'LEN': float(values[15].strip()),
                'Oi': float(values[16].strip()),
                'Fi': float(values[17].strip())
            })

        elif active_section == 'transformer2':
            if (idx - transformer_header_idx - 1) % 4 == 0:  # Check if this is the start of a new generator chunk
                # Process the next three lines for the current generator
                line1_values = line.split(',')
                line2_values = lines[idx + 1].split(',')
                line3_values = lines[idx + 2].split(',')
                line4_values = lines[idx + 3].split(',')

                data_sections[active_section].append({
                    # Line 1
                    'I': int(line1_values[0].strip()),
                    'J': int(line1_values[1].strip()),
                    'K': int(line1_values[2].strip()),
                    'CKT': line1_values[3].strip().replace("'", "").strip(),
                    'CW': int(line1_values[4].strip()),
                    'CZ': int(line1_values[5].strip()),
                    'CM': int(line1_values[6].strip()),
                    'MAG1': float(line1_values[7].strip()),
                    'MAG2': float(line1_values[8].strip()),
                    'NMETR': int(line1_values[9].strip()),
                    'NAME': line1_values[10].strip().replace("'", "").strip(),
                    'STAT': int(line1_values[11].strip()),
                    'O1': int(line1_values[12].strip()),
                    'F1': float(line1_values[13].strip()),
                    'O2': int(line1_values[14].strip()),
                    'F2': float(line1_values[15].strip()),
                    'O3': int(line1_values[16].strip()),
                    'F3': float(line1_values[17].strip()),
                    'O4': int(line1_values[18].strip()),
                    'F4': float(line1_values[19].strip()),
                    # Line 2
                    'R1-2': float(line2_values[0].strip()),
                    'X1-2': float(line2_values[1].strip()),
                    'SBASE1-2': float(line2_values[2].strip()),
                    # Line 3
                    'WINDV1': float(line3_values[0].strip()),
                    'NOMV1': float(line3_values[1].strip()),
                    'ANG1': float(line3_values[2].strip()),
                    'RATA1': float(line3_values[3].strip()),
                    'RATB1': float(line3_values[4].strip()),
                    'RATC1': float(line3_values[5].strip()),
                    'COD1': int(line3_values[6].strip()),
                    'CONT1': int(line3_values[7].strip()),
                    'RMA1': float(line3_values[8].strip()),
                    'RMI1': float(line3_values[9].strip()),
                    'VMA1': float(line3_values[10].strip()),
                    'VMI1': float(line3_values[11].strip()),
                    'NTP1': float(line3_values[12].strip()),
                    'TAB1': float(line3_values[13].strip()),
                    'CR1': float(line3_values[14].strip()),
                    'CX1': float(line3_values[15].strip()),
                    # Line 4
                    'WINDV2': float(line4_values[0].strip()),
                    'NOMV2': float(line4_values[1].strip())
                })

    return data_sections

# Converts data dictionary to matpower format and saves to a .m file (used for validation efforts against commerical software)
def convert_to_matpower(data, filename, bus_vlim=[0.95,1.05]):
    """
    Convert PSSE data to Matpower format.
    """
    matpower_data = {
        'version': '2',
        'baseMVA': data['case'][0]['SBASE'],
    }
    
    # Convert bus data
    bus_dict = {bus['I']: bus for bus in data['bus']}
    bus_list = []
    for bus in data['bus']:
        bus_type = bus['IDE']
        # Matpower bus types: 1 = PQ, 2 = PV, 3 = ref, 4 = isolated
        # PSSE bus types: 1 = load bus (PQ), 2 = generator bus (PV), 3 = swing bus (ref)
        bus_type_mp = bus_type if bus_type != 4 else 1  # Isolated bus treated as PQ bus in Matpower

        Pd = 0; Qd = 0
        # Add load data to bus
        for load in data['load']:
            if load['I'] == bus['I']:
                Pd = load['PL']
                Qd = load['QL']

        bus_list.append([
            int(bus['I']),          # Bus number
            int(bus_type_mp),       # Bus type
            Pd, Qd, 0, 0,           # Load and shunt data
            int(bus['AREA']),       # Area number
            bus['VM'],              # Voltage magnitude
            bus['VA'],              # Voltage angle
            bus.get('BASKV', 0),    # Base voltage
            1,                      # Zone (default to 1)
            bus_vlim[1],            # Maximum voltage magnitude
            bus_vlim[0]             # Minimum voltage magnitude
        ])
    matpower_data['bus'] = np.array(bus_list)
    
    # Convert branch and transformer data
    branch_list = []
    for branch in data['branch']:
        branch_list.append([
            int(branch['I']),     # From bus
            int(branch['J']),     # To bus
            branch['R'],          # Resistance
            branch['X'],          # Reactance
            branch['B'],          # Total line charging susceptance
            branch['RATEA'],      # MVA rating A
            branch['RATEB'],      # MVA rating B
            branch['RATEC'],      # MVA rating C
            0,                    # Transformer off nominal turns ratio
            0,                    # Transformer phase shift angle
            1,                    # Branch status
            -360,                 # Minimum angle difference
            360                   # Maximum angle difference
        ])
    for transformer in data['transformer2']:
        branch_list.append([
            int(transformer['I']), # From bus
            int(transformer['J']), # To bus
            transformer['R1-2'],   # Resistance
            transformer['X1-2'],   # Reactance
            0,                     # Total line charging susceptance
            transformer['RATA1'], # MVA rating A
            transformer['RATB1'], # MVA rating B
            transformer['RATC1'], # MVA rating C
            transformer['WINDV1'], # Transformer off nominal turns ratio
            transformer['ANG1'],   # Transformer phase shift angle
            1,                     # Branch status
            -360,                  # Minimum angle difference
            360                    # Maximum angle difference
        ])

    matpower_data['branch'] = np.array(branch_list)
    
    # Convert generator data
    gen_list = []
    for gen in data['generator']:
        gen_bus = bus_dict[gen['I']]
        gen_type = 2 if gen_bus['IDE'] == 3 else 3  # 2: PV bus, 3: PQ bus
        gen_list.append([
            int(gen['I']),       # Generator bus number
            gen['PG'],           # Real power output
            gen['QG'],           # Reactive power output
            gen['QT'],           # Maximum reactive power output
            gen['QB'],           # Minimum reactive power output
            gen['VS'],           # Voltage magnitude setpoint
            gen['MBASE'],        # MVA base of the machine
            1,                   # Generator status
            gen['PT'],           # Maximum real power output
            gen['PB'],           # Minimum real power output
            gen_type,            # Generator type
            0,                   # Qc1min
            0,                   # Qc1max
            0,                   # Qc2min
            0,                   # Qc2max
            0,                   # ramp_agc
            0,                   # ramp_10
            0,                   # ramp_30
            0,                   # ramp_q
            0                    # apf
        ])
    matpower_data['gen'] = np.array(gen_list)
    
    """
    Save Matpower data to a .m file.
    """
    with open(filename, 'w') as f:
        # Write Matpower version
        f.write(f"function mpc = casefile\n")
        f.write(f"%CASENAME Power flow data for case.\n")
        f.write(f"%   MATPOWER\n\n")
        
        # Write function signature
        f.write("mpc.version = '2';\n")
        f.write(f"mpc.baseMVA = {matpower_data['baseMVA']};\n\n")
        
        # Write bus data
        f.write("%% bus data\n")
        f.write("%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin\n")
        f.write("mpc.bus = [\n")
        np.savetxt(f, matpower_data['bus'], fmt='%.6f')
        f.write("];\n\n")
        
        # Write branch data
        f.write("%% branch data\n")
        f.write("%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax\n")
        f.write("mpc.branch = [\n")
        np.savetxt(f, matpower_data['branch'], fmt='%.6f')
        f.write("];\n\n")
        
        # Write generator data
        f.write("%% gen data\n")
        f.write("%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin\n")
        f.write("mpc.gen = [\n")
        np.savetxt(f, matpower_data['gen'], fmt='%.6f')
        f.write("];\n\n")
        
        f.write("end\n")
    
    print('Successfully reformatted datafile for MATPOWER and saved.')

#---- Support Functions ----#

# Prints out matrices nicely
def print_matrix(matrix, format_spec='>10.2f'):
    for row in matrix:
        formatted_row = [format(x, format_spec) for x in row]
        print(' '.join(formatted_row))

# Creates spy plot of admittance matrix
def spy_admittance(Y, width=8, height=5, markersize=5):

    plt.figure(figsize=(width, height))
    plt.spy(Y, markersize=markersize)
    plt.title("Spy plot of Admittance Matrix (Y)")
    plt.xlabel("Bus")
    plt.ylabel("Bus")

    ax = plt.gca()  # Get the current Axes instance

    # Get the current locations of the ticks
    locs = ax.get_xticks()
    locs_y = ax.get_yticks()

    # Set the tick locations
    ax.set_xticks(locs[1:-1])  # This skips the first and last tick location
    ax.set_yticks(locs_y[1:-1])  # This skips the first and last tick location

    # Set the tick labels to be one more than the tick locations
    ax.set_xticklabels([int(loc) + 1 for loc in locs[1:-1]])
    ax.set_yticklabels([int(loc) + 1 for loc in locs_y[1:-1]])

    plt.show()


