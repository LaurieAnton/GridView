import plotly.graph_objects as go
import matplotlib.pyplot as plt
import powerflow as pf
import numpy as np
import inspect
import pickle
import time

class sync_machine_Order_III_with_PID_controllers:
    def __init__(self, bus_ID, bus_type=0, D=1, M=5,
                 p_m = 1, p_m_ref = 1, K_p_g = 0.03, K_i_g = 100, K_d_g=0, integral_error_omega = 0,
                 v_f_0 = 1, v_ref=1, K_p_V = 1e-3, K_i_V = 1, K_d_V=0, integral_error_V = 0,
                 omega_s=2*np.pi*60, r_a=1e-3, x_d=1.8, x_d_prime=0.5, T_d0_prime=6,
                 omega_0 = 1, delta_0 = 0, eqprime_0=0):
        # Network reference frame
        self.bus_ID = bus_ID
        self.bus_type = bus_type
        self.P = 0
        self.Q = 0
        self.V = 0
        self.theta = 0

        # Machine reference frame
        self.v_d = 0
        self.v_q = 0
        self.i_d = 0
        self.i_q = 0

        # Mechanical machine parameters
        self.D = D
        self.M = M
        self.p_m_0 = p_m
        self.p_m = p_m
        self.omega_s = omega_s

        # Governor control parameters
        self.p_m_ref = p_m_ref
        self.K_p_g = K_p_g
        self.K_i_g = K_i_g
        self.K_d_g = K_d_g
        self.integral_error_omega = integral_error_omega
        self.previous_error_omega = 0

        # Electrical machine parameters
        self.r_a = r_a
        self.x_d = x_d
        self.x_d_prime = x_d_prime
        self.T_d0_prime = T_d0_prime

        # Electrical control parameters
        self.v_f_0 = v_f_0
        self.v_f = v_f_0
        self.v_ref = v_ref
        self.K_p_V = K_p_V
        self.K_i_V = K_i_V
        self.K_d_V = K_d_V
        self.integral_error_V = integral_error_V
        self.previous_error_V = 0

        # States and derivatives
        self.delta_0 = delta_0       # Copy of IC for reset
        self.omega_0 = omega_0       # Copy of IC for reset
        self.eqprime_0 = eqprime_0   # Copy of IC for reset
        self.delta = delta_0
        self.omega = omega_0
        self.eqprime = eqprime_0
        self.ddelta_dt = 0
        self.domega_dt = 0
        self.deqprime_dt = 0

        # Time domain history
        self.t = 0
        self.dt = 0                  
        self.t_hist = []
        self.delta_hist = [self.delta_0]
        self.omega_hist = [self.omega_0]
        self.eqprime_hist = [self.eqprime_0]

        # Z matrix
        # Terminal currents in machine reference frame
        # Had to change the sign on x_d_prime quantities
        self.A = np.array([[self.x_d_prime, -self.r_a],
                      [-self.r_a, -self.x_d_prime]])
        self.Ainv = np.linalg.solve(self.A, np.eye(2))

    def reset(self):
        # Reset states to initial conditions
        self.delta = self.delta_0
        self.omega = self.omega_0
        self.eqprime = self.eqprime_0
        self.ddelta_dt = 0          # Redundant
        self.domega_dt = 0          # Redundant
        self.deqprime_dt = 0        # Redundant

        # Reset control inputs to initial conditions
        self.v_f = self.v_f_0
        self.p_m = self.p_m_0

        # Reset histories
        self.t_hist = []
        self.delta_hist = [self.delta_0]
        self.omega_hist = [self.omega_0]
        self.eqprime_hist = [self.eqprime_0]

        # Re-initialize other inputs
        self.P = 0
        self.Q = 0
        self.V = 0
        self.theta = 0

        # Machine reference frame
        self.v_d = 0
        self.v_q = 0
        self.i_d = 0
        self.i_q = 0

    def differential_equations(self, states=None):

        # Allows implicit solves
        if states is None:
            delta=self.delta
            omega=self.omega
            eqprime=self.eqprime
        else:
            delta = states[0]
            omega = states[1]
            eqprime = states[2]

        '''
        # PID Governor control
        speed_error = self.omega - 1
        self.integral_error_omega += speed_error * self.dt

        if self.dt == 0: derivative_error_omega = 0
        else: derivative_error_omega = (speed_error - self.previous_error_omega) / self.dt

        self.p_m = (self.p_m_ref - self.K_p_g * speed_error -
                    self.K_i_g * self.integral_error_omega -
                    self.K_d_g * derivative_error_omega)

        self.previous_error_omega = speed_error
        '''
        '''
        # PID Field excitation control
        voltage_error = self.v_f - self.v_ref
        self.integral_error_V += voltage_error * self.dt

        if self.dt == 0: derivative_error_V = 0
        else: derivative_error_V = (voltage_error - self.previous_error_V) / self.dt

        self.v_f = (self.K_p_V * voltage_error -
                    self.K_i_V * self.integral_error_V -
                    self.K_d_V * derivative_error_V)

        self.previous_error_V = voltage_error'''

        # Terminal components in network reference frame
        self.V_D = self.V*np.cos(self.theta)
        self.V_Q = self.V*np.sin(self.theta)
        self.I_D = (self.P*self.V_D + self.Q*self.V_Q)/(self.V_D**2 + self.V_Q**2)
        self.I_Q = (self.P*self.V_Q - self.Q*self.V_D)/(self.V_D**2 + self.V_Q**2)

        # Terminal voltage components in machine reference frame
        self.v_d = self.V_D*np.sin(delta) - self.V_Q*np.cos(delta)
        self.v_q = self.V_D*np.cos(delta) + self.V_Q*np.sin(delta)
        self.i_d = self.I_D*np.sin(delta) - self.I_Q*np.cos(delta)
        self.i_q = self.I_D*np.cos(delta) + self.I_Q*np.sin(delta)

        # Calculate electrical power
        self.p_e = eqprime*self.i_q
        
        # For tuning
        if self.t == 0:
            E_dprime = self.eqprime*np.cos(self.delta)
            E_qprime = self.eqprime*np.sin(self.delta)
            E_prime = complex(E_dprime, E_qprime)
            I = complex(self.I_D, self.I_Q)

            self.r_a = np.real((E_prime - complex(self.V_D, self.V_Q)) / I)
            self.x_d = np.imag((E_prime - complex(self.V_D, self.V_Q)) / I)

        '''print(f'Machine {self.bus_ID} ra, xd: ', ra, xd)
        print('Inputs eqp, delta, ID, IQ, VD, VQ', self.eqprime, self.delta, self.I_D, self.I_Q,
              self.V_D, self.V_Q)
        print('Real', self.V, self.theta, self.P, self.Q)

        E_dprime = self.eqprime*np.cos(self.delta)
        E_qprime = self.eqprime*np.sin(self.delta)
        E_prime = complex(E_dprime, E_qprime)
        I = complex(self.I_D, self.I_Q)

        V = E_prime - (ra + 1j * xd) * I
        V_mag = np.abs(V)
        theta = np.angle(V)
        V_D = V_mag*np.cos(theta)
        V_Q = V_mag*np.sin(theta)
        
        P = V_D*np.real(I) + V_Q*np.imag(I)
        Q = V_Q*np.real(I) - V_D*np.imag(I)

        print('Inputs eqp, delta, ID, IQ', self.eqprime, self.delta, self.I_D, self.I_Q)
        print('Recovered', V_mag, theta, P, Q, V_D, V_Q, '\n')'''
        
        # Differential equations
        self.ddelta_dt = self.omega_s*(omega - 1) #self.omega_s * (omega - 1)
        self.domega_dt = (self.p_m - self.p_e - self.D*(omega - 1)) / self.M
        self.deqprime_dt = (-eqprime -(self.x_d - self.x_d_prime)*self.i_d + self.v_f)/self.T_d0_prime

        return self.ddelta_dt, self.domega_dt, self.deqprime_dt

    def update_network_quantities(self):
        E_dprime = self.eqprime*np.cos(self.delta)
        E_qprime = self.eqprime*np.sin(self.delta)
        E_prime = complex(E_dprime, E_qprime)
        I = complex(self.I_D, self.I_Q)

        V = E_prime - (self.r_a + 1j * self.x_d) * I
        self.V = np.abs(V)
        self.theta = np.angle(V)
        V_D = self.V*np.cos(self.theta)
        V_Q = self.V*np.sin(self.theta)
        
        self.P = V_D*np.real(I) + V_Q*np.imag(I)
        self.Q = V_Q*np.real(I) - V_D*np.imag(I)

        #print('Inputs eqp, delta, ID, IQ', self.eqprime, self.delta, self.I_D, self.I_Q)
        #print(f'M{self.bus_ID} Update', V_mag, theta, P, Q, V_D, V_Q,)
        #print('Real', self.V, self.theta, self.P, self.Q, '\n')
        
        return self.P, self.Q, self.V, self.theta

def PS_forward_euler(network, machines=None, t0=0, tf=10, dt=1e-3, reset=False):
    '''
    Must take a network. Machines are optional. Will use the Richardson approximation for the network.
    '''
    # Helper function
    states_per_machine = 3
    def combined_differential_equations(x):
        # Split x back into individual machine states
        derivatives = []

        for i, machine in enumerate(machines):
            state_start = i * states_per_machine
            state_end = state_start + states_per_machine
            machine_state = x[state_start:state_end]

            # Compute the derivatives for this machine
            machine_derivatives = machine.differential_equations(states=machine_state)
            derivatives.append(machine_derivatives)

        # Include the Richardson states
        derivatives.append(network.powerflow_decoupled_Richardson())

        return np.concatenate(derivatives)

    m_states = len(machines)*states_per_machine
    n_states = network.state.shape[0]
    states = m_states + n_states

    E_inv = np.eye(states)  
    E_inv[m_states:, m_states:] *= network.recip_gamma  
    
    if reset:
        network.reset()
        if machines is not None:
            for machine in machines:
                machine.reset()

    if machines is not None:
        for machine in machines:
            machine.t = t0
            machine.dt = dt
            machine.t_hist.append(t0)

    network.t_hist.append(t0)

    t=t0
    while t<tf:
        # Step t forward
        t += dt

        if machines is not None:
            # Update each machine
            for machine in machines:

                # Extract generator outputs from current state
                generator_outputs = network.extract_generator_outputs(external=True)

                # When indexing is needed
                bus_idx_machine = machine.bus_ID-1

                # Update each machine based on the power flow results
                machine.P = generator_outputs[bus_idx_machine]['P']
                machine.Q = generator_outputs[bus_idx_machine]['Q']
                machine.V = network.V[bus_idx_machine]
                machine.theta = network.theta[bus_idx_machine]
                machine.bus_type = network.bus_types[bus_idx_machine] # Can change from Qlims

        x = [np.array([m.delta, m.omega, m.eqprime]) for m in machines]
        x.append(network.state.copy())
        x = np.concatenate(x)
        f = combined_differential_equations(x)
        x += dt * E_inv @ f

        for i, machine in enumerate(machines):
            # Extracting states for the current machine from the large state vector
            state_start = i * states_per_machine
            state_end = state_start + states_per_machine
            machine_state = x[state_start:state_end]

            # Update machine states
            machine.delta, machine.omega, machine.eqprime = machine_state

            # Update the history of the machine
            machine.t = t
            machine.t_hist.append(t)
            machine.delta_hist.append(machine.delta)
            machine.omega_hist.append(machine.omega)
            machine.eqprime_hist.append(machine.eqprime)
            #machine.update_network_quantities()
            #network.update_network_from_machine(machine.P, machine.Q, machine.V, machine.theta, 
            #                                    machine.bus_ID-1, machine.bus_type)

        # Update the network
        network.state = x[len(machines)*states_per_machine:].copy()
        network.update_PQVtheta()
        network.t=t
        network.t_hist.append(t)
        network.state_hist.append(network.state.copy())

def finite_difference_jacobian(f, x, u=None, t=None, epsilon=1e-5):
    '''
    returns jacobian

    Requires all input functions to wrap their parameters, e.g. via a lambda function.
    '''

    # Helper functions
    def call_f(x, u, t):
        if num_args_f == 1:
            return f(x)
        elif num_args_f == 2:
            return f(x, u)
        elif num_args_f == 3:
            return f(x, u, t)
        else:
            raise AttributeError("Unexpected number of arguments in f($\cdot$).")

    def call_u(x, t):
        if num_args_u == 1:
            return u(t)
        elif num_args_u == 2:
            return u(x, t)
        else:
            raise AttributeError("Unexpected number of arguments in u($\cdot$).")

    # Number of states
    n = len(x)

    # Determine the number of arguments f expects
    num_args_f = len(inspect.signature(f).parameters)

    # Number of controls
    if callable(u):
        num_args_u = len(inspect.signature(u).parameters)
        u_value = call_u(x, t)
    elif u is not None:
        if not np.isscalar(u):
            u_value = u.copy()
            m = len(u_value)
        else:
            u_value = [u]
            m = 1
    else:
        u_value = 0
        m = 0

    # Initialize Jacobian
    jacobian = np.zeros((n, n + m))

    # Compute partial derivatives wrt x
    for i in range(n):
        x_plus = np.copy(x)
        x_plus[i] += epsilon
        f_plus = call_f(x_plus, u_value, t)

        x_minus = np.copy(x)
        x_minus[i] -= epsilon
        f_minus = call_f(x_minus, u_value, t)

        jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)

    # Compute partial derivatives wrt u, if u is provided
    if u_value is not None:
        for i in range(m):
            u_plus = np.copy(u_value)
            u_plus[i] += epsilon
            f_plus = call_f(x, u_plus, t)

            u_minus = np.copy(u_value)
            u_minus[i] -= epsilon
            f_minus = call_f(x, u_minus, t)

            jacobian[:, n + i] = (f_plus - f_minus) / (2 * epsilon)

    return jacobian

def NewtonNd(f, Jf, x0, eps_f, eps_dx=np.inf, eps_xrel=np.inf, u=None, t=None, max_iter=50,
             homotopy_continuation=False):
    '''
    returns x_k, f_k, f_values, x_values, converged

    Requires all input functions to wrap their parameters, e.g. via a lambda function.
    '''
    # Helper functions
    num_args_f = len(inspect.signature(f).parameters)
    num_args_Jf = len(inspect.signature(Jf).parameters)
    def call_f(x, u=None, t=None):
        if num_args_f == 1:
            return f(x)
        elif num_args_f == 2:
            return f(x, u)
        elif num_args_f == 3:
            return f(x, u, t)
        else:
            raise AttributeError("Unexpected number of arguments in f($\cdot$).")

    def call_Jf(x, u, t):
        if num_args_Jf == 1:
            return Jf(x)
        elif num_args_Jf == 2:
            return Jf(x, u)
        elif num_args_Jf == 3:
            return Jf(x, u, t)
        else:
            raise AttributeError("Unexpected number of arguments in Jf($\cdot$).")

    def call_u(x, t):
        if num_args_u == 1:
            return u(t)
        elif num_args_u == 2:
            return u(x, t)
        else:
            raise AttributeError("Unexpected number of arguments in u($\cdot$).")

    # Initialize
    x_k = x0.copy()

    norm_hist = []
    if callable(u):
        num_args_u = len(inspect.signature(u).parameters)
        u_k = call_u(x_k, t)
    elif u is not None:
        u_k = u.copy() if not np.isscalar(u) else u
    else:
        u_k=None

    q=0
    qp = 0.01
    qm = 0.1
    f_k = call_f(x_k, u_k, t)
    Jf_k = call_Jf(x_k, u_k, t)

    # Lists for storing historical values
    f_values = [f_k.copy() if not np.isscalar(f_k) else f_k]
    x_values = [x_k.copy() if not np.isscalar(x_k) else x_k]
    converged = False

    # Iterate until solved or max_iter
    for k in range(max_iter):

        # Try to solve
        if not np.isscalar(f_k):
            try: # N > 1
                delta_x = np.linalg.solve(Jf_k, -f_k)
            except np.linalg.LinAlgError:
                return x_k, f_k, f_values, x_values, False
        elif Jf_k != 0: # N = 1
            delta_x = -f_k/Jf_k
        else:
            return ArithmeticError('Cannot solve due to division by zero.')

        # Prepare next iteration
        x_k = x_k.astype(float) + delta_x
        if callable(u): u_k = call_u(x_k, t)
        elif u is not None: u_k = u.copy() if not np.isscalar(u) else u

        f_k = call_f(x_k, u_k, t)
        Jf_k = call_Jf(x_k, u_k, t)

        # Append results to history
        f_values.append(f_k.copy() if not np.isscalar(f_k) else f_k)
        x_values.append(x_k.copy() if not np.isscalar(x_k) else x_k)

        if homotopy_continuation and len(norm_hist)>=2:
            if (norm_hist[-1][i] > norm_hist[-2][i] for i in range(3)):
                if q-qm>=0: q-= qm
                else: q=0
                f_k = q*call_f(x_k, u_k, t) + (1-q)*x_k.copy()
                J_k = q*call_Jf(x_k, u_k, t) + (1-q)*np.eye(f_k.shape[0])
            elif (norm_hist[-1][i] >= norm_hist[-2][i] for i in range(3)):
                if q+qp<=1: q+=0.01 
                else: q=1
                f_k = q*call_f(x_k, u_k, t) + (1-q)*x_k.copy()
                J_k = q*call_Jf(x_k, u_k, t) + (1-q)*np.eye(J_k.shape)        

        # Check convergence
        norm_f = np.linalg.norm(f_k, np.inf)
        norm_dx = np.linalg.norm(delta_x, np.inf)
        norm_dx_xk = np.linalg.norm(delta_x, np.inf) / max(np.linalg.norm(x_k, np.inf), 1e-12)
        norm_hist.append(np.array([norm_f, norm_dx, norm_dx_xk]))

        if not np.isscalar(f_k): # N > 1
            if (norm_f < eps_f and norm_dx < eps_dx and norm_dx_xk < eps_xrel):
                converged = True
                break
        else: # N = 1
            if (np.abs(f_k) < eps_f and
                np.abs(delta_x) < eps_dx and
                np.abs(delta_x) / max(np.abs(x_k), 1e-12) < eps_xrel):
                converged = True
                break

    return x_k, f_k, f_values, x_values, converged

def PS_trapezoidal(network, machines, t0, tf, dt, eps_f=1e-5, eps_dx=1e-5, eps_xrel=1e-5, 
                   reset=False, dynamic=False, homotopy_continuation=False):
    
    states_per_machine = 3
    m_states = len(machines)*states_per_machine
    n_states = network.state.shape[0]
    states = m_states + n_states
    E = np.eye(states)  
    E[m_states:, m_states:] *= network.recip_gamma

    # Helper function
    def combined_differential_equations(x):
        # Split x back into individual machine states
        derivatives = []

        for i, machine in enumerate(machines):
            state_start = i * states_per_machine
            state_end = state_start + states_per_machine
            machine_state = x[state_start:state_end]

            # Compute the derivatives for this machine
            machine_derivatives = machine.differential_equations(states=machine_state)
            derivatives.append(machine_derivatives)

        # Include the Richardson states
        derivatives.append(network.powerflow_decoupled_Richardson(states=x[len(machines)*states_per_machine:]))

        return np.concatenate(derivatives)

    if reset:
        network.reset()
        for machine in machines:
            machine.reset()

    t=t0
    dt0=dt
    for machine in machines:
        machine.t = t0
        machine.dt = dt
        machine.t_hist.append(t0)

    while t < tf:

        generator_outputs = network.extract_generator_outputs(external=True)

        # Update machine parameters based on power flow results
        for i, machine in enumerate(machines):
            bus_idx_machine = machine.bus_ID - 1
            machine.P = generator_outputs[bus_idx_machine]['P']
            machine.Q = generator_outputs[bus_idx_machine]['Q']
            machine.V = network.V[bus_idx_machine]
            machine.theta = network.theta[bus_idx_machine]
            machine.bus_type = network.bus_types[bus_idx_machine]

        # Concatenating the initial states of all machines
        x_nm1 = [np.array([m.delta, m.omega, m.eqprime]) for m in machines]
        x_nm1.append(network.state.copy())
        x_nm1 = np.concatenate(x_nm1)

        # Dynamic timestep adjustment
        if dynamic:
            # Estimate the rate of change of states
            derivatives = []
            for machine in machines:
                derivatives.append(machine.ddelta_dt)
                derivatives.append(machine.domega_dt)
                derivatives.append(machine.deqprime_dt)
            derivatives += network.dstate_dt.copy().tolist() # concatenates
            max_change_rate = max(derivatives)
            if max_change_rate > 0.01:
                dt = max(dt0, dt / 2)  # Halve dt if change is rapid
            elif max_change_rate < 0.01:
                dt = min(dt0, dt * 1.5)  # Increase dt by 50% if change is slow

        # Step t forward
        t += dt

        # Compute gamma for the entire system
        gamma = E @ x_nm1 + (dt/2) * combined_differential_equations(x_nm1)

        # Define F and J for the entire system
        f = lambda x: combined_differential_equations(x)
        Jf = lambda x: finite_difference_jacobian(f, x=x)

        F_Trap = lambda x: E @ x - (dt/2) * f(x) - gamma
        J_Trap = lambda x: E - (dt/2) * Jf(x)

        # Run the Newton solver for the entire system at timestep t
        x, _, _, _, converged = NewtonNd(
            F_Trap, J_Trap, x0=x_nm1,
            eps_f=eps_f, eps_dx=eps_dx, eps_xrel=eps_xrel,
            max_iter=100, homotopy_continuation=homotopy_continuation
        )

        # Check convergence
        if not converged:
            print(f'Newton solver failed to converge at t={t}.')
            break

        for i, machine in enumerate(machines):
            # Extracting states for the current machine from the large state vector
            state_start = i * states_per_machine
            state_end = state_start + states_per_machine
            machine_state = x[state_start:state_end]

            # Update machine states
            machine.delta, machine.omega, machine.eqprime = machine_state.copy()

            # Update the history of the machine
            machine.t = t
            machine.t_hist.append(t)
            machine.delta_hist.append(machine.delta)
            machine.omega_hist.append(machine.omega)
            machine.eqprime_hist.append(machine.eqprime)
            #machine.update_network_quantities()
            #network.update_network_from_machine(machine.P, machine.Q, machine.V, machine.theta, 
            #                                    machine.bus_ID-1, machine.bus_type)

        network.state = x[len(machines)*states_per_machine:].copy()
        network.update_PQVtheta()
        network.t=t
        network.t_hist.append(t)
        network.state_hist.append(network.state.copy())

def PS_transient_reference(network, machines, t0, tf, min_n=1, max_n=10, desired_confidence=1e-4, version='0'):
    '''
    returns trajectory, times, reference_confidence

    Requires all input functions to wrap their parameters, e.g. via a lambda function.
    '''

    decades = []
    computation_time = []

    # Function to check if the solution contains NaN or Inf values
    def is_invalid_solution(solution):
        return np.isnan(solution).any() or np.isinf(solution).any()

    # Initialize
    prev_solution = None
    reference_confidence = None

    # Run and rerun ForwardEuler, and test the reference_confidence
    for n in range(min_n, max_n + 1):

        print(f'Running Forward Euler with a timestep of dt = 10^(-{n}) seconds.')
        dt = 10**(-n)
        decades.append(dt)

        start_time = time.time()
        _ = PS_forward_euler(network, machines, t0, tf, dt, reset=True)
        end_time = time.time()
        computation_time.append(end_time-start_time)

        # Assuming 'machines' is a list of your machine instances
        num_machines = len(machines)
        num_timesteps = len(machines[0].delta_hist)  # Assuming all machines have the same number of time steps

        # Initialize arrays to store histories
        solution = np.zeros((3*num_machines, num_timesteps))

        # Populate the arrays
        for i, machine in enumerate(machines):
            solution[3*i+0, :] = machine.delta_hist
            solution[3*i+1, :] = machine.omega_hist
            solution[3*i+2, :] = machine.eqprime_hist

        if is_invalid_solution(solution):
            print('Solution contains NaN or Inf values. Skipping this timestep.\n')
            continue

        # Update reference solution and calculate confidence
        if prev_solution is not None:
            # Calculate reference_confidence
            reference_confidence = np.max(np.abs(solution[:,-1] - prev_solution[:,-1]))

            print(f'Reference Confidence: {reference_confidence}\n')

            if reference_confidence < desired_confidence:
                print(f'Surpassed desired confidence with: {reference_confidence} at dt = 10^(-{n}) seconds.')
                break

        prev_solution = solution

        if n == max_n:
            raise ValueError(f'Failed to achieve desired confidence at dt = 10^(-{n}) seconds timestep.')

    # Save the reference trajectory, times, and confidence level
    with open(f'TransientReference_{version}.pickle', 'wb') as file:
        pickle.dump({
            'trajectory': solution,
            'times': machine.t_hist,
            'reference_confidence': reference_confidence,
            'decades' : decades,
            'computation_times': computation_time
            }, file)

    return solution, machine.t_hist, reference_confidence, decades, computation_time