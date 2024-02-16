import dash
from dash import html, dcc, Input, Output, State, ALL
from dash import dash_table
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dynamic_models as dm
import powerflow as pf
import numpy as np
import pickle
import time
import os

current_directory = os.getcwd()
filename = 'Final V2/IEEE 14 bus.raw'
filepath = os.path.join(current_directory, filename)

network = pf.power_network(filepath)
network.presolved_results()

pm1 = p_m=network.data['generator'][0]['PG']/network.S_base
pm2 = p_m=network.data['generator'][1]['PG']/network.S_base
pm3 = p_m=network.data['generator'][2]['PG']/network.S_base
pm6 = p_m=network.data['generator'][3]['PG']/network.S_base
pm8 = p_m=network.data['generator'][4]['PG']/network.S_base

machine1 = dm.sync_machine_Order_III_with_PID_controllers(bus_ID=1, delta_0=0, M=10, p_m=pm1-0.55, p_m_ref=pm1, eqprime_0=0.8,
                                                       K_p_g=0.0, K_i_g=160, K_d_g=0,
                                                   r_a=1e-3, x_d=1.8, x_d_prime=0.093) # Need to tune
machine2 = dm.sync_machine_Order_III_with_PID_controllers(bus_ID=2, delta_0=0, M=5, p_m=pm2, p_m_ref=pm2, eqprime_0=1,
                                                   r_a=1e-3, x_d=1.8, x_d_prime=0.093) # Need to tune
machine3 = dm.sync_machine_Order_III_with_PID_controllers(bus_ID=3, delta_0=0, M=5, p_m=pm3, p_m_ref=pm3, eqprime_0=1,
                                                   r_a=1e-3, x_d=1.8, x_d_prime=0.093) # Need to tune
machine6 = dm.sync_machine_Order_III_with_PID_controllers(bus_ID=6, delta_0=0, M=5, p_m=pm6, p_m_ref=pm6, eqprime_0=1,
                                                   r_a=1e-3, x_d=1.8, x_d_prime=0.093) # Need to tune
machine8 = dm.sync_machine_Order_III_with_PID_controllers(bus_ID=8, delta_0=0, M=5, p_m=pm8, p_m_ref=pm8, eqprime_0=1,
                                                   r_a=1e-3, x_d=1.8, x_d_prime=0.266)

machines =[machine1, machine2, machine3, machine6, machine8]

version = '0'
filename2 = f'Final V2/TransientReference_{version}.pickle'

# Load the data
with open(filename2, 'rb') as file:
    data = pickle.load(file)

solution = data['trajectory']
sol_time = data['times']
reference_confidence = data['reference_confidence']

# Load the data
with open(filename2, 'rb') as file:
    data = pickle.load(file)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Main Container
    html.Div([
        # Left Section
        html.Div([
            # Top Left Sub-section (Buttons)
            html.Div([
                # "Run Powerflow using Newton Method" Button
                html.Button('Run Powerflow using Newton Method', 
                            id='run-powerflow-button', 
                            n_clicks=0,
                            style={'width': '50%', 'margin-bottom': '10px'}),

                # "Run N-1 Contingency Analysis" Button
                html.Button('Run N-1 Contingency Analysis', 
                            id='run-contingency-button', 
                            n_clicks=0,
                            style={'width': '50%', 'margin-bottom': '10px'}),

                # "Reset Network" Button
                html.Button('Reset Network', 
                            id='reset-network-button', 
                            style={'width': '50%', 'margin-bottom': '10px'}),

                # Status display Div
                html.Div(id='status')

            ], style={'width': '98%', 'display': 'flex', 
                    'flex-direction': 'column', 'align-items': 'center', 
                    'height': '28.5%'}),

            # Bottom Left Sub-section (Dropdowns and Display Container)
            html.Div([
                # Container for Dropdowns
                html.Div([
                    # Dropdown for selecting Case
                    dcc.Dropdown(
                        id='case-dropdown',
                        options=[{'label': case, 'value': case} for case in network.results.keys()],
                        value='Presolved'  # Default value
                    ),
                    
                    # Dropdown for selecting Attribute
                    dcc.Dropdown(
                        id='attribute-dropdown',
                        options=[{'label': 'Bus DataFrame', 'value': 'bus_df'},
                                {'label': 'Branch DataFrame', 'value': 'branch_df'},
                                {'label': 'Figure', 'value': 'figure'}],
                        value='bus_df'  # Default value
                    )
                ], style={'flex': 1}),  # Adjust style as needed

                # Container for Displaying DataFrame or Figure
                html.Div(id='display-container', style={'width': '100%'})

            ], style={'width': '98%', 'display': 'flex', 
                      'flex-direction': 'column'})


        ], style={'width': '50%', 'display': 'flex', 'flex-direction': 'column'}),

        # Right Section
        html.Div([
            html.Div(id='dummy-output', style={'display': 'none'}),
            
            # Top Right Sub-section (Machine Selection and Attribute Display)
            html.Div([
                # Dropdown for selecting a machine
                dcc.Dropdown(
                    id='machine-dropdown',
                    options=[{'label': f'Machine {machine.bus_ID}', 'value': i} for i, machine in enumerate(machines)],
                    value=0  # Default to the first machine
                ),

                # Container for displaying machine attributes
                html.Div(id='machine-attributes-container')
            ], style={'width': '98%', 'display': 'flex', 
                      'flex-direction': 'column', 'height': '100%'}),

            # Bottom Right Sub-section
            html.Div([
                # Solver Selection Dropdown
                dcc.Dropdown(
                    id='solver-dropdown',
                    options=[{'label': 'Forward Euler', 'value': 'forward_euler'},
                            {'label': 'Trapezoidal', 'value': 'trapezoidal'}],
                    value='forward_euler'  # Default value
                ),
                
                # Solver parameters container
                html.Div([
                    # Row containing two columns
                    html.Div([
                        # Column 1
                        html.Div([
                            html.Div([
                                html.Label('Reset Simulation:'),
                                dcc.Checklist(
                                    options=[{'label': ' Reset', 'value': 'reset'}],
                                    value=['reset'], id='reset-input'
                                )
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('Start Time (t0):'),
                                dcc.Input(id='t0-input', type='number', placeholder='t0', value=0)
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('End Time (tf):'),
                                dcc.Input(id='tf-input', type='number', placeholder='tf', value=10)
                            ], style={'margin-bottom': '10px'})
                        ], style={'display': 'flex', 'flex-direction': 'column', 'width': '50%', 'padding': '10px'}),

                        # Column 2
                        html.Div([
                            html.Div([
                                html.Label('Time Step (dt):'),
                                dcc.Input(id='dt-input', type='number', placeholder='dt', value=0.01)
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('Use Dynamic Timestepping:'),
                                dcc.Checklist(
                                    options=[{'label': ' Dynamic dt', 'value': 'dynamic_dt'}],
                                    value=['dynamic_dt'], id='dynamic_dt-input'
                                )
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('Use Homotopy Continuation:'),
                                dcc.Checklist(
                                    options=[{'label': ' Continuation', 'value': 'homo_cont'}],
                                    value=['homo_cont'], id='homo_cont-input'
                                )
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('Error Tolerance (eps_f):'),
                                dcc.Input(id='eps_f-input', type='number', placeholder='eps_f', value=1e-8, style={'display': 'none'})
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('Derivative Tolerance (eps_dx):'),
                                dcc.Input(id='eps_dx-input', type='number', placeholder='eps_dx', value=1e-6, style={'display': 'none'})
                            ], style={'margin-bottom': '10px'}),

                            html.Div([
                                html.Label('Relative State Tolerance (eps_xrel):'),
                                dcc.Input(id='eps_xrel-input', type='number', placeholder='eps_xrel', value=1e-6, style={'display': 'none'})
                            ], style={'margin-bottom': '10px'})

                        ], style={'display': 'flex', 'flex-direction': 'column', 'width': '50%', 'padding': '10px'})

                    ], style={'display': 'flex', 'flex-direction': 'row'})

                ], id='solver-parameters-container'),

                # Run Simulation Button
                html.Button('Run Simulation', id='run-simulation-button', n_clicks=0),
                html.Div(id='hidden-div-for-triggering', style={'display': 'none'}),

                # Container for Plotly Plots
                dcc.Graph(id='simulation-results-plot')

            ], style={'width': '98%', 'display': 'flex', 
                      'flex-direction': 'column', 'height': '100%'})
        ], style={'width': '50%', 'display': 'flex', 'flex-direction': 'column'}),
    ], style={'display': 'flex', 'flex-direction': 'row'})
])

#region #---------- Consolodated Buttons ----------#
@app.callback(
    [Output('case-dropdown', 'options'),
     Output('status', 'children'),
     Output('hidden-div-for-triggering', 'children')],
    [Input('reset-network-button', 'n_clicks'),
     Input('run-powerflow-button', 'n_clicks'),
     Input('run-contingency-button', 'n_clicks'),
     Input('run-simulation-button', 'n_clicks')],
    [State('solver-dropdown', 'value'),
     State('machine-dropdown', 'value'),
     State('t0-input', 'value'),
     State('tf-input', 'value'),
     State('dt-input', 'value'),
     State('reset-input', 'value'),
     State('dynamic_dt-input', 'value'),
     State('homo_cont-input', 'value')]
)
def combined_callback(n_clicks_reset, n_clicks_powerflow, n_clicks_contingency, n_clicks_simulation, 
                      solver, machine_index, t0, tf, dt, reset, dynamic, continuation, *trapezoidal_args):
    
    ctx = dash.callback_context
    status=''
    hidden=''
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Logic for each button
    if button_id == 'reset-network-button' and n_clicks_reset > 0:
        # Reset network logic
        network.reset()
        network.result_dataframes()
        network.create_figure()

        network.results['Presolved'] = {
            'bus_df': network.bus_df,
            'branch_df': network.branch_df,
            'graph': network.G,
            'figure': network.fig
        }
        status = 'Network Reset Successfully.'
    elif button_id == 'run-powerflow-button' and n_clicks_powerflow > 0:
        # Powerflow logic
        network.newton_raphson_power_flow()
        network.result_dataframes()
        network.create_figure()

        network.results[f'Solved_V{n_clicks_powerflow}'] = {
            'bus_df': network.bus_df,
            'branch_df': network.branch_df,
            'graph': network.G,
            'figure': network.fig
        }
        status = f'Solved Powerflow V{n_clicks_powerflow}'
    elif button_id == 'run-contingency-button' and n_clicks_contingency > 0:
        # Contingency logic
        network.N_1_contingency_screening()
        status = 'N-1 Contingency Screening Complete.'
    elif button_id == 'run-simulation-button' and n_clicks_simulation > 0:
        # Simulation logic
        reset = True if 'reset' in reset else False
        dynamic = True if 'dynamic_dt' in dynamic else False
        continuation = True if 'dynamic_dt' in continuation else False
        try:
            if solver == 'forward_euler':
                start_time = time.time()
                dm.PS_forward_euler(network, machines, t0, tf, dt, reset)
                end_time = time.time()
            elif solver == 'trapezoidal':
                eps_f, eps_dx, eps_xrel = trapezoidal_args if trapezoidal_args else (1e-8, 1e-6, 1e-6)
                start_time = time.time()
                dm.PS_trapezoidal(network, machines, t0, tf, dt, eps_f, eps_dx, eps_xrel, 
                                  reset=reset, dynamic=dynamic, homotopy_continuation=continuation)
                end_time = time.time()

            sol_idx = int(np.array(sol_time).round(5).tolist().index(1))
            sol_val = solution[:,sol_idx]
            xlist = [np.array([m.delta_hist, m.omega_hist, m.eqprime_hist]) for m in machines]
            x = np.vstack(xlist)
            x_idx = np.array(machine1.t_hist).round(5).tolist().index(1)
            x_val = x[:,x_idx]
            confidence = np.max(np.abs(x_val - sol_val))

            status = f'Simulation Completed at: {datetime.now()} in {end_time-start_time:.2f} seconds with error {confidence.round(3)}.'
            network.presolved_results()
            network.dynamic_results()
            hidden = status
        except Exception as e:
            status = f'Error during simulation: {e}'
    # Update case-dropdown options
    return [{'label': case, 'value': case} for case in network.results.keys()], status, hidden
#endregion

#region #---------- Botton Left ----------#
@app.callback(
    Output('attribute-dropdown', 'options'),
    [Input('case-dropdown', 'value')]
)
def update_attribute_dropdown(selected_case):
    return [{'label': attr, 'value': attr} for attr in network.results[selected_case].keys()]

@app.callback(
    Output('display-container', 'children'),
    [Input('case-dropdown', 'value'),
     Input('attribute-dropdown', 'value')],
    prevent_initial_call=True
)
def update_display(selected_case, selected_attribute):
    case_item = network.results.get(selected_case)

    # Check if the case item is a string (error message)
    if isinstance(case_item, str):
        return html.Div(case_item)  # Display the error message

    if selected_attribute in ['bus_df', 'branch_df']:
        # Format DataFrame to limit decimals to three places
        formatted_df = network.results[selected_case][selected_attribute].map(lambda x: round(x, 3) if isinstance(x, (float, int)) else x)

        # Display DataFrame with horizontal scroll
        return dash_table.DataTable(
            data=formatted_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in formatted_df.columns],
            style_table={'overflowX': 'auto'}  # Enable horizontal scrolling
        )
    elif isinstance(case_item[selected_attribute], go.Figure):
        # Display Plotly Figure
        return dcc.Graph(figure=case_item[selected_attribute])
#endregion

#region #---------- Top Right ----------#
@app.callback(
    Output('machine-attributes-container', 'children'),
    [Input('machine-dropdown', 'value')]
)
def display_machine_attributes(selected_machine_index):
    machine = machines[selected_machine_index]
    attributes = [
        'D', 'M', 'omega_s', 'p_m_0', 'p_m_ref', 'K_p_g', 'K_i_g', 'K_d_g',
        'r_a', 'x_d', 'x_d_prime', 'T_d0_prime', 'v_f_0', 'v_ref', 'K_p_V', 'K_i_V', 'K_d_V'
    ]

    # Split the attributes into two columns
    half_point = len(attributes) // 2
    column_1_attributes = attributes[:half_point]
    column_2_attributes = attributes[half_point:]

    # Create a Div for each attribute
    def create_attribute_div(attr):
        return html.Div([
            html.Label(f'{attr}: '),
            dcc.Input(
                id={'type': 'machine-attribute-input', 'index': selected_machine_index, 'attribute': attr},
                type='number',
                value=getattr(machine, attr),
                style={'margin-left': '5px'}
            )
        ], style={'margin-bottom': '10px'})

    # Construct the two-column layout
    two_column_layout = html.Div([
        html.Div([create_attribute_div(attr) for attr in column_1_attributes], 
                 style={'display': 'flex', 'flex-direction': 'column', 'flex': '1'}),
        html.Div([create_attribute_div(attr) for attr in column_2_attributes], 
                 style={'display': 'flex', 'flex-direction': 'column', 'flex': '1'})
    ], style={'display': 'flex', 'flex-direction': 'row'})

    return two_column_layout

@app.callback(
    Output('dummy-output', 'children'),  # This can be a hidden Div as we don't need to update anything on the page
    [Input({'type': 'machine-attribute-input', 'index': ALL, 'attribute': ALL}, 'value')],
    [State({'type': 'machine-attribute-input', 'index': ALL, 'attribute': ALL}, 'id')]
)
def update_machine_attribute(values, ids):
    for value, id in zip(values, ids):
        machine_index = id['index']
        attribute = id['attribute']
        setattr(machines[machine_index], attribute, value)
    return None  # No actual page update needed
#endregion

#region #---------- Bottom Right ----------#
@app.callback(
    Output('solver-parameters-container', 'children'),
    [Input('solver-dropdown', 'value')]
)
def update_solver_parameters(selected_solver):
    # Common parameters in the first column
    column_1_parameters = [
        html.Div([
            html.Label('Reset Simulation:'),
            dcc.Checklist(
                options=[{'label': ' Reset', 'value': 'reset'}],
                value=['reset'], id='reset-input'
            )
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label('Start Time (t0):'),
            dcc.Input(id='t0-input', type='number', placeholder='t0', value=0)
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label('End Time (tf):'),
            dcc.Input(id='tf-input', type='number', placeholder='tf', value=3)
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label('Time Step (dt):'),
            dcc.Input(id='dt-input', type='number', placeholder='dt', value=0.01)
        ], style={'margin-bottom': '10px'})
    ]

    # Parameters for both solvers in the second column
    column_2_parameters = [
        # Additional parameters for the "trapezoidal" solver
        html.Div([
            dcc.Checklist(
                options=[{'label': ' Dynamic dt', 'value': 'dynamic_dt'}],
                value=['dynamic_dt'], id='dynamic_dt-input'
            )
        ], style={'margin-bottom': '10px'}),

        html.Div([
            dcc.Checklist(
                options=[{'label': ' Homotopy Continuation', 'value': 'homo_cont'}],
                value=['homo_cont'], id='homo_cont-input'
            )
        ], style={'margin-bottom': '10px'}),

        html.Div([
            html.Label('Error Tolerance (eps_f):'),
            dcc.Input(id='eps_f-input', type='number', placeholder='eps_f', value=1e-8)
        ], id='eps_f-container', style={'margin-bottom': '10px', 'display': 'none' if selected_solver != 'trapezoidal' else ''}),

        html.Div([
            html.Label('Derivative Tolerance (eps_dx):'),
            dcc.Input(id='eps_dx-input', type='number', placeholder='eps_dx', value=1e-6)
        ], id='eps_dx-container', style={'margin-bottom': '10px', 'display': 'none' if selected_solver != 'trapezoidal' else ''}),

        html.Div([
            html.Label('Relative State Tolerance (eps_xrel):'),
            dcc.Input(id='eps_xrel-input', type='number', placeholder='eps_xrel', value=1e-6)
        ], id='eps_xrel-container', style={'margin-bottom': '10px', 'display': 'none' if selected_solver != 'trapezoidal' else ''}),
    ]

    # Construct the two-column layout
    two_column_layout = html.Div([
        html.Div(column_1_parameters, style={'display': 'flex', 'flex-direction': 'column', 'flex': '1'}),
        html.Div(column_2_parameters, style={'display': 'flex', 'flex-direction': 'column', 'flex': '1'})
    ], style={'display': 'flex', 'flex-direction': 'row'})

    return two_column_layout

@app.callback(
    [Output('homo_cont-input', 'style'), 
     Output('dynamic_dt-input', 'style'), 
     Output('eps_f-input', 'style'),
     Output('eps_dx-input', 'style'),
     Output('eps_xrel-input', 'style')],
    [Input('solver-dropdown', 'value')]
)
def update_solver_input_visibility(selected_solver):
    if selected_solver == 'trapezoidal':
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('simulation-results-plot', 'figure'),
    [Input('machine-dropdown', 'value'),
     Input('hidden-div-for-triggering', 'children')],
)
def update_plot_based_on_machine_or_simulation(machine_index, _):
    selected_machine = machines[machine_index]
    
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Delta vs. Time', 'Omega vs. Time', "Eq' vs. Time"))

    # Delta vs Time
    fig.add_trace(go.Scatter(x=selected_machine.t_hist, y=180/np.pi*np.array(selected_machine.delta_hist), mode='lines'), row=1, col=1)
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_yaxes(title_text='Delta (deg)', row=1, col=1)

    # Omega vs Time
    fig.add_trace(go.Scatter(x=selected_machine.t_hist, y=60*np.array(selected_machine.omega_hist), mode='lines'), row=2, col=1)
    fig.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig.update_yaxes(title_text='Omega (deg/s)', row=2, col=1)

    # Eq' vs Time
    fig.add_trace(go.Scatter(x=selected_machine.t_hist, y=selected_machine.eqprime_hist, mode='lines'), row=3, col=1)
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text="Eq' (V)", row=3, col=1)

    fig.update_layout(height=600, showlegend=False)
    return fig
#endregion

if __name__ == '__main__':
    app.run_server(debug=True)



# Ideas
# Implement Homotopy Continuation
# See if there exists an HC for Powerflow
# Fix Q and see if it solves
