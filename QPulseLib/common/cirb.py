# -*- encoding: utf-8 -*-
'''
cirb is a python file that has 4 main functions:
1. do some operation on the experimental circuit, including dynamical decoupling add, align circuit sequence before and after and so on.
2. simplify experimental circuit with pyzx.
3. transfer single qubit gates into PVZ gate.
4. simulate circuit in either cirq type or exp type.
5. qst simplify circuit
'''

import numpy as np
import copy
import matplotlib.pyplot as plt
from functools import reduce
import itertools
from scipy.linalg import expm
from consts import BASIS_GATES

# plt.rcParams['font.size'] = 18

try:
    import qiskit
except:
    print('qiskit is not installled')
try:
    import pyzx as zx
except:
    print('pyzx is not installed')
try:
    import cirq
    from cirq.contrib.qasm_import import circuit_from_qasm
except:
    print('cirq is not installed')
try:
    import qsimcirq
except:
    print('qsimcirq is not installed')
try:
    import quaternion
except:
    print('numpy-quaternion is not installed')
try:
    import stim
except:
    print('stim is not installed')
try:
    import qutip
except:
    print('qutip is not installed')


class QuaternionRepresentation:
    '''
    quaternion representation of Eular angles
    '''

    def __init__(self):
        self.I2 = np.diag([1, 1])
        self.sigmaX = np.array([[0, 1], [1, 0]])
        self.sigmaY = np.array([[0, -1j], [1j, 0]])
        self.sigmaZ = np.array([[1, 0], [0, -1]])

    def matrix_to_quaternion(self, m):
        i = -1j * self.sigmaX
        j = -1j * self.sigmaY
        k = -1j * self.sigmaZ
        return quaternion.from_float_array(
            np.abs([
                np.trace(m) / 2, -np.trace(np.dot(m, i) / 2),
                -np.trace(np.dot(m, j) / 2), -np.trace(np.dot(m, k) / 2)
            ]))

    def quaternion_to_matrix(self, q):
        i = -1j * self.sigmaX
        j = -1j * self.sigmaY
        k = -1j * self.sigmaZ
        return q.w * self.I2 + q.x * i + q.y * j + q.z * k

    def rotation(self, alpha, theta, phi):
        '''
        auther: xsb
        update: 20220820

        Return the single qubit rotation operator quaternion. Rotate the qubit
        around (1, theta, phi) with angle alpha

        Parameters
        ----------
        alpha : rotation angle.
        theta : polar angle.
        phi : equatorial angle.
        '''
        return np.quaternion(np.cos(alpha / 2),
                             np.sin(alpha / 2) * np.sin(theta) * np.cos(phi),
                             np.sin(alpha / 2) * np.sin(theta) * np.sin(phi),
                             np.sin(alpha / 2) * np.cos(theta))

    def rotation_euler_angle(self, theta1, theta2, theta3):
        '''
        RZ(theta1)RY(theta2)RZ(theta3)
        '''
        return self.rotation(theta1, 0, 0) * self.rotation(
            theta2, np.pi / 2, np.pi / 2) * self.rotation(theta3, 0, 0)

    def rotation_two_gate_PhX_angle(self, theta1, theta2, phi1, phi2=0):
        '''
        PhX(theta1, phi1)PhX(theta2, phi2)
        phi2 = 0 or pi, make sure theta2 <= pi

        Returns
        ---------------
        q: quaternion
        '''
        return self.rotation(theta1, np.pi / 2, phi1) * self.rotation(
            theta2, np.pi / 2, phi2)

    def as_two_gate_PhX_angle(self, q):
        '''
        PhX(theta1, phi1)PhX(theta2, phi2)
        phi2 = 0 or pi, make sure theta2 <= pi
        
        Returns
        ---------------
        [theta1, theta2, phi1, phi2]: float array
        '''
        phi2 = 0
        if np.abs(q.y) < 1e-10:
            if np.abs(q.z) < 1e-10:
                phi1 = 0
                _theta = np.angle(q.w + 1j * q.x) % (np.pi * 2)
                theta1 = theta2 = _theta
            else:
                theta1 = 2 * np.arccos(q.x)
                theta2 = np.pi
                phi1 = np.angle(-(q.w + 1j * q.z) / np.sin(theta1 / 2))
        else:
            theta2 = (2 * np.arctan(-q.z / q.y)) % (2 * np.pi)
            theta1 = 2 * np.arccos(q.x * np.sin(theta2 / 2) +
                                   q.w * np.cos(theta2 / 2))
            sin_phi = q.y / np.sin(theta1 / 2) / np.cos(theta2 / 2)
            cos_phi = (q.x - np.cos(theta1 / 2) * np.sin(theta2 / 2)) / np.sin(
                theta1 / 2) / np.cos(theta2 / 2)
            phi1 = np.angle(cos_phi + 1j * sin_phi) % (2 * np.pi)
        if theta1 > np.pi:
            theta1 = 2 * np.pi - theta1
            phi1 = (phi1 + np.pi) % (2 * np.pi)
        if theta2 > np.pi:
            theta2 = 2 * np.pi - theta2
            phi2 = (phi2 + np.pi) % (2 * np.pi)

        return np.array([theta1, theta2, phi1, phi2])

    def rotation_three_gate_PhX_angle(self, phi1, phi2, phi3):
        '''
        PhX(pi/2, phi1)PhX(pi/2, phi2)PhX(pi, phi3)

        Returns
        ---------------
        q: quaternion
        '''
        return self.rotation(np.pi / 2, np.pi / 2, phi1) * self.rotation(
            np.pi / 2, np.pi / 2, phi2) * self.rotation(np.pi, np.pi / 2, phi3)

    def as_three_gate_PhX_angle(self, q):
        '''
        PhX(pi/2, phi1)PhX(pi/2, phi2)PhX(pi, phi3)
        
        Returns
        ---------------
        [phi1, phi2, phi3]: float array
        '''
        if np.abs(q.x) < 1e-10 and np.abs(q.y) < 1e-10:
            phi1 = phi2 = 0
            phi3 = -np.angle(-q.w - q.z * 1j)
        elif np.abs(q.x) < 1e-10:
            phi3 = np.arcsin(q.y)
            phi1 = np.angle(-1 / np.cos(phi3) * (q.w + 1j * q.z))
            phi2 = phi1 + 2 * phi3
        elif np.abs(q.y) < 1e-10:
            phi3 = np.arccos(q.x)
            phi1 = np.angle(1 / np.sin(phi3) * (q.z - 1j * q.w))
            phi2 = phi1 + 2 * phi3 - np.pi
        elif np.abs(q.w) < 1e-10 and np.abs(q.z) < 1e-10:
            phi1 = np.pi
            phi2 = 0
            phi3 = np.angle(q.x + 1j * q.y)
        elif np.abs(q.w) < 1e-10:
            _phi = np.arccos(-q.z)
            phi1 = np.angle(-1 / np.sin(_phi) * (q.x + 1j * q.y))
            phi2 = phi1 - 2 * _phi
            phi3 = (phi1 + phi2 - np.pi) / 2
        elif np.abs(q.z) < 1e-10:
            _phi = np.arccos(-q.w)
            phi1 = np.angle(1 / np.sin(_phi) * (-q.y + 1j * q.x))
            phi2 = phi1 - 2 * np.arccos(-q.w)
            phi3 = (phi1 + phi2) / 2
        else:
            _phi_a = np.arctan(q.z / q.w)
            _phi_b = np.arctan(-q.x / q.y)
            _phi_c = np.angle(-q.w / np.cos(_phi_a) + 1j * q.x / np.sin(_phi_b))
            phi1 = _phi_a + _phi_b
            phi2 = phi1 - 2 * _phi_c
            phi3 = (phi1 + phi2 - 2 * _phi_a) / 2
        phi1 = phi1 % (2 * np.pi)
        phi2 = phi2 % (2 * np.pi)
        phi3 = phi3 % (2 * np.pi)
        return np.array([phi1, phi2, phi3])


def meas_state_vector(state_vector, meas_indexes=[0, 1, 2], normalized=True):
    '''
    calculate traced data from state vector, developed by L.X.
    '''
    state_vector = np.array(state_vector)
    num_qs = int(np.log2(len(state_vector)))
    env_indexes = list(set(np.arange(num_qs)) - set(meas_indexes))
    num_meas = len(meas_indexes)
    num_envs = len(env_indexes)
    assert 2 ** num_qs == len(state_vector)
    Zs = reduce(np.kron, [[1, -1]] * num_meas)
    sum_env_idx = []
    for env_number in np.arange(2 ** num_envs):
        env_string = '{:0{}b}'.format(env_number, num_envs)
        sum_env_idx.append(
            np.sum([int(env_string[i]) << (num_qs - env_indexes[i] - 1) for i in range(num_envs)]))
    sum_env_idx = np.asarray(sum_env_idx, dtype=int)
    if not normalized:
        normal_divider = np.sum(np.dot(state_vector, state_vector.conj()))
    else:
        normal_divider = 1

    meas_result = 0
    for diag_i, diag_z in enumerate(Zs):
        diag_string = '{:0{}b}'.format(diag_i, num_meas)
        sum_qm_idx = np.sum([int(diag_string[i]) << (num_qs - meas_indexes[i] - 1)
                             for i in range(num_meas)])
        sum_vec_idx = sum_env_idx + sum_qm_idx
        meas_coeffs = state_vector[sum_vec_idx]
        # return sum_vec_idx
        tensor_prob = np.sum(
            np.dot(meas_coeffs, meas_coeffs.conj())) / normal_divider
        meas_result += tensor_prob * diag_z
    return meas_result


def ptrace_dense(state, sel):
    '''
    quick partial trace from a state vector or a rho mat to a rho mat
    '''
    qnum = int(np.log2(len(state)))
    rd = np.asarray([2] * qnum, dtype=np.int32).ravel()
    nd = len(rd)
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)
    sel = list(np.sort(sel))
    for x in sel:
        if not 0 <= x < len(rd):
            raise IndexError("Invalid selection index in ptrace.")
    dkeep = (rd[sel]).tolist()
    qtrace = list(set(np.arange(nd)) - set(sel))
    dtrace = (rd[qtrace]).tolist()
    if len(dkeep) + len(dtrace) != len(rd):
        raise ValueError("Duplicate selection index in ptrace.")
    if not dtrace:
        # If we are keeping all dimensions, no need to construct an ndarray.
        return state
    rd = list(rd)
    if len(np.shape(state)) == 1 or np.shape(state)[0] == 1 or np.shape(state)[1] == 1:
        vmat = (state
                .reshape(rd)
                .transpose(sel + qtrace)
                .reshape([np.prod(dkeep, dtype=np.int32),
                          np.prod(dtrace, dtype=np.int32)]))
        rhomat = vmat.dot(vmat.conj().T)
    else:
        rhomat = np.trace(state
                          .reshape(rd + rd)
                          .transpose(qtrace + [nd + q for q in qtrace] +
                                     sel + [nd + q for q in sel])
                          .reshape([np.prod(dtrace, dtype=np.int32),
                                    np.prod(dtrace, dtype=np.int32),
                                    np.prod(dkeep, dtype=np.int32),
                                    np.prod(dkeep, dtype=np.int32)]))
    return rhomat


def entropy_vn(rho):
    rhodiag = np.linalg.eigvalsh(rho)
    rhodiag = np.abs(rhodiag)
    entropy_v = -np.sum(rhodiag * np.log(rhodiag))
    return entropy_v


def plot_simu_ope_res(exp_res, title=None):
    '''
    plot simulation of results, exp_res in a form {'op1':0.1,'op2':0.5}
    '''
    X_up = []
    Y_up = []
    X_down = []
    Y_down = []
    xs = []
    i = 1
    for key in list(exp_res.keys()):
        value_tmp = exp_res[key]
        if isinstance(value_tmp, (int, float)):
            pass
        elif len(value_tmp) > 0:
            value_tmp = np.mean(value_tmp)
        if value_tmp > 0:
            X_up.append(i)
            Y_up.append(value_tmp)
        else:
            X_down.append(i)
            Y_down.append(value_tmp)
        xs.append(i)
        i += 1
    plt.figure(figsize=(max(len(xs) * 0.75, 8), 6), dpi=100)
    plt.bar(X_up, Y_up, facecolor='#9999ff', edgecolor='white')
    plt.bar(X_down, Y_down, facecolor='#ff9999', edgecolor='white')
    plt.hlines(0.6, 0, i, color='r', linestyles='dotted')
    plt.hlines(-0.6, 0, i, color='b', linestyles='dotted')
    # xticks(X)
    for x, y in zip(X_up, Y_up):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    for x, y in zip(X_down, Y_down):
        plt.text(x, y - 0.15, '%.2f' % y, ha='center', va='bottom')
    for x in xs:
        if x in X_up:
            color = '#9999ff'
        else:
            color = '#ff9999'
        plt.axvline(x=x, ls='--', color=color)
    plt.xticks([index for index in xs], list(exp_res.keys()))
    plt.ylabel('Value')
    plt.xlim(0.2, i - 0.3)
    plt.ylim(-1.25, +1.25)
    if title:
        plt.title(title)
    plt.tight_layout()


def expop_to_cirqgate(op, generator=None):
    if op in ['dcz']:
        return [generator.CZ]
    elif op in ['PASS', 'I', '_I_']:
        return [generator.I]
    else:
        if isinstance(op, str):
            angle = np.pi / 2 if '/2' in op else np.pi
            angle = -angle if '-' in op else angle
            gate_op = op.replace('/2', '').replace('-', '').replace('_', '')
            if gate_op in ['VZ']:
                gate_op = gate_op.replace('V', '')
            elif gate_op == '_Y_':
                gate_op = 'Y'
            sq_rot_str = 'r' + gate_op.lower()
            return [generator.__getattribute__(sq_rot_str)(angle)]
        elif len(op) == 2 and isinstance(op[1], (float, int)):
            (gate_op, angle) = op
            if gate_op in ['VZ', '_VZ_']:
                gate_op = gate_op.replace('V', '')
            if gate_op.startswith('R') or gate_op.startswith('r'):
                sq_rot_str = gate_op.lower()
            else:
                sq_rot_str = 'r' + gate_op.lower()
            return [generator.__getattribute__(sq_rot_str)(angle)]
        elif len(op) == 4 and op[0] == 'PVZ':
            (_, alpha, theta, phi) = op
            return [generator.__getattribute__("rz")(phi),
                    generator.__getattribute__("rz")(-theta),
                    generator.__getattribute__("rx")(alpha),
                    generator.__getattribute__("rz")(theta)]
        elif isinstance(op, tuple) and op[0] == 'IDVZ':
            return [generator.I]
        else:
            raise Exception(f'Unknown op: {op}')


def control_phase_in(phase, phase_range=[-np.pi, np.pi]):
    phase = phase - 2 * np.pi * (phase // (np.pi * 2))
    if phase < phase_range[0]:
        phase += np.pi * 2
    if phase > phase_range[1]:
        phase -= np.pi * 2
    return phase


class ExpCircuit:
    '''
    circuit use for experiment
    ----------------------------------------------------------------
    In a form: 
        [
            ['VX','I','-X/2',('X',np.pi/2),('PVZ',np.pi/2,0,0),...],
            ['dcz0_1','dcz0_1',...],
            ...
        ]
    ----------------------------------------------------------------
    functions:
        plot(scale=0.9,fold=200,t_xy=30,t_cz=60)
        print(width=10)
        simu(meas_ops={'op0':['X_0']})
    '''

    def __init__(self, circuit):
        self._circuit = copy.deepcopy(circuit)
        self._circuit_type = 'exp'

    def plot(self, scale=0.9, fold=200, t_xy=24, t_cz=60):
        qs = qiskit.QuantumRegister(len(self._circuit[0]), 'q')
        cir = qiskit.QuantumCircuit(qs)
        t_all = 0
        d_all = 0
        barrier_idx = 0
        for ops in self._circuit:
            cz_gate_checked = []
            cnot_gate_checked = []
            t_step = 0
            for i, gate in enumerate(ops):
                if len(ops) == 1:
                    custom_gate = qiskit.circuit.Gate(gate, len(qs), [])
                    cir.append(custom_gate, qs)
                    continue
                if 'dcz' not in gate and 'CNOT' not in gate:
                    if isinstance(gate, str):
                        custom_gate = qiskit.circuit.Gate(gate, 1, [])
                    elif isinstance(gate, (tuple, list)):
                        gate, param = gate[0], gate[1:]
                        custom_gate = qiskit.circuit.Gate(gate, 1, param)
                    else:
                        gate = eval(gate)
                        gate, param = gate[0], gate[1:]
                        custom_gate = qiskit.circuit.Gate(gate, 1, param)
                    t_step = max(t_step, t_xy) if gate != 'VZ' else max(t_step, 0)
                    cir.append(custom_gate, [i])
                elif 'dcz' in gate:
                    if gate not in cz_gate_checked:
                        qi, qj = gate[3:].split('_')
                        cir.cz(int(qi), int(qj))
                        cz_gate_checked.append(gate)
                        t_step = max(t_step, t_cz)
                elif 'CNOT' in gate:
                    if gate not in cnot_gate_checked:
                        qi, qj = gate[4:].split('_')
                        cir.cnot(int(qi), int(qj))
                        cnot_gate_checked.append(gate)
                        t_step = max(t_step, t_cz)
            t_all += t_step
            if any([gate != 'VZ' for gate in ops]):
                cir.barrier()
                d_all = d_all + 1
        fig = cir.draw(output='mpl',
                       scale=scale,
                       vertical_compression='high',
                       fold=fold,
                       idle_wires=False,
                       style={
                           "linecolor": "#1e1e1e",
                           "gatefacecolor": "#25a4e8",
                           "displaycolor": {
                               "_Y_": ["#539f18", "#FFFFFF"],
                               "_I_": ["#d7d7d7", "#FFFFFF"],
                               "I": ["#d9d2e9", "#FFFFFF"],
                               "VZ": ["#19CAAD", "#FFFFFF"],
                               'PVZ': ["#F4604C", "#FFFFFF"]
                           }
                       })
        ax = fig.get_axes()[0]
        ax.set_title('circuit depth: {} total time: {} ns'.format(d_all, t_all),
                     fontsize=20 / scale)
        fig.tight_layout()
        # return fig

    def print(self, width=10):
        print('[')
        for ops in self._circuit:
            print('    [', end='')
            for op in ops:
                if isinstance(op, str):
                    print("\"{}\",".format(op).ljust(width, ' '), end='')
                else:
                    print("{},".format(op).ljust(width, ' '), end='')
            print('],')
        print(']')

    def simu(self, meas_ops={'op0': ['X_0']},
             freq_shift_error: dict = None,
             gate_error: dict = None, is_plot=False, ini_state=None):
        '''
        freq_shift_error:dict=None
            t_xy: time of single qubit gate, in ns : float | int 
            t_cz: time of two qubits gate, in ns : float | int
            freq_shifts: frequency shifts, in MHz : list
                if freq_shifts in type int or float, do
                freq_shifts = np.random.random(qnum)*freq_shifts*2-freq_shifts
        gate_error:dict=None
            t1: t1 of all qubits, in ns : int or float
                if t1 is None, pass it
            t2: t2 of all qubits, in ns : int or float
                if t2 is None, pass it
            gate_error_1q, pauli error: dict
                if gate_error_1q in type int or float, do
                gate_error_1q = zip(np.arange(0,qnum),[gate_error_1q]*qnum)
                if gate_error_1q is None, pass it
            gate_error_2q, pauli error: dict
                if gate_error_2q in type int or float, do
                mean_gate_error_2q = gate_error_2q
                gate_error_2q={}
                for i in itertools.combinations(range(qnum),2):
                    gate_error_2q.update({i:mean_gate_error_2q})
        '''
        circuit_tmp = copy.deepcopy(self._circuit)
        qnum = len(circuit_tmp[0])
        if freq_shift_error is not None:
            try:
                t_xy = freq_shift_error['t_xy']
                t_cz = freq_shift_error['t_cz']
                freq_shifts = freq_shift_error['freq_shift']
                if isinstance(freq_shifts, (float, int)):
                    freq_shifts = np.random.random(qnum) * freq_shifts * 2 - freq_shifts
                assert len(freq_shifts) == qnum
            except:
                raise Exception('Input freq_shifts_error type error!')
        self.t1 = None
        self.t2 = None
        t_gate_default = 30
        self.gate_error_1q = None
        self.gate_error_2q = None
        if gate_error is not None:
            try:
                if 't1' in gate_error.keys():
                    self.t1 = gate_error['t1']
                if 't2' in gate_error.keys():
                    self.t2 = gate_error['t2']
                if 'gate_error_1q' in gate_error.keys():
                    self.ate_error_1q = gate_error['gate_error_1q']
                    if isinstance(gate_error['gate_error_1q'], (float, int)):
                        self.gate_error_1q = dict(zip(np.arange(0, qnum), [gate_error['gate_error_1q']] * qnum))
                if 'gate_error_2q' in gate_error.keys():
                    self.gate_error_2q = gate_error['gate_error_2q']
                    if isinstance(gate_error['gate_error_2q'], (float, int)):
                        mean_gate_error_2q = self.gate_error_2q
                        self.gate_error_2q = {}
                        for i in itertools.combinations(range(qnum), 2):
                            self.gate_error_2q.update({i: mean_gate_error_2q})
            except:
                raise Exception('Input gate_error type error!')
        # print('t1',self.t1)
        # print('t2',self.t2)
        # print('1q',self.gate_error_1q)
        # print('2q',self.gate_error_2q)
        circuit = cirq.Circuit()
        gg = cirq
        Qs = [cirq.NamedQubit('Q{}'.format(i)) for i in range(qnum)]
        for Uc_ops in circuit_tmp:
            added_cz_gates = []
            added_cz_qname = []
            for i, op in enumerate(Uc_ops):
                if 'dcz' in op:
                    if op not in added_cz_gates:
                        i0, i1 = list(
                            map(int, op.replace('dcz', '').split('_')))
                        i0, i1 = min(i0, i1), max(i0, i1)
                        added_cz_qname.append(i0)
                        added_cz_qname.append(i1)
                        q_gate = self.noise_channel(cirq.CZ(Qs[i0], Qs[i1]), (i0, i1))
                        circuit.append(q_gate)

                        if self.gate_error_2q is not None:
                            print('add error on ', op)
                            for ii in [i0, i1]:
                                # theta_tmp=(np.random.random()*2-1)*self.gate_error_2q[(i0,i1)]*np.pi*10
                                theta_tmp = self.gate_error_2q[(i0, i1)] * np.pi * 10
                                test_random = np.random.random()
                                if op == 'dcz20_21':
                                    theta_tmp = 1.5e-2 * np.pi * 10
                                if test_random < 1 / 3:
                                    circuit.append(cirq.rx(theta_tmp)(Qs[ii]))
                                elif test_random < 2 / 3:
                                    circuit.append(cirq.ry(theta_tmp)(Qs[ii]))
                                elif test_random < 1:
                                    circuit.append(cirq.rz(theta_tmp)(Qs[ii]))
                        added_cz_gates.append(op)
                else:
                    q_gates = expop_to_cirqgate(op, generator=gg)
                    ops = []
                    for q_gate in q_gates:
                        ops.append(q_gate(Qs[i]))
                    q_gates = self.noise_channel(ops, i)
                    for q_gate in q_gates:
                        circuit.append(q_gate)

                    if self.gate_error_1q is not None:
                        theta_tmp = (np.random.random() * 2 - 1) * self.gate_error_1q[i] * np.pi * 10
                        test_random = np.random.random()
                        if test_random < 1 / 3:
                            circuit.append(cirq.rx(theta_tmp)(Qs[i]))
                        elif test_random < 2 / 3:
                            circuit.append(cirq.ry(theta_tmp)(Qs[i]))
                        elif test_random < 1:
                            circuit.append(cirq.rz(theta_tmp)(Qs[i]))

            if freq_shift_error is not None:
                if len(added_cz_qname) > 0:
                    for i in range(qnum):
                        circuit.append(
                            cirq.rz(freq_shifts[i] * t_xy * 2 * np.pi * 1e-3)(Qs[i]))
                else:
                    for i in range(qnum):
                        circuit.append(
                            cirq.rz(freq_shifts[i] * t_cz * 2 * np.pi * 1e-3)(Qs[i]))
        # if self.t1 is not None:
        #     circuit = circuit.with_noise(cirq.amplitude_damp(gamma=t_gate_default/self.t1))
        # if self.t2 is not None:
        #     circuit = circuit.with_noise(cirq.phase_damp(gamma=t_gate_default/self.t2))
        return CirqCircuit(circuit, qnum).simu(
            meas_ops=meas_ops, is_plot=is_plot, ini_state=ini_state)

    def noise_channel(self, ops, qname):
        if isinstance(ops, list):
            if len(ops) == 1:
                op = ops[0]
                # if self.gate_error_1q is not None and op!=cirq.I:
                #     op = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(self.gate_error_1q[qname])).noisy_operation(op)
                # op = cirq.NoiseModel.from_noise_model_like(cirq.asymmetric_depolarize(p_x=self.gate_error_1q[qname])).noisy_operation(op)
                return [op]
            if len(ops) > 1:
                ops = copy.deepcopy(ops)
                op = ops[0]
                # if self.gate_error_1q is not None and op!=cirq.I:
                #     op = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(self.gate_error_1q[qname])).noisy_operation(op)
                ops[0] = op
                return ops
        else:
            if isinstance(qname, tuple):
                op = copy.deepcopy(ops)
                # if self.gate_error_2q is not None:
                #     op = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(self.gate_error_2q[qname])).noisy_operation(op)
                return op


class StdCircuit:
    '''
    circuit of Standard in cirb, could work in ZxSimplifier, 
        CircuitAligner, RotationCompiler
    ----------------------------------------------------------------
    In a form:
        {
            'qnum':5,
            'layer_num':3,
            {'layer0':
                {'sq':
                    0:'X',  1:('X',np.pi)}
                {'tq':
                    (2,3):'CZ'}},
            {'layer1':...},
            {...},
        }
    ----------------------------------------------------------------
    '''

    def __init__(self, circuit):
        self._circuit = circuit
        self._circuit_type = 'std'


class QiskitCircuit:
    '''
    qiskit circuit, use for functions in ZxSimplifier
    ---------------------------------------------------------------
    functions:
        plot_circuit(scale=0.9,fold=70)
    ---------------------------------------------------------------
    to be developed:
        simu_circuit
    '''

    def __init__(self, circuit):
        self._circuit = circuit
        self._circuit_type = 'qiskit'

    def plot(self, scale=0.9, fold=200):
        fig = self._circuit.draw(output='mpl',
                                 scale=scale,
                                 fold=fold,
                                 vertical_compression='high',
                                 idle_wires=False)
        return fig

    def simu(self, simu_type='vec', meas_ops=[], is_plot=True):
        from qiskit import Aer
        backend = Aer.get_backend('statevector_simulator')
        if simu_type == 'vec':
            job = backend.run(self._circuit)
            result = job.result()
            outputstate = result.get_statevector(self._circuit, decimals=3)
            if is_plot:
                from qiskit.visualization import plot_state_city
                return plot_state_city(outputstate)
            else:
                print(outputstate)
        elif simu_type == 'exp':
            qidx = []
            simu_circuit = copy.deepcopy(self._circuit)
            for m_op in meas_ops:
                if 'Z' in m_op:
                    qidx.append(int(m_op.split('_')[1]))
                elif 'X' in m_op:
                    qidx.append(int(m_op.split('_')[1]))
                    simu_circuit.ry(-np.pi / 2, int(m_op.split('_')[1]))
                elif 'Y' in m_op:
                    qidx.append(int(m_op.split('_')[1]))
                    simu_circuit.rx(np.pi / 2, int(m_op.split('_')[1]))
            job = backend.run(simu_circuit)
            result = job.result()
            outputstate = result.get_statevector(simu_circuit, decimals=3)
            state = np.asarray(outputstate)
            return state


class StimCircuit:
    '''
    stim circuit, use for large number of qubits simulation, with all
        Clifford circuits
    ----------------------------------------------------------------
    functions:
        simu_circuit(meas_ops,is_plot,ptraced)
            meas_ops in a form of {'op1':['X_1']}
            ptracd: get all probs of a single qubit
    
    '''

    def __init__(self, circuit):
        self._circuit = circuit
        self._circuit_type = 'stim'
        self.GATE_MAP = {
            'I': 'I',
            '_I_': 'I',
            'X': 'X',
            '_X_': 'X',
            '-X': 'X',
            'X/2': 'SQRT_X',
            '-X/2': 'SQRT_X_DAG',
            'Y': 'Y',
            '_Y_': 'Y',
            '-Y': 'Y',
            'Y/2': 'SQRT_Y',
            '-Y/2': 'SQRT_Y_DAG',
            'Z': 'Z',
            'VZ': 'Z',
            '-Z': 'Z',
            'Z/2': 'SQRT_Z',
            '-Z/2': 'SQRT_Z_DAG',
            'VZ/2': 'SQRT_Z',
            '-VZ/2': 'SQRT_Z_DAG',
            'H': 'H',
            'S': 'S',
            'S_dagger': 'S_DAG',
            'CZ': 'CZ',
        }
        self.TQ_GATE = ['CX', 'CY', 'CZ', 'CNOT', 'ISWAP', 'SWAP']
        self.qnum = self._circuit.num_qubits
        self._tableau_expired = True
        self._tableau_cache = None

    @property
    def tableau(self):
        if self._tableau_expired:
            tb = stim.Tableau(self.qnum)
            for _c in self._circuit:
                gate_name = _c.name
                if gate_name in self.TQ_GATE:
                    q_idx = [t.value for t in _c.targets_copy()]
                    for i in range(int(len(q_idx) / 2)):
                        tb.append(stim.Tableau.from_named_gate(gate_name),
                                  [q_idx[2 * i], q_idx[2 * i + 1]])
                else:
                    q_idx = [t.value for t in _c.targets_copy()]
                    for i in q_idx:
                        tb.append(stim.Tableau.from_named_gate(gate_name), [i])
            self._tableau_expired = False
            self._tableau_cache = tb
        else:
            tb = self._tableau_cache
        return tb

    def expectation(self, Pauli_string, q_idxs=None):
        '''
        Pauli_string : 'XYZ...'
        '''
        _tb = self.tableau.inverse()
        q_idxs = np.arange(self.qnum) if q_idxs is None else q_idxs
        assert len(Pauli_string) == len(q_idxs)
        Pauli_string = stim.PauliString(Pauli_string)
        op = stim.PauliString('I' * self.qnum)
        for _Pauli_idx, _q_idx in zip(Pauli_string, q_idxs):
            if _Pauli_idx == 1:
                op *= _tb.x_output(_q_idx)
            elif _Pauli_idx == 2:
                op *= _tb.z_output(_q_idx)
                op *= _tb.x_output(_q_idx)
                op /= 1j
            elif _Pauli_idx == 3:
                op *= _tb.z_output(_q_idx)
        _expectation = op.sign
        for _Pauli_idx in op:
            if _Pauli_idx == 1 or _Pauli_idx == 2:
                _expectation = 0
                break
        return _expectation.real

    def simu_circuit(self, meas_ops={'op1': ['X_1']}, is_plot=True, ptraced=False):
        exp_res = {}
        for _op_name, meas_op in meas_ops.items():
            pauli_string_tmp = ''
            m_qubit_dex = []
            for m_op in meas_op:
                pauli_string_tmp += m_op.split('_')[0]
                m_qubit_dex.append(int(m_op.split('_')[1]))
            if ptraced:
                exp_res.update(
                    {_op_name: -self.expectation(pauli_string_tmp, m_qubit_dex) / 2 + 0.5})
            else:
                exp_res.update(
                    {_op_name: self.expectation(pauli_string_tmp, m_qubit_dex)})
        if is_plot:
            plot_simu_ope_res(exp_res)
        return exp_res


class CirqCircuit:
    '''
    cirq circuit, use for random circuit simulation, recommended 
        qubits less than 30
    ---------------------------------------------------------------
    functions:
        plot_circuit(scale=0.9,fold=70,t_xy=30, t_cz=60)
        simu_circuit(simu_type='exp',meas_ops={'op1':['X_1']},is_plot=True)
    
    '''

    def __init__(self, circuit, qnum):
        self._circuit = circuit
        self._circuit_type = 'cirq'
        self.qnum = qnum

    def plot(self, scale=0.9, fold=70, t_xy=30, t_cz=60):
        CircuitConverter(CirqCircuit(self._circuit, self.qnum), 'Exp').plot_circuit(scale, fold, t_xy, t_cz)

    def simu(self, simu_type='exp', meas_ops={'op1': ['X_1']},
             is_plot=True, is_vector=False, ini_state=None):
        '''
        simu_type: ['exp','vec','exp_trace','exp_tensor']
        if 'exp':
            google operation
        if 'exp_tensor': return expection value of measure operators, without partial trace
            recommend for small number of qubits
        if 'exp_trace': return expection value of measure operators, with partial
            trace, recommend for large number of qubits and small number of partial
            traced subsystem
        if 'vec': return result state vector
        '''
        cirq_simulator = qsimcirq.QSimSimulator()
        Qs = [cirq.NamedQubit('Q{}'.format(i))
              for i in range(self.qnum)][0:self.qnum]
        qubit_idx = list(np.arange(self.qnum))[0:self.qnum]
        qubit_order = [Qs[idx] for idx in qubit_idx]

        state_vector = cirq_simulator.simulate(
            self._circuit,
            qubit_order=qubit_order,
            initial_state=ini_state).final_state_vector
        if simu_type == 'vec':
            if is_plot:
                from qiskit.visualization import plot_state_city
                from qiskit.quantum_info import Statevector
                return plot_state_city(Statevector(state_vector, dims=tuple([2] * self.qnum))), state_vector
            return state_vector
        else:
            if simu_type == 'exp_trace':
                exp_res = {}
                for _op_name, meas_op in meas_ops.items():
                    ten_list = []
                    qidx = []
                    for m_op in meas_op:
                        op_name, op_idx = m_op.split('_')
                        qidx.append(int(op_idx))
                        if op_name == 'Z':
                            ten_list.append(qutip.sigmaz())
                        elif op_name == 'X':
                            ten_list.append(qutip.sigmax())
                        elif op_name == 'Y':
                            ten_list.append(qutip.sigmay())
                    exp_res.update(
                        {_op_name:
                            qutip.expect(
                                qutip.tensor(ten_list),
                                qutip.Qobj(ptrace_dense(state_vector, qidx),
                                           dims=[[2] * len(qidx), [2] * len(qidx)]))
                        })
            elif simu_type == 'exp_tensor':
                exp_res = {}
                for _op_name, meas_op in meas_ops.items():
                    ten_list = [qutip.qeye(2)] * self.qnum
                    for m_op in meas_op:
                        op_name, op_idx = m_op.split('_')
                        if op_name == 'Z':
                            ten_list[int(op_idx)] = qutip.sigmaz()
                        elif op_name == 'X':
                            ten_list[int(op_idx)] = qutip.sigmax()
                        elif op_name == 'Y':
                            ten_list[int(op_idx)] = qutip.sigmay()
                    exp_res.update(
                        {_op_name:
                            qutip.expect(
                                qutip.tensor(ten_list),
                                qutip.Qobj(state_vector, dims=[[2] * self.qnum, [1] * self.qnum]))
                        })
            elif simu_type == 'exp':
                observables = []
                for _, meas_op in meas_ops.items():
                    ops_tmp = []
                    for m_op in meas_op:
                        op_name, op_idx = m_op.split('_')
                        if op_name == 'Z':
                            ops_tmp.append(cirq.Z(Qs[int(op_idx)]))
                        elif op_name == 'X':
                            ops_tmp.append(cirq.X(Qs[int(op_idx)]))
                        elif op_name == 'Y':
                            ops_tmp.append(cirq.Y(Qs[int(op_idx)]))
                    if len(ops_tmp) == 1:
                        observables.append(ops_tmp[0])
                    else:
                        op_tmp = ops_tmp[-1]
                        for ii in range(len(ops_tmp) - 1):
                            op_tmp = op_tmp * ops_tmp[ii]
                        observables.append(op_tmp)
                res_list = cirq_simulator.simulate_expectation_values(
                    cirq.Circuit(), observables=observables, qubit_order=qubit_order,
                    initial_state=state_vector
                )
                res_list = [data_tmp.real for data_tmp in res_list]
                exp_res = {}
                for i, key in enumerate(meas_ops.keys()):
                    if '-' in key:
                        res_tmp = -res_list[i]
                    else:
                        res_tmp = res_list[i]
                    exp_res.update({key: res_tmp})
                # exp_res=dict(zip(meas_ops.keys(),res_list))
            if is_plot:
                plot_simu_ope_res(exp_res)
                return exp_res
            if is_vector:
                return state_vector, exp_res
            else:
                return exp_res


class CircuitConverter:
    '''
    I could convert circuit class from A(circuit_in) to B(circuit_type).
    You should input a circuit class as circuit_in and a str as circuit_type
    ----------------------------------------------------------------
    Here is the direct translate routine:
         _________________________________________
        |                                         | 
       \|/                                       \|/
    ----------          ----------            ----------
    |  cirq  |  ------> |   std  |  ------>   | qiskit |
    ----------          ----------            ----------
       /|\                 |  /|\                /|\
        |                  |   |                  |
        |                  |   |                  |
        |                  |   |                  |
        |                 \|/  |                  |
        |               ----------           ----------
        -------------   |   exp  |  ------>  |  stim  |
                        ----------           ----------
    
    ----------------------------------------------------------------
    But you could still translate via:
        stim -> qiskit
        
        std -> qiskit
        std -> exp
        std -> cirq
        std -> stim
        
        exp -> std
        exp -> cirq
        exp -> stim
        exp -> qiskit
        
        cirq -> exp
        cirq -> qiskit
        cirq -> std

        qiskit -> cirq
        stim -> cirq
        stim -> exp
    ---------------------------------------------------------------
    HINT:
        if the input circuit is not all qubits participate in, please 
        update kwargs with qnum = ? 
    '''

    def __init__(self, circuit_in, circuit_type=None, **kwargs):
        self._circuit_in = circuit_in
        self._circuit_type = circuit_in._circuit_type
        if circuit_type is not None:
            if self._circuit_type == 'stim' and circuit_type == 'qiskit':
                self.circuit_out = self.stim2qiskit(**kwargs)
            elif self._circuit_type == 'std' and circuit_type == 'qiskit':
                self.circuit_out = self.std2qiskit(**kwargs)
            elif self._circuit_type == 'exp' and circuit_type == 'std':
                self.circuit_out = self.exp2std()
            elif self._circuit_type == 'std' and circuit_type == 'exp':
                self.circuit_out = self.std2exp()
            elif self._circuit_type == 'exp' and circuit_type == 'qiskit':
                self.circuit_out = self.std2qiskit(circuit_in=self.exp2std(), **kwargs)
            elif self._circuit_type == 'cirq' and circuit_type == 'qiskit':
                self.circuit_out = self.cirq2qiskit()
            elif self._circuit_type == 'cirq' and circuit_type == 'std':
                self.circuit_out = self.cirq2std()
            elif self._circuit_type == 'exp' and circuit_type == 'cirq':
                self.circuit_out = self.exp2cirq()
            elif self._circuit_type == 'exp' and circuit_type == 'stim':
                self.circuit_out = self.exp2stim()
            elif self._circuit_type == 'std' and circuit_type == 'cirq':
                self.circuit_out = self.exp2cirq(circuit_in=self.std2exp(), **kwargs)
            elif self._circuit_type == 'cirq' and circuit_type == 'exp':
                self.circuit_out = self.std2exp(circuit_in=self.cirq2std(), **kwargs)
            elif self._circuit_type == 'std' and circuit_type == 'stim':
                self.circuit_out = self.exp2stim(circuit_in=self.std2exp(), **kwargs)
            elif self._circuit_type == 'qiskit' and circuit_type == 'cirq':
                self.circuit_out = self.qiskit2cirq(**kwargs)
            elif self._circuit_type == 'qiskit' and circuit_type == 'std':
                self.circuit_out = self.cirq2std(self.qiskit2cirq(**kwargs))
            elif self._circuit_type == 'qiskit' and circuit_type == 'exp':
                self.circuit_out = self.std2exp(self.cirq2std(self.qiskit2cirq(**kwargs)))
            elif self._circuit_type == 'stim' and circuit_type == 'cirq':
                self.circuit_out = self.qiskit2cirq(self.stim2qiskit(**kwargs), **kwargs)
            elif self._circuit_type == 'stim' and circuit_type == 'exp':
                self.circuit_out = self.std2exp(self.cirq2std(self.qiskit2cirq(self.stim2qiskit(**kwargs))))
            else:
                raise ValueError(f'Circuit type from {self._circuit_type} to {circuit_type} not supported!')

    def stim2qiskit(self, circuit_in=None, qnum=None):
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        if qnum is None:
            qnum = circuit_in.num_qubits
        _qiskit_circuit = qiskit.QuantumCircuit(qnum)
        for _c in circuit_in:
            name = _c.name
            if name == 'CZ':
                q_idx = [t.value for t in _c.targets_copy()]
                for i in range(int(len(q_idx) / 2)):
                    _qiskit_circuit.cz(q_idx[2 * i], q_idx[2 * i + 1])
            elif name == 'CX':
                q_idx = [t.value for t in _c.targets_copy()]
                for i in range(int(len(q_idx) / 2)):
                    _qiskit_circuit.cx(q_idx[2 * i], q_idx[2 * i + 1])
            else:
                q_idx = [t.value for t in _c.targets_copy()]
                for i in q_idx:
                    if name == 'X':
                        _qiskit_circuit.rx(np.pi, i)
                    elif name == 'SQRT_X':
                        _qiskit_circuit.rx(np.pi / 2, i)
                    elif name == 'SQRT_X_DAG':
                        _qiskit_circuit.rx(-np.pi / 2, i)
                    elif name == 'Y':
                        _qiskit_circuit.ry(np.pi, i)
                    elif name == 'SQRT_Y':
                        _qiskit_circuit.ry(np.pi / 2, i)
                    elif name == 'SQRT_Y_DAG':
                        _qiskit_circuit.ry(-np.pi / 2, i)
                    elif name == 'S':
                        _qiskit_circuit.s(i)
                    elif name == 'S_DAG':
                        _qiskit_circuit.sdg(i)
                    elif name == 'H':
                        _qiskit_circuit.h(i)
                    else:
                        raise Exception(f'uknown sq gate {name}')
        qiskit_cirq = qiskit.transpile(_qiskit_circuit,
                                       basis_gates=['cz', 'rx', 'ry', 'rz'],
                                       optimization_level=3)
        return QiskitCircuit(qiskit_cirq)

    def exp2stim(self, circuit_in=None):
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        GATE_MAP = {
            'I': 'I',
            '_I_': 'I',
            'X': 'X',
            '_X_': 'X',
            '-X': 'X',
            'X/2': 'SQRT_X',
            '-X/2': 'SQRT_X_DAG',
            'Y': 'Y',
            '_Y_': 'Y',
            '-Y': 'Y',
            'Y/2': 'SQRT_Y',
            '-Y/2': 'SQRT_Y_DAG',
            'Z': 'Z',
            'VZ': 'Z',
            '-Z': 'Z',
            'Z/2': 'SQRT_Z',
            '-Z/2': 'SQRT_Z_DAG',
            'VZ/2': 'SQRT_Z',
            '-VZ/2': 'SQRT_Z_DAG',
            'H': 'H',
            'S': 'S',
            'S_dagger': 'S_DAG',
            'CZ': 'CZ',
        }
        circuit_out = stim.Circuit()
        for i in range(len(circuit_in)):
            gate_layer = copy.deepcopy(circuit_in[i])
            add_cz = []
            for jdex, op in enumerate(gate_layer):
                if 'dcz' in op:
                    if op not in add_cz:
                        dcz_is = list(
                            map(int, op.replace('dcz', '').split('_')))
                        add_cz.append(op)
                        circuit_out.append(
                            'CZ', [np.min(dcz_is), np.max(dcz_is)])
                elif op in list(GATE_MAP.keys())[0:-1]:
                    circuit_out.append(GATE_MAP[op], [jdex])
                elif isinstance(op, tuple) and op[0] == 'PVZ':
                    raise Exception(f'No such gate operate! {jdex} - {op}')
                else:
                    raise Exception(f'No such gate operate! {jdex} - {op}')
        return StimCircuit(circuit_out)

    def std2qiskit(self, circuit_in=None, only_rxrz=True):
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        qnum = circuit_in['qnum']
        qs = qiskit.QuantumRegister(qnum, 'q')
        circuit = qiskit.QuantumCircuit(qs)
        for _layer_name, _circuit in circuit_in.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                for qidx, sq_gates in _circuit['sq'].items():
                    for sq_gate in sq_gates:
                        if sq_gate == 'H':
                            circuit.h(qidx)
                        elif sq_gate == 'S':
                            circuit.s(qidx)
                        elif sq_gate == 'S_dagger':
                            circuit.sdg(qidx)
                        elif sq_gate == 'Y':
                            if only_rxrz:
                                circuit.rz(np.pi, qidx)
                                circuit.rx(np.pi, qidx)
                            else:
                                circuit.ry(np.pi, qidx)
                        elif ((sq_gate[0] == 'Y') or (sq_gate[0] == 'Ry')) and isinstance(sq_gate[1], (int, float)):
                            if only_rxrz:
                                circuit.rz(-np.pi / 2, qidx)
                                circuit.rx(sq_gate[1], qidx)
                                circuit.rz(np.pi / 2, qidx)
                            else:
                                circuit.ry(sq_gate[1], qidx)
                        elif sq_gate == 'Z':
                            circuit.rz(np.pi, qidx)
                        elif ((sq_gate[0] == 'Z') or (sq_gate[0] == 'Rz')) and isinstance(sq_gate[1], (int, float)):
                            angle = (sq_gate[1] % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)
                            circuit.rz(angle, qidx)
                        elif sq_gate == 'Z/2':
                            circuit.rz(np.pi / 2, qidx)
                        elif sq_gate == '-Z/2':
                            circuit.rz(-np.pi / 2, qidx)
                        elif sq_gate == 'X':
                            circuit.rx(np.pi, qidx)
                        elif ((sq_gate[0] == 'X') or (sq_gate[0] == 'Rx')) and isinstance(sq_gate[1], (int, float)):
                            circuit.rx(sq_gate[1], qidx)
                        elif sq_gate == 'X/2':
                            circuit.rx(np.pi / 2, qidx)
                        elif sq_gate == '-X/2':
                            circuit.rx(-np.pi / 2, qidx)
                        elif sq_gate == 'Y/2':
                            if only_rxrz:
                                circuit.rz(-np.pi / 2, qidx)
                                circuit.rx(np.pi / 2, qidx)
                                circuit.rz(np.pi / 2, qidx)
                            else:
                                circuit.ry(np.pi / 2, qidx)
                        elif sq_gate == '-Y/2':
                            if only_rxrz:
                                circuit.rz(-np.pi / 2, qidx)
                                circuit.rx(-np.pi / 2, qidx)
                                circuit.rz(np.pi / 2, qidx)
                            else:
                                circuit.ry(-np.pi / 2, qidx)
                        else:
                            raise Exception(f'uknown sq gate {sq_gate}')
                # if len(_circuit['sq'].items()) > 0:
                #     circuit.barrier(label=str(d_all))
                #     d_all = d_all + 1
                for (q0, q1), tq_gate in _circuit['tq'].items():
                    if tq_gate == 'CX':
                        circuit.cx(q0, q1)
                    elif tq_gate == 'CY':
                        circuit.cy(q0, q1)
                    elif tq_gate == 'CZ':
                        circuit.cz(q0, q1)
                    elif tq_gate == 'CNOT':
                        circuit.cnot(q0, q1)
                    else:
                        raise Exception(f'uknown tq gate {tq_gate}')
                # if len(_circuit['tq'].items()) > 0:
                circuit.barrier()
        return QiskitCircuit(circuit)

    def exp2std(self, circuit_in=None):
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        std_cirq = {}
        std_cirq.update({'qnum': len(circuit_in[0])})
        for _layer_num in range(len(circuit_in)):
            std_cirq_tmp = {}
            sq_tmp = {}
            tq_tmp = {}
            for qidx, _op in enumerate(circuit_in[_layer_num]):
                if _op != 'I':
                    if 'dcz' in _op:
                        cname_tmp = tuple([int(i) for i in _op.replace('dcz', '').split('_')])
                        if cname_tmp not in list(tq_tmp.keys()):
                            tq_tmp.update({cname_tmp: 'CZ'})
                    elif 'CNOT' in _op:
                        cname_tmp = tuple([int(i) for i in _op.replace('CNOT', '').split('_')])
                        if cname_tmp not in list(tq_tmp.keys()):
                            tq_tmp.update({cname_tmp: 'CNOT'})
                    elif 'V' in _op:
                        if _op == 'VZ':
                            sq_tmp.update({qidx: ['Z']})
                        elif _op == 'VZ/2':
                            sq_tmp.update({qidx: ['Z/2']})
                        elif _op == '-VZ/2':
                            sq_tmp.update({qidx: ['-Z/2']})
                        else:
                            raise ValueError(f'Unknown gate name ${_op}$!')
                    elif isinstance(_op, tuple) and len(_op) == 2:
                        # if _op[0]=='VZ':
                        #     sq_tmp.update({qidx: [('Z', control_phase_in(_op[1]))]})
                        # elif 'R' in _op[0]:
                        #     sq_tmp.update({qidx: [(_op[0].replace('R', '').upper(), control_phase_in(_op[1]))]})
                        # else:
                        #     sq_tmp.update({qidx: [(_op[0], control_phase_in(_op[1]))]})
                        if _op[0] == 'VZ':
                            sq_tmp.update({qidx: [('Z', _op[1])]})
                        elif 'R' in _op[0]:
                            sq_tmp.update({qidx: [(_op[0].replace('R', '').upper(), _op[1])]})
                        else:
                            sq_tmp.update({qidx: [(_op[0], _op[1])]})
                    elif (len(_op) == 4 and _op[0] == 'PVZ') or isinstance(_op, str):
                        sq_tmp.update({qidx: [_op]})
                    else:
                        raise Exception(f'Unsupported gate ${_op}$!')
            std_cirq_tmp.update({'sq': sq_tmp})
            std_cirq_tmp.update({'tq': tq_tmp})
            std_cirq.update({f'layer{_layer_num}': std_cirq_tmp})
        std_cirq.update({'layer_num': _layer_num + 1})
        return StdCircuit(std_cirq)

    def std2exp(self, circuit_in=None):
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        qnum = circuit_in['qnum']
        circuit_output = []
        for _layer_name, _circuit in circuit_in.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                if len(list(_circuit['sq'].keys())) == 0:
                    layer_num = 1
                else:
                    layer_num = 1
                    for key in list(_circuit['sq'].keys()):
                        ops = _circuit['sq'][key]
                        if all([isinstance(op, (tuple, str)) for op in ops]) and not isinstance(ops, str):
                            layer_num = max(layer_num, len(ops))
                assert layer_num < 3, f"Unknown layer! {_circuit['sq'], {key} }"
                for idx in range(layer_num):
                    cirq_tmp = ['I'] * qnum
                    for key in list(_circuit['sq'].keys()):
                        if all([isinstance(op, (tuple, str)) for op in ops]) and not isinstance(ops, str):
                            gate_ops = _circuit['sq'][key]
                        else:
                            gate_ops = [_circuit['sq'][key]]
                        if idx == 1 and len(gate_ops) == 1:
                            continue
                        gate_op = gate_ops[idx]
                        if gate_op == 'Z' or gate_op == '-Z':
                            cirq_tmp[key] = 'VZ'
                        elif gate_op == 'Z/2':
                            cirq_tmp[key] = 'VZ/2'
                        elif gate_op == '-Z/2':
                            cirq_tmp[key] = '-VZ/2'
                        elif isinstance(gate_op, tuple) and isinstance(gate_op[1], (float, int)):
                            rot, params = gate_op[0], gate_op[1:]
                            rot = rot.replace('R', '').upper()
                            rot = 'VZ' if rot == 'Z' else rot
                            if rot not in ['PVZ']:
                                angle = params[0]
                                if angle == np.pi or angle == -np.pi:
                                    cirq_tmp[key] = f'{rot}'
                                elif angle == np.pi / 2:
                                    cirq_tmp[key] = f'{rot}/2'
                                elif angle == -np.pi / 2:
                                    cirq_tmp[key] = f'-{rot}/2'
                                elif angle == np.pi / 2 * 3:
                                    cirq_tmp[key] = f'-{rot}/2'
                                elif angle == -np.pi / 2 * 3:
                                    cirq_tmp[key] = f'{rot}/2'
                                else:
                                    cirq_tmp[key] = (rot, angle)
                            else:
                                cirq_tmp[key] = gate_op
                        else:
                            cirq_tmp[key] = gate_op
                    if idx == 0:
                        for i0, i1 in _circuit['tq'].keys():
                            if _circuit['tq'][(i0, i1)] == 'CZ':
                                cirq_tmp[i0] = f'dcz{min(i0, i1)}_{max(i0, i1)}'
                                cirq_tmp[i1] = f'dcz{min(i0, i1)}_{max(i0, i1)}'
                            elif _circuit['tq'][(i0, i1)] == 'CNOT':
                                cirq_tmp[i0] = f'CNOT{i0}_{i1}'
                                cirq_tmp[i1] = f'CNOT{i0}_{i1}'
                    # print('add gate layer', idx, '\n', key, cirq_tmp)
                    circuit_output.append(cirq_tmp)
        return ExpCircuit(circuit_output)

    def cirq2qiskit(self, circuit_in=None):
        def cirq_1q_expand_circuit_operation(circuit_operation):
            circuit = circuit_operation.sub_operation.circuit
            ops = []
            for moment in circuit:
                op = moment.operations[0]
                gate = (str(op.gate).split('(')[0], op.gate.exponent)
                ops.append(gate)
            q_idx = int(str(op.qubits[0]).split('_')[-1])
            return {q_idx: ops}

        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        qnum = len(circuit_in.all_qubits())
        circuit = {}
        qiskit_circuit = qiskit.QuantumCircuit(qnum)
        for idx, moment in enumerate(circuit_in):
            layer = 'layer%d' % idx
            circuit[layer] = {'sq': {}, 'tq': {}}
            for op in moment.operations:
                if len(op.tags) > 0:
                    ops = cirq_1q_expand_circuit_operation(op)
                    circuit[layer]['sq'].update(ops)
                    for q_idx, gates in ops.items():
                        for gate in gates:
                            getattr(qiskit_circuit,
                                    gate[0].lower())(gate[1] * np.pi, q_idx)
                else:
                    gate = op.gate
                    q_idxs = [
                        int(qubit.name.split('_')[1]) for qubit in op.qubits
                    ]
                    if str(gate).startswith('R'):
                        circuit[layer]['sq'][q_idxs[0]] = [(str(gate)[:2],
                                                            gate.exponent)]
                        getattr(qiskit_circuit,
                                str(gate)[:2].lower())(gate.exponent * np.pi,
                                                       q_idxs[0])
                    elif str(gate).startswith('CZ'):
                        circuit[layer]['tq'][tuple(q_idxs)] = 'CZ'
                        getattr(qiskit_circuit, str(gate).lower())(*q_idxs)
                    elif str(gate).startswith('CNOT'):
                        circuit[layer]['tq'][tuple(q_idxs)] = 'CNOT'
                        getattr(qiskit_circuit, str(gate).lower())(*q_idxs)
            qiskit_circuit.barrier()
        return QiskitCircuit(qiskit_circuit)

    def cirq2std(self, circuit_in=None):
        def cirq_1q_expand_circuit_operation(circuit_operation):
            circuit = circuit_operation.sub_operation.circuit
            ops = []
            for moment in circuit:
                op = moment.operations[0]
                if str(op.gate).startswith('R'):
                    gate = (str(op.gate)[1].upper(), op.gate.exponent * np.pi)
                # gate = (str(op.gate).split('(')[0], op.gate.exponent * np.pi)
                ops.append(gate)
            q_idx = int(str(op.qubits[0]).split('_')[-1])
            return {q_idx: ops}

        circuit = {}
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit['qnum'] = circuit_in.qnum
        circuit_in = circuit_in._circuit
        for idx, moment in enumerate(circuit_in):
            layer = 'layer%d' % idx
            circuit[layer] = {'sq': {}, 'tq': {}}
            for op in moment.operations:
                if len(op.tags) > 0:
                    ops = cirq_1q_expand_circuit_operation(op)
                    circuit[layer]['sq'].update(ops)
                else:
                    gate = op.gate
                    q_idxs = [int(q.name.replace('q_', '')) for q in op.qubits]
                    if str(gate).startswith('R'):
                        circuit[layer]['sq'][q_idxs[0]] = [(str(gate)[1].upper(),
                                                            gate.exponent * np.pi)]
                    elif str(gate).startswith('CZ'):
                        circuit[layer]['tq'][tuple(q_idxs)] = 'CZ'
                    elif str(gate).startswith('CNOT'):
                        circuit[layer]['tq'][tuple(q_idxs)] = 'CNOT'
        circuit['layer_num'] = idx + 1
        return StdCircuit(circuit)

    def exp2cirq(self, circuit_in=None):
        def op_to_gate(op, generator=None):
            if op in ['dcz']:
                return generator.CZ
            elif op in ['PASS', 'I', '_I_']:
                return generator.I
            elif isinstance(op, (list, tuple)) and op[0] in ['IVZ', 'IDVZ']:
                return generator.I
            else:
                if isinstance(op, str):
                    if op == 'H':
                        return generator.H
                    else:
                        angle = np.pi / 2 if '/2' in op else np.pi
                        angle = -angle if '-' in op else angle
                        gate_op = op.replace('/2', '').replace('-', '').replace('_', '')
                        if gate_op in ['VZ']:
                            gate_op = gate_op.replace('V', '')
                        if gate_op == '_Y_':
                            gate_op = 'Y'
                        sq_rot_str = 'r' + gate_op.lower()
                        return generator.__getattribute__(sq_rot_str)(angle)
                elif len(op) == 2 and isinstance(op[1], (float, int)):
                    (gate_op, angle) = op
                    if gate_op in ['VZ', '_VZ_']:
                        gate_op = gate_op.replace('V', '')
                    if gate_op.startswith('R') or gate_op.startswith('r'):
                        sq_rot_str = gate_op.lower()
                    else:
                        sq_rot_str = 'r' + gate_op.lower()
                    return generator.__getattribute__(sq_rot_str)(angle)
                elif len(op) == 4 and op[0] == 'PVZ':
                    (_, alpha, theta, phi) = op
                    return [generator.__getattribute__("rz")(phi),
                            generator.__getattribute__("rz")(-theta),
                            generator.__getattribute__("rx")(alpha),
                            generator.__getattribute__("rz")(theta)]

        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        qnum = len(circuit_in[0])
        circuit = cirq.Circuit()
        gg = cirq
        Qs = [cirq.NamedQubit('Q{}'.format(i))
              for i in range(qnum)][0:qnum]
        for Uc_ops in circuit_in:
            added_cz_gates = []
            added_cnot_gates = []
            idx_shift = 0
            for i, op in enumerate(Uc_ops):
                if 'dcz' in op:
                    if op not in added_cz_gates:
                        i0, i1 = list(
                            map(int,
                                op.replace('dcz', '').split('_')))
                        i0, i1 = min(i0, i1), max(i0, i1)
                        cz_gate = gg.FSimGate(
                            phi=np.pi, theta=0)
                        circuit.append(cz_gate(Qs[i0], Qs[i1]))
                        added_cz_gates.append(op)
                    else:
                        if op == Uc_ops.index(op, i):
                            idx_shift += 1
                elif 'CNOT' in op:
                    if op not in added_cnot_gates:
                        i0, i1 = list(
                            map(int,
                                op.replace('CNOT', '').split('_')))
                        cnot_gate = gg.CNOT
                        circuit.append(cnot_gate(Qs[i0], Qs[i1]))
                        added_cnot_gates.append(op)
                    else:
                        if op == Uc_ops.index(op, i):
                            idx_shift += 1
                else:
                    if op[0] in ['PVZ']:
                        q_gates = op_to_gate(op, generator=gg)
                        for q_gate in q_gates:
                            circuit.append(q_gate(Qs[i + idx_shift]))
                    else:
                        q_gate = op_to_gate(op, generator=gg)
                        circuit.append(q_gate(Qs[i + idx_shift]))
        return CirqCircuit(circuit, qnum)

    def qiskit2cirq(self, circuit_in=None, qnum=None):
        if circuit_in is None:
            circuit_in = self._circuit_in
        circuit_in = circuit_in._circuit
        if qnum is None:
            qnum = len(circuit_in.qubits)
        cirq_circuit = circuit_from_qasm(circuit_in.qasm())
        return CirqCircuit(cirq_circuit, qnum)


def one_dim_couping(n_qubits):
    return [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]

class ZxSimplifier:
    '''
    make ZX-calculus via pyzx, make circuit align right and merge unitaries
        via cirq.
    Put in: std circuit
    Output: std circuit 
    '''

    def __init__(self, circuit_in, do_swaps=False, is_right=True, is_left=False):
        self.do_swaps = do_swaps
        self._circuit_in = circuit_in
        self._circuit_type = circuit_in._circuit_type
        if self._circuit_type == 'qiskit':
            self._qiskuit_in = circuit_in._circuit
        elif self._circuit_type in ['exp', 'std', 'cirq']:
            self._qiskuit_in = CircuitConverter(circuit_in, 'qiskit').circuit_out._circuit
        else:
            raise Exception(f'Unsupported circuit input type${self._circuit_type}$!')
        self.qnum = self._qiskuit_in.num_qubits

        # self.circuit_out = self.layer_circuit()
        # self.circuit_out = CircuitConverter(self.layer_circuit(is_right=is_right, is_left=is_left), 'std').circuit_out
        # self._circuit_out = self.layer_circuit()

    def qiskit_circuit_zx_optimized(self, coupling_map):
        c_zx = zx.Circuit.from_qasm(self._qiskuit_in.qasm())
        c2 = zx.optimize.basic_optimization(c_zx.split_phase_gates(),
                                            do_swaps=self.do_swaps)
        c2_qasm = zx.Circuit.to_qasm(c2)
        c2_qiskit = qiskit.QuantumCircuit.from_qasm_str(c2_qasm)
        c2_qiskit_cz = qiskit.transpile(c2_qiskit,
                                        basis_gates=BASIS_GATES,
                                        coupling_map=coupling_map)
        # c2_qiskit_cz.draw('mpl')
        # plt.show()
        # return c2_qiskit_cz                       # 直接返回qiskit格式
        return QiskitCircuit(c2_qiskit_cz)  # 返回处理后的qiskit格式

    def layer_circuit(self, is_right=True, is_left=False):
        cirq_circuit = circuit_from_qasm(self.qiskit_circuit_zx_optimized()._circuit.qasm())
        cirq_circuit = cirq.merge_k_qubit_unitaries_to_circuit_op(
            cirq_circuit, k=1, merged_circuit_op_tag='PVZ')
        # cirq_circuit = cirq.align_right(cirq_circuit)
        if is_right:
            cirq_circuit = cirq.stratified_circuit(cirq_circuit[::-1],
                                                   categories=[lambda op: len(op.qubits) == 1,
                                                               lambda op: len(op.qubits) == 2]
                                                   )[::-1]
        elif is_left:
            cirq_circuit = cirq.stratified_circuit(cirq_circuit,
                                                   categories=[lambda op: len(op.qubits) == 1,
                                                               lambda op: len(op.qubits) == 2]
                                                   )
        return CirqCircuit(cirq_circuit, self.qnum)


class CircuitAligner:
    def __init__(self, circuit_in, extra_layer=None, direction=None):
        self._circuit_in = circuit_in
        self._circuit_type = circuit_in._circuit_type
        self.extra_layer = extra_layer
        if self._circuit_type != 'exp':
            self._circuit = CircuitConverter(self._circuit_in, 'exp').circuit_out._circuit
        else:
            self._circuit = self._circuit_in._circuit
        if direction in ['after', 'A', 'After', 'left', 'Left', 'L']:
            self.circuit_out = ExpCircuit(self.align_sq_after())
        elif direction in ['before', 'B', 'Before', 'right', 'Right', 'R']:
            self.circuit_out = ExpCircuit(self.align_sq_before())

    def align_sq_before(self):
        '''
        align the sq_layer circuit to the first of raw_circuit to avoid Idle time
        '''
        sq_layer = self.extra_layer
        raw_circuit = self._circuit
        qnum = len(raw_circuit[0])
        if sq_layer is not None:
            raw_circuit.insert(0, ['I'] * qnum)
            out_circuit = copy.deepcopy(raw_circuit)
            insert_dic = {}
            for i in range(qnum):
                for layer_num, _layer in enumerate(out_circuit):
                    if _layer[i] == 'I':
                        insert_dic.update({i: layer_num})
                    else:
                        break
                out_circuit[insert_dic[i]][i] = sq_layer[i]
        else:
            out_circuit = copy.deepcopy(raw_circuit)
            for i in range(qnum):
                gate_idx = -1
                gate_idx_new = -1
                layer_end = False
                for j, layer in enumerate(out_circuit):
                    if j == 0 and 'dcz' not in out_circuit[j][i] and out_circuit[j][i] != 'I':
                        gate_tmp = out_circuit[j][i]
                        gate_idx = 0
                    if j > 0 and gate_idx == 0 and layer_end is False:
                        is_cz = False
                        for k in range(qnum):
                            if 'dcz' in out_circuit[j][k]:
                                is_cz = True
                                break
                        if out_circuit[j][i] == 'I' and is_cz is False:
                            gate_idx_new = j
                        elif out_circuit[j][i] != 'I':
                            layer_end = True
                if gate_idx_new != -1:
                    out_circuit[gate_idx_new][i] = out_circuit[0][i]
                    out_circuit[0][i] = 'I'
        return out_circuit

    def align_sq_after(self):
        '''
        align the sq_layer circuit of raw_circuit or extra sq_layer to the 
        last of raw_circuit to avoid Idle time
        '''
        sq_layer = self.extra_layer
        raw_circuit = self._circuit
        out_circuit = copy.deepcopy(raw_circuit)
        if sq_layer is not None:
            out_circuit.append(sq_layer)
        qnum = len(raw_circuit[0])
        move_idx = {}
        for idx, ops in enumerate(out_circuit):
            is_cz = False
            move_idx_tmp = {}
            for j, op in enumerate(ops):
                if 'dcz' in op:
                    is_cz = True
                    if f'op_{j}' in move_idx:
                        if move_idx[f'op_{j}'] < idx:
                            del move_idx[f'op_{j}']
                    if f'I_{j}' in move_idx:
                        if move_idx[f'I_{j}'][0] < idx:
                            del move_idx[f'I_{j}']
                elif op == 'I':
                    if f'I_{j}' not in move_idx:
                        move_idx_tmp.update({f'I_{j}': [idx, 0]})
                else:
                    move_idx_tmp.update({f'op_{j}': idx})
                    if f'I_{j}' in move_idx:
                        if move_idx[f'I_{j}'][0] < idx:
                            if move_idx[f'I_{j}'][1] == 1:
                                del move_idx[f'I_{j}']
                            else:
                                move_idx.update({f'I_{j}': [move_idx[f'I_{j}'][0], 1]})
            if not is_cz:
                move_idx.update(move_idx_tmp)
        for i in range(qnum):
            if f'I_{i}' in move_idx and f'op_{i}' in move_idx:
                if move_idx[f'I_{i}'][1] == 1:
                    out_circuit[move_idx[f'I_{i}'][0]][i] = out_circuit[move_idx[f'op_{i}']][i]
                    out_circuit[move_idx[f'op_{i}']][i] = 'I'
        is_all_I = True
        for i in range(qnum):
            if out_circuit[-1][i] != 'I':
                is_all_I = False
        if is_all_I:
            del out_circuit[-1]
        return out_circuit


class CircuitAlignerNOTUSE:
    '''
    NOT USE !
    ----------------------------------------------------------------
    split single gates and two qubit gates to different layers and merge
    Put in: std circuit
    Output: std circuit 
    '''

    def __init__(self, circuit_in, dd_num=3, dd_qname='all'):
        print('WE NOT USE IT')
        self._circuit_in = circuit_in
        self._circuit_type = circuit_in._circuit_type
        if self._circuit_type != 'std':
            raise ValueError(f'Unsupported circuit type ${self._circuit_type}$, plz transfer to $std$ type!')
        self.qnum = circuit_in._circuit['qnum']
        self.circuit_out = StdCircuit(self.merge_circuit(self.split_circuit()))
        self._protected_circuit = StdCircuit(self.protected_circuit(self.circuit_out, dd_num=dd_num, dd_qname=dd_qname))
        # self.protected_circuit_out = StdCircuit(self.protected_circuit(self._circuit_in))

    def split_circuit(self):
        '''make each layer only contain single-qubit or two-qubit operations'''
        circuit_input = self._circuit_in._circuit
        splitted_circuit = {}
        splitted_circuit.update(
            {'qnum': self.qnum})
        num = 0
        split_type = 1  # even: sq - tq; odd: tq - sq
        for layer_name, layer_gates in circuit_input.items():
            if 'layer' in layer_name and layer_name != 'layer_num':
                if f'layer{num}' not in splitted_circuit:
                    splitted_circuit[f'layer{num}'] = {'sq': {}, 'tq': {}}
                sq_qnum = len(layer_gates['sq'].keys())
                tq_qnum = len(layer_gates['tq'].keys())
                if sq_qnum > 0 and tq_qnum > 0:
                    if split_type % 2:
                        splitted_circuit[f'layer{num}']['tq'].update(
                            layer_gates['tq'])
                        num += 1
                        if f'layer{num}' not in splitted_circuit:
                            splitted_circuit[f'layer{num}'] = {'sq': {}, 'tq': {}}
                        splitted_circuit[f'layer{num}']['sq'].update(
                            layer_gates['sq'])
                    else:
                        splitted_circuit[f'layer{num}']['sq'].update(
                            layer_gates['sq'])
                        num += 1
                        if f'layer{num}' not in splitted_circuit:
                            splitted_circuit[f'layer{num}'] = {'sq': {}, 'tq': {}}
                        splitted_circuit[f'layer{num}']['tq'].update(
                            layer_gates['tq'])
                    split_type += 1
                elif sq_qnum > 0:
                    splitted_circuit[f'layer{num}']['sq'].update(layer_gates['sq'])
                else:
                    splitted_circuit[f'layer{num}']['tq'].update(layer_gates['tq'])
                num += 1
        return splitted_circuit

    def merge_circuit(self, circuit):
        '''merge splitted circuit'''
        _merged_circuit = {}
        _merged_circuit['layer0'] = circuit['layer0']
        num = 0
        for layer_name, _layer in circuit.items():
            if 'layer' in layer_name and layer_name != 'layer0' and layer_name != 'layer_num':
                cur_merge_layer = _merged_circuit[f'layer{num}']
                if self.mergable(cur_merge_layer, _layer):
                    _type = self.layer_type(cur_merge_layer)
                    if _type == 'sq':
                        for key, gates in _layer[_type].items():
                            if key in _merged_circuit[f'layer{num}'][_type]:
                                _merged_circuit[f'layer{num}'][_type][key].extend(
                                    _layer[_type][key])
                            else:
                                _merged_circuit[f'layer{num}'][_type][
                                    key] = _layer[_type][key]
                    else:
                        _merged_circuit[f'layer{num}'][_type].update(
                            copy.deepcopy(_layer)[_type])
                else:
                    num += 1
                    _merged_circuit[f'layer{num}'] = copy.deepcopy(_layer)
        _merged_circuit.update({'qnum': self.qnum})
        _merged_circuit.update({'layer_num': num + 1})
        return _merged_circuit

    def mergable(self, layer1, layer2):
        types = [self.layer_type(layer1), self.layer_type(layer2)]
        if types == ['sq', 'sq'] or 'none' in types:
            return True
        elif types == ['tq', 'tq']:
            mq_idxes = []
            for tq_idxes in layer1['tq'].keys():
                for sq_idx in tq_idxes:
                    if sq_idx not in mq_idxes:
                        mq_idxes.append(sq_idx)
            for tq_idxes in layer2['tq'].keys():
                for sq_idx in tq_idxes:
                    if sq_idx in mq_idxes:
                        return False
            return True
        else:
            return False

    def layer_type(self, layer):
        '''types = ['tq','sq','both','none']'''
        if len(layer['sq']) > 0 and len(layer['tq']) > 0:
            return 'both'
        elif len(layer['sq']) > 0:
            return 'sq'
        elif len(layer['tq']) > 0:
            return 'tq'
        else:
            return 'none'

    def protected_circuit(self, circuit, protect_gate='Y', dd_num=2, dd_qname='all'):
        '''add protect gate to circuit'''
        assert protect_gate in ['X', 'Y']
        _protected_circuit = copy.deepcopy(circuit._circuit)
        idle_stack = {}
        start_flag = set()
        if dd_qname == 'all':
            dd_qnames = np.arange(0, self.qnum)
        elif isinstance(dd_qname, list):
            dd_qnames = dd_qname
        for _layer_name, _layer in circuit._circuit.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                _type = self.layer_type(_layer)
                if _type == 'sq':
                    for qname in dd_qnames:
                        if qname in _layer[_type]:
                            start_flag.add(qname)
                            if qname in idle_stack:
                                idle_stack.pop(qname)
                        else:
                            if qname in idle_stack:
                                if len(idle_stack[qname]) >= dd_num:
                                    idle_layers = idle_stack.pop(qname)
                                    _protected_circuit[
                                        idle_layers[1]][_type][qname] = [
                                        protect_gate
                                    ]
                                    _protected_circuit[_layer_name][_type][
                                        qname] = [protect_gate]
                                else:
                                    idle_stack[qname].append(_layer_name)
                            else:
                                if qname in start_flag:
                                    idle_stack[qname] = [_layer_name]
                else:
                    for cname in _layer[_type]:
                        for qname in cname:
                            start_flag.add(qname)
                            if qname in idle_stack:
                                idle_stack.pop(qname)
        return _protected_circuit


class RotationCompiler:
    '''
    make rotation merge, two layers of single qubit gates could be merged into
        a layer of PVZ gates
    Put in: std circuit
    Output: std circuit 
    '''

    def __init__(self, circuit_in, is_all=False, is_layers_combined=False):
        if isinstance(circuit_in, list):
            self.is_all = True
            self._raw_gates = copy.deepcopy(circuit_in)
            self.circuit_out = self.compile_gate()
        else:
            self._circuit_in = circuit_in
            self._circuit_type = circuit_in._circuit_type
            if self._circuit_type != 'std':
                raise ValueError(f'You have put in {self._circuit_type} type, Plz put in $std$ circuit type!')
            self.is_all = is_all
            if is_layers_combined:
                self._circuit_in = combine_std_sq_layers(circuit_in)
            self.circuit_out = self.get_circuit_compiled()

    def equal_I(self):
        '''
        when cross tq gate which don't commute with Z gate, you must check if self._quaternion == I
        '''
        return self._quaternion == QuaternionRepresentation().rotation(0, 0, 0)

    def rotation(self, alpha, theta, phi):
        self._quaternion = QuaternionRepresentation().rotation(alpha, theta, phi) * self._quaternion

    def Rx(self, alpha):
        self.rotation(alpha, np.pi / 2, 0)

    def Ry(self, alpha):
        self.rotation(alpha, np.pi / 2, np.pi / 2)

    def Rz(self, alpha):
        self.rotation(alpha, 0, 0)

    def plus_gate(self, gate):
        if gate == 'X':
            self.rotation(np.pi, np.pi / 2, 0)
        elif gate == '-X':
            self.rotation(np.pi, np.pi / 2, np.pi)
        elif gate == 'Y' or gate == '_Y_':
            self.rotation(np.pi, np.pi / 2, np.pi / 2)
        elif gate == '-Y':
            self.rotation(np.pi, np.pi / 2, np.pi * 3 / 2)
        elif gate == 'X/2':
            self.rotation(np.pi / 2, np.pi / 2, 0)
        elif gate == 'X/2+Y/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi / 4)
        elif gate == 'Y/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi / 2)
        elif gate == '-X/2+Y/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 3 / 4)
        elif gate == '-X/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi)
        elif gate == '-X/2-Y/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 5 / 4)
        elif gate == '-Y/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 3 / 2)
        elif gate == 'X/2-Y/2':
            self.rotation(np.pi / 2, np.pi / 2, np.pi * 7 / 4)
        elif gate == 'H':
            self.rotation(np.pi, 0, 0)
            self.rotation(np.pi / 2, np.pi / 2, np.pi / 2)
        elif gate == 'Z' or gate == 'VZ':
            self.rotation(np.pi, 0, 0)
        elif gate == 'Z/2' or gate == 'VZ/2':
            self.rotation(np.pi / 2, 0, 0)
        elif gate == '-Z/2' or gate == '-VZ/2':
            self.rotation(-np.pi / 2, 0, 0)
        elif gate == 'I':
            self.rotation(0, 0, 0)
        elif (gate[0] == 'Rx' or gate[0] == 'X') and isinstance(gate[1], (int, float)):
            self.Rx(gate[1])
        elif (gate[0] == 'Ry' or gate[0] == 'Y') and isinstance(gate[1], (int, float)):
            self.Ry(gate[1])
        elif (gate[0] == 'Rz' or gate[0] == 'Z' or gate[0] == 'IVZ' or gate[0] == 'VZ' or gate[
            0] == 'IDVZ') and isinstance(gate[1], (int, float)):
            self.Rz(gate[1])
        elif (gate[0] == 'PVZ') and len(gate) == 4:
            (_, alpha, theta, phi) = gate
            self.Rz(phi)
            self.Rz(-theta)
            self.Rx(alpha)
            self.Rz(theta)
        else:
            raise Exception(f'gate {gate} not supported!')

    def get_all_gates(self):
        self._quaternion = QuaternionRepresentation().rotation(0, 0, 0)
        for _gate in self._raw_gates:
            self.plus_gate(_gate)

    def compile_gate(self, compile_type='PVZ'):
        '''
        Note that Phased_X is based on pi/2 rotation, which means alphaHalf and piDphaseHalf will be used

        Parameters
        ----------
        compile type:
        1-1.'PZ':Phased_X(theta1, phi)RZ(theta2)
        1-2.'PVZ':Phased_X(theta1, phi)RZ(theta2)
        2-1.'PPZ':Phased_X(np.pi/2, phi1)Phased_X(np.pi/2, phi2)RZ(theta)
        2-2.'PPVZ':Phased_X(np.pi/2, phi1)Phased_X(np.pi/2, phi2)RZ(theta)
        3.'PP':Phased_X(theta1, phi1)Phased_X(theta2, phi2)
        4.'PPP':Phased_X(np.pi/2, phi1)Phased_X(np.pi/2, phi2)Phased_X(np.pi, phi3)

        if virtual Z can commute with every gate in circuit(CZ or CPhase), 'PVZ' and 'PPVZ' are recommended
        '''
        if self.is_all or len(self._raw_gates) != 1:
            self.get_all_gates()
            if compile_type == 'PVZ':
                euler_angles = quaternion.as_euler_angles(self._quaternion)
                euler_angles[1] = euler_angles[1] % (2 * np.pi)
                if euler_angles[1] >= np.pi:
                    euler_angles[1] = 2 * np.pi - euler_angles[1]
                    euler_angles[0] += np.pi
                    euler_angles[2] -= np.pi
                theta1 = euler_angles[1]
                theta2 = euler_angles[2] + euler_angles[0]
                phi = np.pi / 2 + euler_angles[0]
                if theta2 > np.pi:
                    theta2 = theta2 - np.pi * 2
                elif theta2 < -np.pi:
                    theta2 = theta2 + np.pi * 2
                if theta1 > np.pi:
                    theta1 = theta1 - np.pi * 2
                elif theta1 < -np.pi:
                    theta1 = theta1 + np.pi * 2
                if phi > np.pi:
                    phi = phi - np.pi * 2
                elif phi < -np.pi:
                    phi = phi + np.pi * 2
                return ('PVZ', theta1, phi, theta2)
            # elif compile_type == 'PPVZ':
            #     euler_angles = quaternion.as_euler_angles(self._quaternion)
            #     theta = euler_angles[0] + euler_angles[1] + euler_angles[2]
            #     phi1 = np.pi + euler_angles[0]
            #     phi2 = euler_angles[0] + euler_angles[1]
            #     start, _ = self._GC.plus_gate(start, 'arbitrary_vz', phaseZ=theta)
            #     start, _ = self._GC.plus_gate(start, 'X/2', dphase=phi2)
            #     start, _ = self._GC.plus_gate(start, 'X/2', dphase=phi1)
            else:
                raise Exception('No such type!')
        else:
            if len(self._raw_gates) == 1:
                if self._raw_gates[0] == 'Z':
                    return 'VZ'
                elif self._raw_gates[0] == 'Z/2':
                    return 'VZ/2'
                elif self._raw_gates[0] == '-Z/2':
                    return '-VZ/2'
                else:
                    return self._raw_gates[0]

    def get_circuit_compiled(self):
        circuit_raw = copy.deepcopy(self._circuit_in._circuit)
        qnum = circuit_raw['qnum']
        layer_num = circuit_raw['layer_num']
        circuit_out = {}
        circuit_out.update({'qnum': qnum, 'layer_num': layer_num})
        for _layer_name, _layer in circuit_raw.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                exp_cirq_tmp = {'sq': {}, 'tq': _layer['tq']}
                for qidx in range(qnum):
                    if qidx in list(_layer['sq'].keys()):
                        self._raw_gates = copy.deepcopy(_layer['sq'][qidx])
                        exp_cirq_tmp['sq'].update({qidx: self.compile_gate()})
                circuit_out.update({_layer_name: copy.deepcopy(exp_cirq_tmp)})
        return StdCircuit(circuit_out)


def combine_std_sq_layers(std_circuit):
    circuit = std_circuit._circuit
    layer_new = 0
    qnum = circuit['qnum']
    layer_num = circuit['layer_num']
    circuit_out = {}
    circuit_out.update({'qnum': qnum})
    circuit_tmp = {}
    for _layer_name, _layer in circuit.items():
        if 'layer' in _layer_name and _layer_name != 'layer_num':
            if _layer['sq'] == {} and _layer['tq'] == {}:
                pass
            elif _layer['sq'] == {}:
                if circuit_tmp != {}:
                    circuit_out.update({f'layer{layer_new}': {'sq': copy.deepcopy(circuit_tmp), 'tq': {}}})
                    layer_new += 1
                    circuit_tmp = {}
                circuit_out.update({f'layer{layer_new}': copy.deepcopy(_layer)})
                layer_new += 1
            else:
                if circuit_tmp == {}:
                    circuit_tmp = copy.deepcopy(_layer['sq'])
                else:
                    for qidx in _layer['sq'].keys():
                        if qidx not in circuit_tmp.keys():
                            circuit_tmp.update({qidx: _layer['sq'][qidx]})
                        else:
                            circuit_tmp.update({qidx: circuit_tmp[qidx] + _layer['sq'][qidx]})
    if circuit_tmp != {}:
        circuit_out.update({f'layer{layer_new}': {'sq': copy.deepcopy(circuit_tmp), 'tq': {}}})
        layer_new += 1
    circuit_out.update({'layer_num': layer_new})
    return StdCircuit(circuit_out)


def layer_type(layer):
    '''types = ['tq','sq','both','none']'''
    if len(layer['sq']) > 0 and len(layer['tq']) > 0:
        return 'both'
    elif len(layer['sq']) > 0:
        return 'sq'
    elif len(layer['tq']) > 0:
        return 'tq'
    else:
        return 'none'


def cirb(raw_circuit, qnum,
         circuit_in_type='exp', circuit_out_type='std',
         is_align_right=False, is_align_left=False,
         rotation_compile=False,
         is_dd=False, dd_num=3, dd_qname='all', dd_version='v0',
         is_do_swap=False):
    '''
    a combined function to:
        input a raw, exp circuit and make ZxSimplifier, CircuitAligner, 
            RotationCompiler, output a circuit in a form of 
            ['cirq','exp','std'], in which cirq could quickly do simulation,
            exp could run without any other operations, std could check
    '''
    if circuit_in_type == 'exp':
        exp_cirq = ExpCircuit(copy.deepcopy(raw_circuit))
    elif circuit_in_type == 'qiskit':
        exp_cirq = QiskitCircuit(copy.deepcopy(raw_circuit))
    elif circuit_in_type == 'stim':
        exp_cirq = CircuitConverter(StimCircuit(copy.deepcopy(raw_circuit)), 'exp', qnum=qnum).circuit_out
    else:
        raise Exception(f'Unsupported circuit type ${circuit_in_type}$!')
    std_cirq = ZxSimplifier(exp_cirq, do_swaps=is_do_swap,
                            is_right=is_align_right, is_left=is_align_left).circuit_out
    if rotation_compile:
        complied_cirq = RotationCompiler(std_cirq).circuit_out
    else:
        complied_cirq = std_cirq
    exp_cirq = CircuitConverter(complied_cirq, 'exp').circuit_out
    if is_dd:
        exp_cirq = dd_add(exp_cirq, 'exp', dd_num=dd_num, dd_qname=dd_qname, version=dd_version)
    if is_align_right:
        exp_cirq = CircuitAligner(exp_cirq, direction='After').circuit_out
    if circuit_out_type in ['cirq', 'std']:
        return CircuitConverter(exp_cirq, circuit_out_type).circuit_out
    elif circuit_out_type == 'exp':
        return exp_cirq
    else:
        raise Exception(f'Unsupported circuit type ${circuit_out_type}$!')


'''
below are some functions maybe ueseful for exp circuits mainly

I bless you no bug:
                         _ooOoo_
                        o8888888o
                        88" . "88
                        (| -_- |)
                         O\ = /O
                     ____/`---'\____
                    .   ' \\| |// `.
                     / \\||| : |||// \
                 / _||||| -:- |||||- \
                     | | \\\ - /// | |
                 | \_| ''\---/'' | |
                 \ .-\__ `-` ___/-. /
                 ___`. .' /--.--\ `. . __
             ."" '< `.___\_<|>_/___.' >'"".
           | | : `- \`.;`\ _ /`;.`/ - ` : | |
             \ \ `-. \_ __\ /__ _/ .-` / /
     ======`-.____`-.___\_____/___.-`____.-'======
                         `=---='

    .............................................
'''


class DynamicalDecoupling:
    '''
    Input a exp circuit and design your dd add type.
        -add_type: 'layer': add dd based on number of layers, require dd_num
        -add_type: 'time': add dd based on dd time, require t_dd, t_xy, t_cz
    '''

    def __init__(self, circuit, protect_gate='_Y_', dd_qname='all', add_type='layer', **kwargs):
        assert protect_gate in ['X', 'Y', '_Y_']
        assert add_type in ['layer', 'time']
        if len(circuit) == 1:
            dd_qname = []
        self.qnum = len(circuit[0])
        if dd_qname == 'all':
            self.dd_qnames = np.arange(0, self.qnum)
        elif isinstance(dd_qname, list):
            self.dd_qnames = dd_qname
        else:
            raise Exception(f'Unknown dd_qname ${dd_qname}$')
        self.raw_circuit = CircuitConverter(ExpCircuit(copy.deepcopy(circuit)), 'std').circuit_out
        self.protect_gate = protect_gate
        self.add_type = add_type
        if self.add_type == 'layer':
            self.dd_num = kwargs.get('dd_num', 6)
        elif self.add_type == 'time':
            self.t_dd = kwargs.get('t_dd', 200)
            self.t_xy = kwargs.get('t_xy', 24)
            self.t_cz = kwargs.get('t_cz', 50)
        self.circuit_out = self.dd_add()

    def add_dd_layer_depend(self, _layer_name, qname, idle_stack, start_flag, _protected_circuit):
        if len(idle_stack[qname]) >= self.dd_num:
            idle_layers = idle_stack.pop(qname)
            sq_layers = [idle_layer_tmp for idle_layer_tmp in idle_layers
                         if idle_layer_tmp[0] == 'xy']
            # if len(sq_layers)>1:
            #     _protected_circuit[sq_layers[1][1]]['sq'][qname] = [self.protect_gate]
            # else:
            #     _protected_circuit[sq_layers[0][1]]['sq'][qname] = [self.protect_gate]
            _protected_circuit[sq_layers[0][1]]['sq'][qname] = [self.protect_gate]
            _protected_circuit[_layer_name]['sq'][qname] = [self.protect_gate]
        else:
            idle_stack[qname].append(('xy', _layer_name))
        return idle_stack, start_flag, _protected_circuit

    def add_dd_time_depend(self, _layer_name, qname, idle_stack, start_flag, _protected_circuit):
        dd_time_tmp = 0
        for layer_tmp in idle_stack[qname]:
            if layer_tmp[0] == 'xy':
                dd_time_tmp += self.t_xy
            elif layer_tmp[0] == 'cz':
                dd_time_tmp += self.t_cz
        if dd_time_tmp >= self.t_dd:
            idle_layers = idle_stack.pop(qname)
            sq_layers = [idle_layer_tmp for idle_layer_tmp in idle_layers
                         if idle_layer_tmp[0] == 'xy']
            _protected_circuit[sq_layers[0][1]]['sq'][qname] = [self.protect_gate]
            _protected_circuit[_layer_name]['sq'][qname] = [self.protect_gate]
        else:
            idle_stack[qname].append(('xy', _layer_name))
        return idle_stack, start_flag, _protected_circuit

    def delete_final_dd(self, circuit):
        for i in range(self.qnum):
            dd_idx = []
            for j in range(len(circuit)):
                if circuit[j][i] == self.protect_gate:
                    dd_idx.append(j)
                elif circuit[j][i] != 'I':
                    dd_idx = []
            if len(dd_idx) > 0:
                for idx in dd_idx:
                    circuit[idx][i] = 'I'
        return ExpCircuit(circuit)

    def dd_add(self):
        _raw_circuit = self.raw_circuit._circuit
        idle_stack = {}
        start_flag = set()
        _protected_circuit = copy.deepcopy(_raw_circuit)
        for _layer_name, _layer in _raw_circuit.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                _type = layer_type(_layer)
                if _type == 'sq':
                    for qname in self.dd_qnames:
                        if qname in _layer[_type]:
                            start_flag.add(qname)
                            if qname in idle_stack:
                                idle_stack.pop(qname)
                        else:
                            if qname in idle_stack:
                                if self.add_type == 'layer':
                                    idle_stack, start_flag, _protected_circuit = self.add_dd_layer_depend(
                                        _layer_name, qname,
                                        idle_stack, start_flag,
                                        _protected_circuit)
                                elif self.add_type == 'time':
                                    idle_stack, start_flag, _protected_circuit = self.add_dd_time_depend(
                                        _layer_name, qname,
                                        idle_stack, start_flag,
                                        _protected_circuit)
                            else:
                                if qname in start_flag:
                                    idle_stack[qname] = [('xy', _layer_name)]
                elif _type == 'tq':
                    tq_qnames = []
                    for cname in _layer[_type]:
                        for qname in cname:
                            if qname in self.dd_qnames and qname not in tq_qnames:
                                tq_qnames.append(qname)
                    for qname in self.dd_qnames:
                        if qname in tq_qnames:
                            start_flag.add(qname)
                            if qname in idle_stack:
                                idle_stack.pop(qname)
                        else:
                            if qname in idle_stack:
                                idle_stack[qname].append(('cz', _layer_name))
                            else:
                                if qname in start_flag:
                                    idle_stack[qname] = [('cz', _layer_name)]
                elif _type == 'both':
                    raise Exception("Not support yet for layer type both !")
        exp_circuit = CircuitConverter(StdCircuit(_protected_circuit), 'exp').circuit_out
        return self.delete_final_dd(exp_circuit._circuit)


def dd_add(raw_circuit, circuit_type='exp',
           dd_num=6, dd_qname='all', version='v1',
           protect_gate='_Y_', is_rename=False, **kwargs):
    '''
    a quick function to:
        input a raw, exp circuit and make dynamic decoupling with
        dd_num: the density of dd, 2: max; 3: middle; 4: min
        dd_qname: add dd on which qubit? if 'all' then add on all 
            qubits.
    ----------------------------------------------------------------
    version:
        v0: periodic dynamical decoupling
        v1: as less gates add as possible
        
    '''
    qnum = len(raw_circuit[0])

    def protected_circuit_v0(circuit, protect_gate='Y'):
        '''add protect gate to circuit'''
        assert protect_gate in ['X', 'Y', '_Y_']
        _protected_circuit = circuit._circuit
        idle_stack = {}
        start_flag = set()
        if dd_qname == 'all':
            dd_qnames = np.arange(0, qnum)
        elif isinstance(dd_qname, list):
            dd_qnames = dd_qname
        for _layer_name, _layer in circuit._circuit.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                _type = layer_type(_layer)
                if _type == 'sq':
                    for qname in dd_qnames:
                        if qname in _layer[_type]:
                            start_flag.add(qname)
                            if qname in idle_stack:
                                idle_stack.pop(qname)
                        else:
                            if qname in idle_stack:
                                if len(idle_stack[qname]) >= dd_num:
                                    idle_layers = idle_stack.pop(qname)
                                    _protected_circuit[
                                        idle_layers[0]][_type][qname] = [
                                        protect_gate
                                    ]
                                    _protected_circuit[_layer_name][_type][
                                        qname] = [protect_gate]
                                else:
                                    idle_stack[qname].append(_layer_name)
                            else:
                                if qname in start_flag:
                                    idle_stack[qname] = [_layer_name]
                elif _type == 'tq':
                    tq_qnames = []
                    for cname in _layer[_type]:
                        for qname in cname:
                            if qname in dd_qnames and qname not in tq_qnames:
                                tq_qnames.append(qname)
                    for qname_i in range(len(dd_qnames)):
                        if dd_qnames[qname_i] in tq_qnames:
                            start_flag.add(dd_qnames[qname_i])
                            if dd_qnames[qname_i] in idle_stack:
                                idle_stack.pop(dd_qnames[qname_i])
                        else:
                            if dd_qnames[qname_i] in idle_stack:
                                idle_stack[dd_qnames[qname_i]].append(_layer_name)
                elif _type == 'both':
                    for cname in _layer[_type]:
                        for qname in cname:
                            if qname in dd_qnames:
                                start_flag.add(qname)
                                if qname in idle_stack:
                                    idle_stack.pop(qname)
        return _protected_circuit

    def protected_circuit_v1(circuit, protect_gate='Y'):
        assert protect_gate in ['X', 'Y', '_Y_']
        _protected_circuit = copy.deepcopy(circuit._circuit)
        idle_stack = {}
        start_flag = set()

        if dd_qname == 'all':
            dd_qnames = np.arange(0, qnum)
        elif isinstance(dd_qname, list):
            dd_qnames = dd_qname

        def pop_idle_stack(qname):
            if qname in idle_stack:
                idle_layers = idle_stack.pop(qname)
                if qname == 'q9_15':
                    if len(idle_layers) >= 2:
                        for i in range(int(len(idle_layers) / 2)):
                            _protected_circuit[idle_layers[2 *
                                                           i]]['sq'][qname] = [
                                protect_gate
                            ]
                            _protected_circuit[idle_layers[2 * i +
                                                           1]]['sq'][qname] = [
                                protect_gate
                            ]
                else:
                    if len(idle_layers) >= dd_num:
                        if (len(idle_layers) - 2) % 4 < 2:
                            protect_layer_idx = [
                                int((len(idle_layers) - 2) / 4),
                                int((len(idle_layers) - 2) / 4) * 3 + 1
                            ]
                        else:
                            protect_layer_idx = [
                                int((len(idle_layers) - 2) / 4),
                                int((len(idle_layers) - 2) / 4) * 3 + 2
                            ]
                        _protected_circuit[idle_layers[
                            protect_layer_idx[0]]]['sq'][qname] = [
                            protect_gate
                        ]
                        _protected_circuit[idle_layers[
                            protect_layer_idx[1]]]['sq'][qname] = [
                            protect_gate
                        ]

        for _layer_name, _layer in circuit._circuit.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                _type = layer_type(_layer)
                if _type == 'sq':
                    for qname in dd_qnames:
                        if qname in _layer[_type]:
                            start_flag.add(qname)
                            pop_idle_stack(qname)
                        else:
                            if qname in idle_stack:
                                idle_stack[qname].append(_layer_name)
                            else:
                                if qname in start_flag:
                                    idle_stack[qname] = [_layer_name]
                else:
                    for cname in _layer[_type]:
                        for qname in cname:
                            if qname in dd_qnames:
                                start_flag.add(qname)
                                pop_idle_stack(qname)
        return _protected_circuit

    def protected_circuit_v2(circuit, protect_gate='Y', **kwargs):
        assert protect_gate in ['X', 'Y', '_Y_']

        t_sq = kwargs.get('t_sq', 24)
        t_cz = kwargs.get('t_cz', 50)
        dd_time = kwargs.get('dd_time', 200)

        def cal_dd_time(idle_stack):
            return idle_stack

        _protected_circuit = circuit._circuit
        idle_stack = {}
        start_flag = set()
        if dd_qname == 'all':
            dd_qnames = np.arange(0, qnum)
        elif isinstance(dd_qname, list):
            dd_qnames = dd_qname
        for _layer_name, _layer in circuit._circuit.items():
            if 'layer' in _layer_name and _layer_name != 'layer_num':
                _type = layer_type(_layer)
                if _type == 'sq':
                    for qname in dd_qnames:
                        if qname in _layer[_type]:
                            start_flag.add(qname)
                            if qname in idle_stack:
                                idle_stack.pop(qname)
                        else:
                            if qname in idle_stack:
                                if len(idle_stack[qname]) > dd_num or cal_dd_time(idle_stack[qname]) > dd_time:
                                    idle_layers = idle_stack.pop(qname)
                                    _protected_circuit[
                                        idle_layers[1]][_type][qname] = [protect_gate]
                                    _protected_circuit[_layer_name][_type][qname] = [protect_gate]
                                else:
                                    idle_stack[qname].append(_layer_name)
                            else:
                                if qname in start_flag:
                                    idle_stack[qname] = [_layer_name]
                elif _type == 'tq':
                    tq_qnames = []
                    for cname in _layer[_type]:
                        for qname in cname:
                            if qname in dd_qnames and qname not in tq_qnames:
                                tq_qnames.append(qname)
                    for qname_i in range(len(dd_qnames)):
                        if dd_qnames[qname_i] in tq_qnames:
                            start_flag.add(dd_qnames[qname_i])
                            if dd_qnames[qname_i] in idle_stack:
                                idle_stack.pop(dd_qnames[qname_i])
                        else:
                            if dd_qnames[qname_i] in idle_stack:
                                idle_stack[dd_qnames[qname_i]].append(_layer_name)
                elif _type == 'both':
                    for cname in _layer[_type]:
                        for qname in cname:
                            if qname in dd_qnames:
                                start_flag.add(qname)
                                if qname in idle_stack:
                                    idle_stack.pop(qname)
        return _protected_circuit

    def delete_final_dd(circuit, is_rename=False):
        for i in range(qnum):
            dd_idx = []
            for j in range(len(circuit)):
                if circuit[j][i] == protect_gate:
                    dd_idx.append(j)
                elif circuit[j][i] != 'I':
                    dd_idx = []
            if len(dd_idx) > 0:
                for idx in dd_idx:
                    circuit[idx][i] = 'I'
        if is_rename:
            for i in range(qnum):
                for j in range(len(circuit)):
                    if circuit[j][i] == protect_gate:
                        circuit[j][i] = protect_gate[1]
        return ExpCircuit(circuit)

    if len(raw_circuit) == 1:
        return ExpCircuit(raw_circuit)
    else:
        exp_cirq = ExpCircuit(copy.deepcopy(raw_circuit))
        std_cirq = CircuitConverter(exp_cirq, 'std').circuit_out
        if version == 'v0':
            dd_cirq = StdCircuit(protected_circuit_v0(std_cirq, protect_gate=protect_gate))
        elif version == 'v1':
            dd_cirq = StdCircuit(protected_circuit_v1(std_cirq, protect_gate=protect_gate))
        elif version == 'v2':
            dd_cirq = StdCircuit(protected_circuit_v2(std_cirq, protect_gate=protect_gate, **kwargs))

        exp_cirq = CircuitConverter(dd_cirq, 'exp').circuit_out._circuit
        dd_cirq = delete_final_dd(exp_cirq, is_rename)

        if circuit_type in ['cirq', 'std']:
            return CircuitConverter(dd_cirq, circuit_type).circuit_out
        elif circuit_type == 'exp':
            return dd_cirq
        else:
            raise Exception(f'Unsupported circuit type ${circuit_type}$!')


def circuit_merge_tqs(raw_circuit, c_smt_list):
    raw_circuit = copy.deepcopy(raw_circuit)
    tqs_list_tmp = []
    circuit_out = []
    for layer_num, ops in enumerate(raw_circuit):
        for op in ops:
            if 'dcz' in op:
                is_cz = True
                if layer_num not in tqs_list_tmp:
                    tqs_list_tmp.append(layer_num)
            elif op == 'I':
                pass
            else:
                is_cz = False
        if not is_cz:
            if len(tqs_list_tmp) > 0:
                circuit_out.extend(merge_tqs(raw_circuit[tqs_list_tmp[0]:tqs_list_tmp[-1] + 1], c_smt_list))
                tqs_list_tmp = []
            circuit_out.append(raw_circuit[layer_num])
    return circuit_out


def merge_tqs(raw_circuit, c_smt_list):
    out_circuit = []
    qnum = len(raw_circuit[0])
    cz_all = []
    for ops in raw_circuit:
        cz_layer = []
        for op in ops:
            if 'dcz' in op and op not in cz_layer:
                cz_layer.append(op)
        cz_all.append(set(cz_layer))
    if len(cz_all) == 1:
        return raw_circuit
    else:
        print('---------------')
        print('before')
        print(cz_all)
        for i in range(len(cz_all)):
            for c_smt in c_smt_list:
                if cz_all[i] < c_smt:
                    for j in np.arange(i + 1, len(cz_all)):
                        for op in copy.deepcopy(cz_all[j]):
                            if op in c_smt:
                                cz_all[i].add(op)
                                cz_all[j].remove(op)
        print('after')
        print(cz_all)
        for cz_layer in cz_all:
            out_circuit_tmp = ['I'] * qnum
            for i in range(qnum):
                for op in cz_layer:
                    if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                        out_circuit_tmp[i] = op
            out_circuit.append(out_circuit_tmp)
    return out_circuit


def circuit_split_tqs(raw_circuit, c_env_dic):
    raw_circuit = copy.deepcopy(raw_circuit)
    _circuit = []
    for _layer_num, ops in enumerate(raw_circuit):
        is_cz = False
        for op in ops:
            if 'dcz' in op:
                is_cz = True
        if is_cz:
            _circuit.extend(split_tqs(ops, c_env_dic, _layer_num))
        else:
            _circuit.append(ops)
    return _circuit


def split_tqs(tq_layer, c_env_dic, layer_num=0):
    '''
    tq_layer: a list of ops
    c_env_dic: dict of {op: [ops]}
    '''
    tqs = []
    qnum = len(tq_layer)
    for op in tq_layer:
        if op not in tqs and 'dcz' in op:
            tqs.append(op)
    group_1 = []
    group_2 = []
    group_3 = []
    group_4 = []
    for op in c_env_dic.keys():
        if op in tqs:
            is_pop = False
            for op_avoid in c_env_dic[op]:
                if op_avoid in tqs:
                    is_pop = True
                    tqs.remove(op_avoid)
                    group_2.append(op_avoid)
            if is_pop:
                tqs.remove(op)
                group_1.append(op)
    group_1.extend(tqs)
    for op in c_env_dic.keys():
        if op in group_2:
            is_pop = False
            for op_avoid in c_env_dic[op]:
                if op_avoid in group_2:
                    is_pop = True
                    group_2.remove(op_avoid)
                    group_3.append(op_avoid)
            if is_pop:
                group_2.append(op)
    for op in c_env_dic.keys():
        if op in group_3:
            is_pop = False
            for op_avoid in c_env_dic[op]:
                if op_avoid in group_3:
                    is_pop = True
                    group_3.remove(op_avoid)
                    group_4.append(op_avoid)
            if is_pop:
                group_3.append(op)
    for op in c_env_dic.keys():
        if op in group_4:
            for op_avoid in c_env_dic[op]:
                if op_avoid in group_4:
                    print(f'layer num: {layer_num}')
                    print(f'group4: {op_avoid} is avoid because of {op}')
    if len(group_1) == 0:
        out_circuit_tmp_1 = ['I'] * qnum
        for i in range(qnum):
            for op in tqs:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_1[i] = op
        out_circuit = [out_circuit_tmp_1]
    elif len(group_2) == 0:
        out_circuit_tmp_1 = ['I'] * qnum
        for i in range(qnum):
            for op in group_1:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_1[i] = op
        out_circuit = [out_circuit_tmp_1]
    elif len(group_3) == 0:
        out_circuit_tmp_1 = ['I'] * qnum
        out_circuit_tmp_2 = ['I'] * qnum
        for i in range(qnum):
            for op in group_1:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_1[i] = op
            for op in group_2:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_2[i] = op
        out_circuit = [out_circuit_tmp_1, out_circuit_tmp_2]
    elif len(group_4) == 0:
        out_circuit_tmp_1 = ['I'] * qnum
        out_circuit_tmp_2 = ['I'] * qnum
        out_circuit_tmp_3 = ['I'] * qnum
        for i in range(qnum):
            for op in group_1:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_1[i] = op
            for op in group_2:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_2[i] = op
            for op in group_3:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_3[i] = op
        out_circuit = [out_circuit_tmp_1, out_circuit_tmp_2, out_circuit_tmp_3]
    else:
        out_circuit_tmp_1 = ['I'] * qnum
        out_circuit_tmp_2 = ['I'] * qnum
        out_circuit_tmp_3 = ['I'] * qnum
        out_circuit_tmp_4 = ['I'] * qnum
        for i in range(qnum):
            for op in group_1:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_1[i] = op
            for op in group_2:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_2[i] = op
            for op in group_3:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_3[i] = op
            for op in group_4:
                if i in [int(op.split('dcz')[1].split('_')[0]), int(op.split('dcz')[1].split('_')[1])]:
                    out_circuit_tmp_4[i] = op
        out_circuit = [out_circuit_tmp_1, out_circuit_tmp_2, out_circuit_tmp_3, out_circuit_tmp_4]
        # print('I cannot make decision, but here is my suggestion:')
        # print(f'layer num: {layer_num}')
        # print(f'group_1:{group_1}; group_2:{group_2}; group_3:{group_3}; extra:{tqs}')
        # raise Exception()
    return out_circuit


def replace_tqs(raw_circuit, target_circuit):
    out_circuit = []
    layer_num = 0
    for _layer in raw_circuit:
        is_cz_raw = False
        for op in _layer:
            if 'dcz' in op:
                is_cz_raw = True
                break
        if not is_cz_raw:
            out_circuit.append(_layer)
            layer_num += 1
        else:
            for _ in range(100):
                is_cz = False
                for op in target_circuit[layer_num]:
                    if 'dcz' in op:
                        is_cz = True
                        break
                if is_cz:
                    out_circuit.append(target_circuit[layer_num])
                    layer_num += 1
                else:
                    break
    # ExpCircuit(out_circuit).plot_circuit(fold=170)
    ExpCircuit(out_circuit).print_circuit()


def merge_sq_layer(layer_a, layer_b):
    '''
    rotation compiler merge layers of single qubits
    '''
    assert len(layer_a) == len(layer_b)
    circuit_out = ['I'] * len(layer_a)
    for i, op in enumerate(layer_a):
        if layer_a[i] == 'I':
            circuit_out[i] = layer_b[i]
        elif layer_b[i] == 'I':
            circuit_out[i] = layer_a[i]
        else:
            circuit_out[i] = RotationCompiler([layer_a[i], layer_b[i]]).circuit_out
    return circuit_out


def merge_end_op(raw_circuit, ops, qidxs,
                 is_align_left=True, rotation_compiler=False):
    '''
    merge end op in the end of raw_circuit, with op be a list or a string (together with qidx be a list or a string). 
    
    is_align_left - align op as left as possible to avoid idle time
    
    rotation_compiler - merge op in the circuit's PVZ gate or single qubit gate as possible to cost less gates.
    
    qidx - qubit idx corresponding to op
    '''

    def check_is_cz(layer):
        is_cz = False
        for op in layer:
            if 'dcz' in op:
                is_cz = True
                break
        return is_cz

    if isinstance(ops, str):
        ops = [ops]
    if isinstance(qidxs, str):
        qidxs = [qidxs]
    qnum = len(raw_circuit[0])
    circuit_out = copy.deepcopy(raw_circuit)

    for i, op in enumerate(ops):
        if op == 'I':
            pass
        else:
            is_cz = check_is_cz(circuit_out[-1])
            if is_cz or (circuit_out[-1][qidxs[i]] != 'I' and rotation_compiler is False):
                extra_layer = ['I'] * qnum
                extra_layer[qidxs[i]] = op
                circuit_out.append(extra_layer)
            elif circuit_out[-1][qidxs[i]] != 'I' and rotation_compiler is True:
                circuit_out[-1][qidxs[i]] = RotationCompiler([circuit_out[-1][qidxs[i]], op]).circuit_out
            else:
                gate_align_idx = [-1]
                for j in np.arange(len(circuit_out) - 2, -1, -1):
                    is_cz = check_is_cz(circuit_out[j])
                    if circuit_out[j][qidxs[i]] == 'I':
                        if is_cz is False:
                            gate_align_idx.append(j)
                    else:
                        if is_align_left and rotation_compiler:
                            if is_cz is False:
                                circuit_out[j][qidxs[i]] = RotationCompiler([circuit_out[j][qidxs[i]], op]).circuit_out
                            else:
                                circuit_out[gate_align_idx[-1]][qidxs[i]] = op
                            break
                        if is_align_left:
                            circuit_out[gate_align_idx[-1]][qidxs[i]] = op
                            break
                else:
                    circuit_out[-1][qidxs[i]] = op
    return circuit_out


def del_ops(raw_circuit, ops):
    circuit_out = copy.deepcopy(raw_circuit)
    for i, layer in enumerate(circuit_out):
        for j, op in enumerate(layer):
            if op in ops:
                circuit_out[i][j] = 'I'
    return circuit_out


'''
    Below are some QST functions which would take full advantage of circuit,
i.e., make least idle time and cost less gates as possible.
'''


def Rmat(axis, angle):
    return expm(-1j * angle / 2.0 * axis)


def dot3(A, B, C):
    """Compute the dot product of three matrices"""
    return np.dot(np.dot(A, B), C)


def tensor(matrices):
    """Compute the tensor product of a list (or array) of matrices"""
    return reduce(np.kron, matrices)


def tensor_combinations(matrices, repeat):
    return [tensor(ms) for ms in itertools.product(matrices, repeat=repeat)]


def str_combinations(string_list, repeat):
    return [list(ms) for ms in itertools.product(string_list, repeat=repeat)]


sigmaI = np.eye(2, dtype=complex)
sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)

sigmaP = (sigmaX - 1j * sigmaY) / 2
sigmaM = (sigmaX + 1j * sigmaY) / 2

Xpi2 = Rmat(sigmaX, np.pi / 2)
Ypi2 = Rmat(sigmaY, np.pi / 2)
Zpi2 = Rmat(sigmaZ, np.pi / 2)

Xpi = Rmat(sigmaX, np.pi)
Ypi = Rmat(sigmaY, np.pi)
Zpi = Rmat(sigmaZ, np.pi)

Xmpi2 = Rmat(sigmaX, -np.pi / 2)
Ympi2 = Rmat(sigmaY, -np.pi / 2)
Zmpi2 = Rmat(sigmaZ, -np.pi / 2)

Xmpi = Rmat(sigmaX, -np.pi)
Ympi = Rmat(sigmaY, -np.pi)
Zmpi = Rmat(sigmaZ, -np.pi)


class Qst:
    def __init__(self, circuit_in=[], tomo_qidx=[], tomo_basis=['I', 'X/2', 'Y/2']):
        self.circuit = copy.deepcopy(circuit_in)
        self.qnum = len(tomo_qidx)
        self.tomo_qidx = tomo_qidx
        self.tomo_basis = tomo_basis
        self.tomo_ops = self.str2ops(tomo_basis)

    def str2ops(self, tomo_basis):
        tomo_ops = []
        for tomo_str in tomo_basis:
            if tomo_str == 'I':
                tomo_ops.append(sigmaI)
            elif tomo_str == 'X/2':
                tomo_ops.append(Xpi2)
            elif tomo_str == 'Y/2':
                tomo_ops.append(Ypi2)
            elif tomo_str == 'Z/2':
                tomo_ops.append(Zpi2)
            elif tomo_str == 'X':
                tomo_ops.append(Xpi)
            elif tomo_str == 'Y':
                tomo_ops.append(Ypi)
            elif tomo_str == 'Z':
                tomo_ops.append(Zpi)
        return tomo_ops

    def gen_qst_circuits(self, is_align_left=False, rotation_compiler=False):
        circuits = []
        self.tomo_strs = str_combinations(self.tomo_basis, self.qnum)
        for tomo_str in self.tomo_strs:
            # print(tomo_str)
            circuit_tmp = merge_end_op(self.circuit,
                                       tomo_str,
                                       self.tomo_qidx,
                                       is_align_left=is_align_left,
                                       rotation_compiler=rotation_compiler)
            circuits.append(copy.deepcopy(circuit_tmp))
        return circuits

    def get_qst_A(self):
        '''
        Us - a list of unitary operations that will be applied to the
            state before measuring the diagonal elements.  These unitaries
            should form a 'complete' set to allow the full density matrix
            to be determined, though this is not enforced.
        
        Returns a transformation matrix that should be passed to qst along
        with measurement data to perform the state tomography.
        '''
        self.Us = tensor_combinations(self.tomo_ops, self.qnum)
        Us = np.asarray(self.Us)

        M = len(Us)  # number of different measurements
        N = len(Us[0])  # number of states (= number of diagonal elements)

        if N <= 16:
            # 1-4 qubits
            def transform(K, L):
                i, j = divmod(K, N)
                m, n = divmod(L, N)
                return Us[i, j, m] * Us[i, j, n].conj()

            U = np.fromfunction(transform, (M * N, N ** 2), dtype=int)
        else:
            # 5+ qubits
            U = np.zeros((M * N, N ** 2), dtype=complex)
            for K in range(M * N):
                for L in range(N ** 2):
                    i, j = divmod(K, N)
                    m, n = divmod(L, N)
                    U[K, L] = Us[i, j, m] * Us[i, j, n].conj()
        return U

    def qst(self, diags, U):
        """Convert a set of diagonal measurements into a density matrix.
        
        diags - measured probabilities (diagonal elements) after acting
            on the state with each of the unitaries from the qst protocol
        
        U - transformation matrix from init_qst for this protocol, or 
            key passed to init_qst under which the transformation was saved
        """
        diags = np.asarray(diags)
        N = diags.shape[1]
        rhoFlat, resids, rank, s = np.linalg.lstsq(U, diags.flatten(), rcond=None)

        return rhoFlat.reshape((N, N))


'''
below are some test functions
'''


def stupid_dd_add(raw_circuit, dd_qname='all', protect_gate='Y'):
    '''
    All my experience (Berret).
    ----------------------------------------------------------------
    input: exp circuit
    output: exp circuit
    ----------------------------------------------------------------
    main rules: 
        1. protect gate in combinations of 2 or 4, with the side two 
        gates away from each other as far as possible.
        2. if idle time is too long, fill them up with 4, take
        residue 2
    '''
    assert protect_gate in ['X', 'Y']
    qnum = len(raw_circuit[0])
    exp_cirq = ExpCircuit(copy.deepcopy(raw_circuit))
    std_cirq = CircuitConverter(exp_cirq, 'std').circuit_out._circuit

    idle_stack = {}
    start_flag = set()
    if dd_qname == 'all':
        dd_qnames = np.arange(0, qnum)
    elif isinstance(dd_qname, list):
        dd_qnames = dd_qname

    for _layer_name, _layer in std_cirq.items():
        if 'layer' in _layer_name and _layer_name != 'layer_num':
            _type = layer_type(_layer)
            if _type == 'sq':
                for qname in dd_qnames:
                    if qname in _layer[_type]:
                        start_flag.add(qname)
                        if qname in idle_stack:
                            idle_layers = idle_stack.pop(qname)
                            sq_layers = []
                            tq_layers = []
                            for layer in idle_layers:
                                if layer_type(std_cirq[layer]) == 'sq':
                                    sq_layers.append(layer)
                                elif layer_type(std_cirq[layer]) == 'tq':
                                    tq_layers.append(layer)
                            delta_layers = int(sq_layers[-1].split('layer')[1]) - int(sq_layers[0].split('layer')[1])
                            delta_layers_all = int(idle_layers[-1].split('layer')[1]) - int(
                                idle_layers[0].split('layer')[1])
                            if delta_layers_all >= 4 and delta_layers <= 9 and (len(sq_layers) >= 3):
                                std_cirq[sq_layers[0]]['sq'][qname] = [
                                    protect_gate]
                                std_cirq[sq_layers[-1]]['sq'][qname] = [
                                    protect_gate]
                            elif delta_layers >= 10 and delta_layers <= 13:
                                if len(sq_layers) >= 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[1]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[-1]]['sq'][qname] = [
                                        protect_gate]
                            elif delta_layers > 13 and delta_layers < 18:
                                if len(sq_layers) == 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[1]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[-1]]['sq'][qname] = [
                                        protect_gate]
                                elif len(sq_layers) > 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[3]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[-1]]['sq'][qname] = [
                                        protect_gate]
                    else:
                        if qname in idle_stack:
                            sq_layers = []
                            for layer in idle_stack[qname]:
                                if layer_type(std_cirq[layer]) == 'sq':
                                    sq_layers.append(layer)
                            delta_layers = int(_layer_name.split('layer')[1]) - int(sq_layers[0].split('layer')[1])
                            if delta_layers >= 18:
                                if len(sq_layers) == 3:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[1]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[_layer_name]['sq'][qname] = [
                                        protect_gate]
                                elif len(sq_layers) >= 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[3]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[_layer_name]['sq'][qname] = [
                                        protect_gate]
                                idle_stack.pop(qname)
                            else:
                                idle_stack[qname].append(_layer_name)
                        else:
                            if qname in start_flag:
                                idle_stack[qname] = [_layer_name]
            elif _type == 'tq':
                tq_qnames = []
                for cname in _layer[_type]:
                    for qname in cname:
                        if qname in dd_qnames and qname not in tq_qnames:
                            tq_qnames.append(qname)
                for qname in dd_qnames:
                    if qname in tq_qnames:
                        start_flag.add(qname)
                        if qname in idle_stack:
                            idle_layers = idle_stack.pop(qname)
                            sq_layers = []
                            tq_layers = []
                            for layer in idle_layers:
                                if layer_type(std_cirq[layer]) == 'sq':
                                    sq_layers.append(layer)
                                elif layer_type(std_cirq[layer]) == 'tq':
                                    tq_layers.append(layer)
                            delta_layers = int(sq_layers[-1].split('layer')[1]) - int(sq_layers[0].split('layer')[1])
                            delta_layers_all = int(idle_layers[-1].split('layer')[1]) - int(
                                idle_layers[0].split('layer')[1])
                            if delta_layers_all >= 4 and delta_layers <= 9 and (len(sq_layers) >= 3):
                                std_cirq[sq_layers[0]]['sq'][qname] = [
                                    protect_gate]
                                std_cirq[sq_layers[-1]]['sq'][qname] = [
                                    protect_gate]
                            elif delta_layers >= 10 and delta_layers <= 13:
                                if len(sq_layers) >= 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[1]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[-1]]['sq'][qname] = [
                                        protect_gate]
                            elif delta_layers > 13 and delta_layers < 18:
                                if len(sq_layers) == 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[1]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[-1]]['sq'][qname] = [
                                        protect_gate]
                                elif len(sq_layers) > 4:
                                    std_cirq[sq_layers[0]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[2]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[3]]['sq'][qname] = [
                                        protect_gate]
                                    std_cirq[sq_layers[-1]]['sq'][qname] = [
                                        protect_gate]
                    else:
                        if qname in idle_stack:
                            idle_stack[qname].append(_layer_name)
            elif _type == 'both':
                raise ValueError('Not supported yet')
    return CircuitConverter(StdCircuit(std_cirq), 'exp').circuit_out._circuit


def serialize_circuit(opss, exclude_smt_list=None, others_idx=None):
    """
    exclude_smt_list:
        None: not serialize any cz gates
        []: serialized all cz gates
    others_idx: 
        <int>: to keep other gates at the No.<others_idx> serialized smt_layers
        None: delay other gates after the serialzed smt layers
    """
    # 把被包含的set放在后面使用
    exclude_smt_list = copy.deepcopy(exclude_smt_list)
    new_set_list = []
    for c_smt in exclude_smt_list:
        if any([set(c_smt) > cset_exist for cset_exist in new_set_list]):
            new_set_list.insert(0, set(c_smt))
        elif any([set(c_smt) == cset_exist for cset_exist in new_set_list]):
            continue
        # elif any([set(c_smt) < cset_exist for cset_exist in new_set_list]):
        #     new_set_list.append(set(c_smt))
        else:
            new_set_list.append(set(c_smt))
    # print(new_set_list)
    opss_ser = []
    width = len(opss[0])
    for n, ops_raw in enumerate(copy.deepcopy(opss)):
        ops = copy.copy(ops_raw)
        # step 1, serialize target gates
        smt_ops_list = []
        if new_set_list is None:
            smt_ops = ['I'] * width
            for i, op in enumerate(ops):
                if 'dcz' in op:
                    smt_ops[i] = op
                    ops[i] = 'I'
            if not all([op in ['I', '_I_'] for op in smt_ops]):
                smt_ops_list.append(smt_ops)
        else:
            for ex_set in new_set_list:
                ex_set = set(ex_set)
                op_set_this_layer = set(ops)
                if ex_set <= op_set_this_layer:
                    smt_ops = ['I'] * width
                    for i, op in enumerate(ops):
                        if op in ex_set:
                            smt_ops[i] = op
                            ops[i] = 'I'
                    smt_ops_list.append(smt_ops)
        if len(smt_ops_list):
            if others_idx is not None:
                for i, op in enumerate(ops):
                    if op != 'I':
                        assert smt_ops_list[others_idx][i] == 'I'
                        smt_ops_list[others_idx][i] = op
            elif not all([op in ['I', '_I_'] for op in ops]):
                opss_ser.append(ops)
            opss_ser.extend(smt_ops_list)
        else:
            if any(['dcz' in op for op in ops]):
                for i, op in enumerate(ops):
                    if 'dcz' in op:
                        ops[i] = 'I'
                        smt_ops = ['I'] * width
                        smt_ops[i] = op
                        smt_ops[ops.index(op)] = op
                        ops[ops.index(op)] = 'I'
                        opss_ser.append(smt_ops)
            if not all([op in ['I', '_I_'] for op in ops]):
                opss_ser.append(ops)
    return opss_ser
